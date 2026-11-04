from __future__ import annotations

import json
import re

from fastapi import FastAPI, Header, HTTPException, Request
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.session import AsyncSessionLocal, init_db
from app.providers.factory import get_llm_provider
from app.rate_limit import extract_user_id, rate_limiter
from app.retriever import retrieve_chunks
from app.schemas import ChatRequest, ChatResponse, Citation


def setup_telemetry() -> TracerProvider|None:
    """Initialize the OpenTelemetry exporter for Phoenix so traces flow from API requests into the observability backend."""
    resource = Resource.create({"service.name": settings.OTEL_SERVICE_NAME, "service.namespace": "norvia"})
    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(
        endpoint=settings.OTEL_EXPORTER_OTLP_ENDPOINT,
        insecure=True,
    )
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    return provider


app = FastAPI(title="Norvia Insurance API", version="0.1.0")
setup_telemetry()
FastAPIInstrumentor.instrument_app(app)


@app.on_event("startup")
async def startup_event() -> None:
    """Create the database schema and make sure the backend is ready for vector queries on startup."""
    await init_db()


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    """Simple liveness endpoint for container orchestration and smoke checks."""
    return {"status": "ok"}


def _trim_context_to_window(context: str, max_tokens: int = settings.CONTEXT_WINDOW_TOKENS) -> str:
    """Respect the minimum 32K context window required by the architecture decision for safe RAG prompting."""
    words = context.split()
    if len(words) <= max_tokens:
        return context
    return " ".join(words[:max_tokens])


def _build_rejection_response(citations: list[Citation], question: str) -> ChatResponse:
    """Return a clear refusal when the retrieval score or evidence is insufficient."""
    return ChatResponse(
        answer=(
            "No tengo suficiente evidencia en los documentos disponibles para responder con seguridad. "
            "Por favor, proporcione más contexto o un documento relevante."
        ),
        citations=citations,
        has_sufficient_evidence=False,
    )


def _parse_llm_response(raw_response: str, fallback_citations: list[Citation]) -> ChatResponse:
    """Normalize the model output into the strict Pydantic schema expected by the API contract."""
    cleaned = raw_response.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    if not cleaned:
        return _build_rejection_response(fallback_citations, "")

    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match is None:
            return ChatResponse(
                answer=cleaned,
                citations=fallback_citations,
                has_sufficient_evidence=False,
            )
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return ChatResponse(
                answer=cleaned,
                citations=fallback_citations,
                has_sufficient_evidence=False,
            )

    answer = str(payload.get("answer") or "No tengo suficiente evidencia para responder con seguridad.")
    citations_payload = payload.get("citations") or []
    citations = [Citation.model_validate(item) for item in citations_payload]
    has_sufficient_evidence = bool(payload.get("has_sufficient_evidence"))

    if not citations:
        citations = fallback_citations

    if has_sufficient_evidence and answer.strip() and citations:
        return ChatResponse(answer=answer, citations=citations, has_sufficient_evidence=True)

    return _build_rejection_response(citations or fallback_citations, answer)


@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    request: Request,
    authorization: str | None = Header(default=None, alias="Authorization"),
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
) -> ChatResponse:
    """Return a structured answer grounded in vector search results with explicit rejection when evidence is weak."""
    user_id = extract_user_id(authorization, x_user_id)
    if not await rate_limiter.allow(user_id):
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Please retry later.")

    async with AsyncSessionLocal() as session:
        hits = await retrieve_chunks(session=session, question=payload.question, document_ids=payload.document_ids, k=payload.k)

    citations = [Citation(**hit) for hit in hits]
    if not citations:
        return _build_rejection_response([], payload.question)

    max_score = max((citation.score for citation in citations), default=0.0)
    if max_score < 0.35:
        return _build_rejection_response(citations, payload.question)

    context = "\n\n".join(
        f"[{index + 1}] {hit['chunk']}\nDocumento: {hit['document_id']} | Página: {hit['page']} | Score: {hit['score']}"
        for index, hit in enumerate(hits)
    )
    context = _trim_context_to_window(context, settings.CONTEXT_WINDOW_TOKENS)

    llm = get_llm_provider()
    prompt = (
        "Responde ÚNICAMENTE con un JSON válido con estas claves: answer, citations, has_sufficient_evidence. "
        "Si no hay evidencia suficiente, responde con has_sufficient_evidence=false y un texto de rechazo explícito. "
        "Nunca inventes una respuesta. Usa el siguiente contexto para fundamentar tu respuesta:\n\n"
        f"{context}\n\nPregunta del usuario: {payload.question}"
    )
    raw_response = llm.generate(prompt)
    return _parse_llm_response(raw_response, citations)
