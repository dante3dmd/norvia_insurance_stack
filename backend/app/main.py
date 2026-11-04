from __future__ import annotations

import hashlib
import json
import re

from fastapi import FastAPI, File, Header, HTTPException, Request, UploadFile
from opentelemetry import propagate, trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import SpanKind
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.models import Document, DocumentStatus
from app.db.session import AsyncSessionLocal, init_db
from app.providers.factory import get_llm_provider
from app.rate_limit import extract_user_id, rate_limiter
from app.retriever import retrieve_chunks
from app.schemas import (
    ChatRequest,
    ChatResponse,
    Citation,
    DocumentStatusResponse,
    DocumentSummary,
    DocumentUploadResponse,
)
from app.storage import save_document_file
from worker.app.tasks import process_document_task


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


tracer = trace.get_tracer("norvia.api")


def _build_celery_headers() -> dict[str, str]:
    """Inject the current tracing context so the async worker continues the same distributed trace."""
    carrier: dict[str, str] = {}
    propagate.inject(carrier)
    return carrier


def _enqueue_document_task(document_id: str) -> None:
    """Queue the Celery ingestion job while remaining compatible with the test stubs that use delay()."""
    if hasattr(process_document_task, "apply_async"):
        process_document_task.apply_async(args=[document_id], headers=_build_celery_headers())
        return
    process_document_task.delay(document_id)


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


@app.post("/api/v1/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    authorization: str | None = Header(default=None, alias="Authorization"),
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
) -> DocumentUploadResponse:
    """Accept a document upload, persist it on disk, and enqueue the async ingest task."""
    user_id = extract_user_id(authorization, x_user_id)
    if not await rate_limiter.allow(user_id):
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Please retry later.")

    file_bytes = await file.read()
    filename = file.filename or "document.bin"
    checksum = hashlib.sha256(file_bytes).hexdigest()
    content_type = file.content_type or "application/octet-stream"

    async with AsyncSessionLocal() as session:
        existing = await session.execute(select(Document).where(Document.checksum == checksum))
        existing_document = existing.scalar_one_or_none()
        if existing_document is not None:
            return DocumentUploadResponse(
                id=existing_document.id,
                filename=existing_document.filename,
                content_type=existing_document.content_type,
                status=existing_document.status.value,
                checksum=existing_document.checksum,
                file_size=existing_document.file_size,
                message="Document already exists and is queued or processed.",
            )

        document = Document(
            filename=filename,
            content_type=content_type,
            checksum=checksum,
            file_size=len(file_bytes),
            status=DocumentStatus.PENDING,
        )
        session.add(document)
        await session.commit()
        await session.refresh(document)

    save_document_file(document.id, document.filename, file_bytes)
    _enqueue_document_task(document.id)
    return DocumentUploadResponse(
        id=document.id,
        filename=document.filename,
        content_type=document.content_type,
        status=document.status.value,
        checksum=document.checksum,
        file_size=document.file_size,
        message="Document accepted for ingestion.",
    )


@app.get("/api/v1/documents", response_model=list[DocumentSummary])
async def list_documents() -> list[DocumentSummary]:
    """List tracked documents and their current ingest states for the platform."""
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Document).order_by(Document.created_at.desc()))
        documents = result.scalars().all()

    return [
        DocumentSummary(
            id=document.id,
            filename=document.filename,
            content_type=document.content_type,
            status=document.status.value,
            checksum=document.checksum,
            file_size=document.file_size,
            error_message=document.error_message,
            created_at=document.created_at.isoformat() if document.created_at else None,
            updated_at=document.updated_at.isoformat() if document.updated_at else None,
        )
        for document in documents
    ]


@app.get("/api/v1/documents/{document_id}", response_model=DocumentStatusResponse)
async def get_document_detail(document_id: str) -> DocumentStatusResponse:
    """Fetch the current lifecycle status for one specific ingested document."""
    async with AsyncSessionLocal() as session:
        document = await session.get(Document, document_id)
        if document is None:
            raise HTTPException(status_code=404, detail="Document not found")
        return DocumentStatusResponse(
            id=document.id,
            filename=document.filename,
            status=document.status.value,
            error_message=document.error_message,
        )


@app.get("/api/v1/documents/{document_id}/status", response_model=DocumentStatusResponse)
async def get_document_status(document_id: str) -> DocumentStatusResponse:
    """Return the document status and any ingest error message for monitoring and UI polling."""
    return await get_document_detail(document_id)


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
    with tracer.start_as_current_span("llm.generate", kind=SpanKind.CLIENT) as span:
        span.set_attribute("llm.provider", type(llm).__name__)
        span.set_attribute("llm.prompt_chars", len(prompt))
        raw_response = llm.generate(prompt)
    return _parse_llm_response(raw_response, citations)
