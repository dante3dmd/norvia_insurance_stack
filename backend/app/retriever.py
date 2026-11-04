from __future__ import annotations

import time
from contextlib import AbstractContextManager

from opentelemetry import trace
from opentelemetry.trace import Span, SpanKind
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import DocumentChunk
from app.observability import get_observability_logger, log_event
from app.providers.factory import get_embedding_provider

tracer = trace.get_tracer("norvia.retrieval")
logger = get_observability_logger()


class RetrievalSpan(AbstractContextManager):
    """OpenInference-style span that wraps vector retrieval and records query metadata for Phoenix."""

    def __init__(self, question: str, document_ids: list[str] | None = None, k: int = 5) -> None:
        """Store the retrieval request metadata before starting an OpenTelemetry span."""
        self.question = question
        self.document_ids = document_ids or []
        self.k = k
        self._span: Span | None = None

    def __enter__(self):
        """Start the instrumentation span and expose the context manager for the query."""
        self._span = tracer.start_span(
            "RetrievalSpan",
            kind=SpanKind.INTERNAL,
            attributes={
                "openinference.span.kind": "retrieval",
                "retrieval.query_chars": len(self.question),
                "retrieval.top_k": self.k,
                "retrieval.filter_count": len(self.document_ids),
            },
        )
        self._span.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Close the span even if the retrieval fails so traces remain coherent."""
        if self._span is not None:
            self._span.__exit__(exc_type, exc_val, exc_tb)

    def add_documents(self, documents: list[dict[str, object]]) -> None:
        """Record result counts and aggregate scores without storing document text or identifiers."""
        if self._span is None:
            return
        scores: list[float] = []
        for item in documents:
            score = item.get("score")
            if isinstance(score, (int, float)):
                scores.append(float(score))
        self._span.set_attribute("retrieval.result_count", len(documents))
        self._span.set_attribute("retrieval.score_max", max(scores, default=0.0))
        self._span.set_attribute("retrieval.score_avg", sum(scores) / len(scores) if scores else 0.0)

    def set_attribute(self, key: str, value: str | bool | int | float) -> None:
        """Set a non-sensitive metric on the active retrieval span when one is available."""
        if self._span is not None:
            self._span.set_attribute(key, value)


async def retrieve_chunks(
    session: AsyncSession,
    question: str,
    document_ids: list[str] | None = None,
    k: int = 5,
) -> list[dict[str, object]]:
    """Query pgvector with cosine distance and return the top fragments with similarity scores."""
    retrieval_started = time.perf_counter()
    with RetrievalSpan(question=question, document_ids=document_ids, k=k) as retrieval_span:
        embedding_started = time.perf_counter()
        embedding_provider = get_embedding_provider()
        query_vector = embedding_provider.embed(question)
        embedding_ms = (time.perf_counter() - embedding_started) * 1000
        retrieval_span.set_attribute("retrieval.embedding_duration_ms", embedding_ms)

        stmt = select(
            DocumentChunk,
            DocumentChunk.embedding.cosine_distance(query_vector).label("distance"),
        )
        if document_ids:
            stmt = stmt.where(DocumentChunk.document_id.in_(document_ids))
        stmt = stmt.order_by(DocumentChunk.embedding.cosine_distance(query_vector)).limit(k)

        database_started = time.perf_counter()
        result = await session.execute(stmt)
        database_ms = (time.perf_counter() - database_started) * 1000
        retrieval_span.set_attribute("retrieval.database_duration_ms", database_ms)
        chunks: list[dict[str, object]] = []
        for chunk, distance in result.all():
            similarity = max(0.0, min(1.0, 1.0 - float(distance)))
            chunk_payload = {
                "document_id": chunk.document_id,
                "chunk": chunk.text,
                "page": chunk.page,
                "score": round(similarity, 6),
            }
            chunks.append(chunk_payload)

        retrieval_span.add_documents(chunks)
        duration_ms = (time.perf_counter() - retrieval_started) * 1000
        retrieval_span.set_attribute("retrieval.duration_ms", duration_ms)
        log_event(
            logger,
            "retrieval.completed",
            duration_ms=round(duration_ms, 3),
            embedding_duration_ms=round(embedding_ms, 3),
            database_duration_ms=round(database_ms, 3),
            result_count=len(chunks),
            top_k=k,
        )
        return chunks
