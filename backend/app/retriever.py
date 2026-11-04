from __future__ import annotations

import json
from contextlib import AbstractContextManager

from opentelemetry import trace
from opentelemetry.trace import SpanKind
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import DocumentChunk
from app.providers.factory import get_embedding_provider

tracer = trace.get_tracer("norvia.retrieval")


class RetrievalSpan(AbstractContextManager):
    """OpenInference-style span that wraps vector retrieval and records query metadata for Phoenix."""

    def __init__(self, question: str, document_ids: list[str] | None = None, k: int = 5) -> None:
        """Store the retrieval request metadata before starting an OpenTelemetry span."""
        self.question = question
        self.document_ids = document_ids or []
        self.k = k
        self._span = None

    def __enter__(self):
        """Start the instrumentation span and expose the context manager for the query."""
        self._span = tracer.start_span(
            "RetrievalSpan",
            kind=SpanKind.INTERNAL,
            attributes={
                "openinference.span.kind": "retrieval",
                "retrieval.query": self.question,
                "retrieval.top_k": self.k,
                "retrieval.document_ids": json.dumps(self.document_ids),
            },
        )
        self._span.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Close the span even if the retrieval fails so traces remain coherent."""
        if self._span is not None:
            self._span.__exit__(exc_type, exc_val, exc_tb)

    def add_documents(self, documents: list[dict[str, object]]) -> None:
        """Attach the final matched chunks to the trace as evidence for the response."""
        if self._span is None:
            return
        self._span.set_attribute("retrieval.documents", json.dumps(documents))


async def retrieve_chunks(
    session: AsyncSession,
    question: str,
    document_ids: list[str] | None = None,
    k: int = 5,
) -> list[dict[str, object]]:
    """Query pgvector with cosine distance and return the top fragments with similarity scores."""
    embedding_provider = get_embedding_provider()
    query_vector = embedding_provider.embed(question)

    with RetrievalSpan(question=question, document_ids=document_ids, k=k) as retrieval_span:
        stmt = select(
            DocumentChunk,
            DocumentChunk.embedding.cosine_distance(query_vector).label("distance"),
        )
        if document_ids:
            stmt = stmt.where(DocumentChunk.document_id.in_(document_ids))
        stmt = stmt.order_by(DocumentChunk.embedding.cosine_distance(query_vector)).limit(k)

        result = await session.execute(stmt)
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
        return chunks
