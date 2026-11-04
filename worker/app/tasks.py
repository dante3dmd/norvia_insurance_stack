from __future__ import annotations

import asyncio
import hashlib
import os
import re
from pathlib import Path
from typing import Any

from celery import Task
from celery.exceptions import Retry
from sqlalchemy import select

try:
    import pymupdf4llm
except ModuleNotFoundError:  # pragma: no cover - dependency is installed in production runtime
    pymupdf4llm = None

try:
    from llama_index.core import Document
    from llama_index.core.node_parser import HierarchicalNodeParser, MarkdownNodeParser
    from llama_index.core.text_splitter import SentenceSplitter
except ModuleNotFoundError:  # pragma: no cover - optional for local/test fallback
    Document = None
    HierarchicalNodeParser = None
    MarkdownNodeParser = None
    SentenceSplitter = None

from app.db.models import Document as SqlAlchemyDocument, DocumentChunk, DocumentStatus
from app.db.session import AsyncSessionLocal
from app.providers.factory import get_embedding_provider
from app.storage import read_document_file
from worker.app.celery_app import celery_app


def _checksum_bytes(data: bytes) -> str:
    """Compute a SHA-256 checksum so identical uploads are rejected as duplicates."""
    return hashlib.sha256(data).hexdigest()


def _normalize_text(value: str) -> str:
    """Normalize embedded whitespace so chunking remains stable across text inputs."""
    return re.sub(r"\s+", " ", value or "").strip()


def _extract_pdf_text_safely(file_path: str | Path) -> str:
    """Convert the untrusted PDF to markdown so the data preserves document structure before chunking."""
    if pymupdf4llm is None:
        raise RuntimeError("PyMuPDF4LLM is required to convert PDFs to markdown before ingestion.")

    markdown_output = pymupdf4llm.to_markdown(str(file_path))
    if isinstance(markdown_output, dict):
        for key in ("markdown", "text", "content", "output"):
            value = markdown_output.get(key)
            if isinstance(value, str):
                markdown_output = value
                break
        else:
            markdown_output = "\n\n".join(str(value) for value in markdown_output.values())
    elif isinstance(markdown_output, list):
        markdown_output = "\n\n".join(
            str(item.get("text") if isinstance(item, dict) and "text" in item else item)
            for item in markdown_output
        )

    return _normalize_text(str(markdown_output))


def chunk_text(
    text: str,
    chunk_size: int = 600,
    chunk_overlap: int = 80,
    source_type: str = "txt",
) -> list[dict[str, Any]]:
    """Use LlamaIndex parsing rules that match the document type: markdown stays structured and plain text uses hierarchical splitting."""
    normalized = _normalize_text(text)
    if not normalized:
        return []

    source_name = (source_type or "txt").lower()

    if source_name in {"md", "markdown"} and Document is not None and MarkdownNodeParser is not None:
        doc = Document(text=normalized)
        parser = MarkdownNodeParser()
        nodes = parser.get_nodes_from_documents([doc])
        chunks: list[dict[str, Any]] = []
        for index, node in enumerate(nodes):
            node_text = _normalize_text(getattr(node, "text", str(node)))
            if not node_text:
                continue
            chunks.append(
                {
                    "text": node_text,
                    "offset": index * max(chunk_size - chunk_overlap, 1),
                    "page": index,
                    "metadata": {"source": "markdown_chunk", "chunk_index": index},
                }
            )
        if chunks:
            return chunks

    if source_name in {"txt", "text"} and Document is not None and HierarchicalNodeParser is not None and SentenceSplitter is not None:
        hierarchical_parser = HierarchicalNodeParser.from_defaults(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        sentence_splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        hierarchical_nodes = hierarchical_parser.get_nodes_from_documents([Document(text=normalized)])
        fragments = sentence_splitter.split_text(normalized)
        if hierarchical_nodes:
            fragments = [
                _normalize_text(getattr(node, "text", str(node))) for node in hierarchical_nodes if _normalize_text(getattr(node, "text", str(node)))
            ]

        chunks = []
        for index, fragment in enumerate(fragments):
            if not fragment:
                continue
            chunks.append(
                {
                    "text": fragment,
                    "offset": index * max(chunk_size - chunk_overlap, 1),
                    "page": index,
                    "metadata": {"source": "txt_chunk", "chunk_index": index},
                }
            )
        if chunks:
            return chunks

    chunks = []
    start = 0
    index = 0
    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        fragment = normalized[start:end]
        chunks.append(
            {
                "text": fragment,
                "offset": start,
                "page": index,
                "metadata": {"source": "text_chunk", "chunk_index": index},
            }
        )
        start += chunk_size - chunk_overlap
        index += 1
    return chunks


async def _set_document_status(document_id: str, status: DocumentStatus, error_message: str | None = None) -> None:
    """Persist the current document state so the API can show processing or failure states."""
    async with AsyncSessionLocal() as session:
        document = await session.get(SqlAlchemyDocument, document_id)
        if document is None:
            return
        document.status = status
        document.error_message = error_message
        await session.commit()


async def _fetch_document(document_id: str) -> Document | None:
    """Load a document record from PostgreSQL before starting the async work pipeline."""
    async with AsyncSessionLocal() as session:
        return await session.get(SqlAlchemyDocument, document_id)


async def _completed_checksum_exists(checksum: str) -> bool:
    """Skip already processed documents to keep reruns idempotent and avoid duplicate embeddings."""
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(SqlAlchemyDocument).where(
                SqlAlchemyDocument.checksum == checksum,
                SqlAlchemyDocument.status == DocumentStatus.COMPLETED,
            )
        )
        return result.scalar_one_or_none() is not None


async def _bulk_insert_chunks(document_id: str, chunks: list[dict[str, Any]]) -> None:
    """Store chunk records in bulk to keep the ingest pipeline fast and consistent with PostgreSQL."""
    if not chunks:
        return

    async with AsyncSessionLocal() as session:
        session.add_all(
            [
                DocumentChunk(
                    document_id=document_id,
                    text=item["text"],
                    page=item.get("page"),
                    offset=item.get("offset"),
                    embedding=item["embedding"],
                    metadata=item.get("metadata", {}),
                )
                for item in chunks
            ]
        )
        await session.commit()


def _extract_text_for_document(document_id: str, filename: str, payload: bytes) -> str:
    """Decode text from TXT/MD files directly and from PDFs via a safe reader for untrusted input."""
    suffix = Path(filename).suffix.lower()

    if suffix in {".txt", ".md"}:
        try:
            content = payload.decode("utf-8")
        except UnicodeDecodeError:
            content = payload.decode("utf-8", errors="replace")
        return content if suffix == ".md" else _normalize_text(content)

    if suffix == ".pdf":
        temp_path = Path(os.getenv("UPLOAD_DIR", "/tmp/norvia_uploads")) / f"{document_id}.pdf"
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path.write_bytes(payload)
        return _extract_pdf_text_safely(temp_path)

    raise ValueError(f"Unsupported document type for ingestion: {filename}")


@celery_app.task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 4})
def process_document_task(self: Task, document_id: str) -> str:
    """Celery task that handles document extraction, chunking, embeddings, and persistence with retries."""
    return _process_document_task_impl(document_id)


def _process_document_task_impl(document_id: str) -> str:
    """Execute the ingest flow and record a failure state when processing collapses."""
    try:
        document = asyncio.run(_fetch_document(document_id))
        if document is None:
            raise FileNotFoundError(f"Document {document_id} not found")

        if document.status == DocumentStatus.COMPLETED and document.checksum:
            return f"Document {document_id} already processed"

        asyncio.run(_set_document_status(document_id, DocumentStatus.PROCESSING))

        file_bytes = read_document_file(document_id, document.filename)
        checksum = _checksum_bytes(file_bytes)

        if asyncio.run(_completed_checksum_exists(checksum)):
            asyncio.run(_set_document_status(document_id, DocumentStatus.COMPLETED))
            return f"Document {document_id} skipped: duplicate checksum already completed"

        text = _extract_text_for_document(document_id, document.filename, file_bytes)
        chunks = chunk_text(text)
        if not chunks:
            raise ValueError("No extractable text was found in the uploaded document.")

        embedding_provider = get_embedding_provider()
        embedded_chunks: list[dict[str, Any]] = []
        for chunk in chunks:
            try:
                vector = embedding_provider.embed(chunk["text"])
            except Exception as exc:
                raise Retry() from exc
            embedded_chunks.append({**chunk, "embedding": vector})

        asyncio.run(_bulk_insert_chunks(document_id, embedded_chunks))
        asyncio.run(_set_document_status(document_id, DocumentStatus.COMPLETED))
        return f"Processed document {document_id} with {len(embedded_chunks)} chunks"
    except Retry:
        raise
    except Exception as exc:
        asyncio.run(_set_document_status(document_id, DocumentStatus.FAILED, str(exc)))
        raise Retry() from exc
