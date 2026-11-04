import pytest
from pgvector.sqlalchemy import Vector
from sqlalchemy import inspect

from app.db.base import Base
from app.db.models import Document, DocumentChunk, DocumentStatus


@pytest.mark.asyncio
async def test_database_schema_exposes_expected_tables_and_fields():
    assert "documents" in Base.metadata.tables
    assert "document_chunks" in Base.metadata.tables

    documents = Base.metadata.tables["documents"]
    document_chunks = Base.metadata.tables["document_chunks"]

    assert "filename" in documents.columns
    assert "checksum" in documents.columns
    assert "status" in documents.columns
    assert "file_size" in documents.columns
    assert "error_message" in documents.columns

    assert "document_id" in document_chunks.columns
    assert "text" in document_chunks.columns
    assert "page" in document_chunks.columns
    assert "offset" in document_chunks.columns
    assert "embedding" in document_chunks.columns
    assert "metadata" in document_chunks.columns

    assert document_chunks.columns["embedding"].type.__class__ is Vector
    assert inspect(DocumentChunk).attrs["embedding"].columns[0].nullable is False

    assert Document.status.property.columns[0].type.python_type is DocumentStatus
    assert any(index.name == "ix_document_chunk_embedding_hnsw" for index in document_chunks.indexes)
