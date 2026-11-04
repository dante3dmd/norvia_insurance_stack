import os

import pytest
from sqlalchemy import select, text

from app.config import settings
from app.db.base import Base
from app.db.models import Document, DocumentChunk
from app.db.session import AsyncSessionLocal, engine, init_db


@pytest.mark.asyncio
async def test_real_pgvector_integration_smoke():
    postgres_url = os.getenv("PGVECTOR_TEST_URL") or settings.DATABASE_URL

    if not postgres_url:
        pytest.skip("No PostgreSQL pgvector URL configured for integration test.")

    try:
        async with engine.begin() as connection:
            await connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await connection.run_sync(Base.metadata.create_all)
    except Exception as exc:  # pragma: no cover - integration environment only
        pytest.skip(f"pgvector integration test unavailable: {exc}")

    async with AsyncSessionLocal() as session:
        document = Document(
            filename="policy.pdf",
            content_type="application/pdf",
            checksum="integration-checksum",
            file_size=123,
        )
        session.add(document)
        await session.commit()
        await session.refresh(document)

        chunk = DocumentChunk(
            document_id=document.id,
            text="La póliza cubre daños materiales.",
            page=1,
            offset=0,
            embedding=[1.0] + [0.0] * 1535,
            metadata_={"source": "integration_test"},
        )
        session.add(chunk)
        await session.commit()

        result = await session.execute(
            select(DocumentChunk).where(DocumentChunk.document_id == document.id).order_by(DocumentChunk.id)
        )
        stored = result.scalar_one()
        assert stored.text == "La póliza cubre daños materiales."
        assert stored.page == 1
        assert stored.metadata_["source"] == "integration_test"
        assert len(stored.embedding) == 1536

        nearest = await session.execute(
            select(DocumentChunk, DocumentChunk.embedding.cosine_distance([1.0] + [0.0] * 1535).label("distance")).where(
                DocumentChunk.document_id == document.id
            ).order_by(DocumentChunk.embedding.cosine_distance([1.0] + [0.0] * 1535))
        )
        match = nearest.first()
        assert match is not None
        assert match[1] == pytest.approx(0.0, abs=1e-9)
