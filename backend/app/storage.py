from __future__ import annotations

import os
from pathlib import Path

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "/tmp/norvia_uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def get_document_file_path(document_id: str, filename: str) -> Path:
    """Return the local path where the uploaded file is stored for a document id."""
    safe_name = Path(filename).name or f"{document_id}.bin"
    return UPLOAD_DIR / f"{document_id}_{safe_name}"


def save_document_file(document_id: str, filename: str, file_bytes: bytes) -> Path:
    """Persist the uploaded file on disk so the worker can process it later."""
    target = get_document_file_path(document_id, filename)
    target.write_bytes(file_bytes)
    return target


def read_document_file(document_id: str, filename: str) -> bytes:
    """Read the source document bytes from disk for the async ingestion pipeline."""
    target = get_document_file_path(document_id, filename)
    if not target.exists():
        raise FileNotFoundError(f"Document file not found for document_id={document_id}")
    return target.read_bytes()
