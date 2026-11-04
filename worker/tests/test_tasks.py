from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from celery.exceptions import Retry

import worker.app.tasks as tasks


def test_chunk_text_splits_content_and_keeps_metadata():
    content = "palabra " * 600
    chunks = tasks.chunk_text(content, chunk_size=120, chunk_overlap=20, source_type="txt")

    assert len(chunks) >= 2
    assert all("text" in chunk for chunk in chunks)
    assert all("offset" in chunk for chunk in chunks)
    assert all("page" in chunk for chunk in chunks)
    assert chunks[0]["offset"] == 0


def test_chunk_uses_markdown_parser_for_markdown_files(monkeypatch):
    calls = []

    class FakeMarkdownNodeParser:
        def get_nodes_from_documents(self, documents):
            calls.append("markdown")
            return [type("Node", (), {"text": "# Cobertura\n\nLa póliza cubre daños."})()]

    monkeypatch.setattr(tasks, "MarkdownNodeParser", FakeMarkdownNodeParser)
    monkeypatch.setattr(tasks, "Document", type("Doc", (), {"__init__": lambda self, text=None: setattr(self, "text", text)}))

    chunks = tasks.chunk_text("# Cobertura\n\nLa póliza cubre daños.", source_type="md")

    assert calls == ["markdown"]
    assert chunks[0]["text"].startswith("# Cobertura")


def test_chunk_uses_hierarchical_and_sentence_splitter_for_text_files(monkeypatch):
    calls = []

    class FakeSentenceSplitter:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def split_text(self, text):
            calls.append("sentence")
            return ["Primera frase.", "Segunda frase."]

    class FakeHierarchicalNodeParser:
        @classmethod
        def from_defaults(cls, **kwargs):
            calls.append("hierarchical")
            return cls()

        def get_nodes_from_documents(self, documents):
            return [type("Node", (), {"text": "Primera frase."})(), type("Node", (), {"text": "Segunda frase."})()]

    monkeypatch.setattr(tasks, "SentenceSplitter", FakeSentenceSplitter)
    monkeypatch.setattr(tasks, "HierarchicalNodeParser", FakeHierarchicalNodeParser)
    monkeypatch.setattr(tasks, "Document", type("Doc", (), {"__init__": lambda self, text=None: setattr(self, "text", text)}))

    chunks = tasks.chunk_text("Primera frase. Segunda frase.", source_type="txt")

    assert "sentence" in calls and "hierarchical" in calls
    assert len(chunks) == 2


def test_extract_text_for_document_uses_pymupdf4llm_for_pdf(monkeypatch, tmp_path):
    pdf_path = tmp_path / "example.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    captured = {}

    def fake_to_markdown(path: str):
        captured["path"] = path
        return "# Cobertura\n\nLa póliza cubre daños."

    monkeypatch.setattr("pymupdf4llm.to_markdown", fake_to_markdown)

    result = tasks._extract_text_for_document("doc-1", "example.pdf", pdf_path.read_bytes())

    assert result.startswith("# Cobertura")
    assert captured["path"].lower().endswith(".pdf")


def test_corrupt_pdf_raises_a_permanent_error(monkeypatch, tmp_path):
    """Verify corrupt PDFs fail with a user-readable cause instead of entering retries."""
    monkeypatch.setattr(
        tasks,
        "pymupdf4llm",
        SimpleNamespace(to_markdown=Mock(side_effect=RuntimeError("invalid xref"))),
    )

    with pytest.raises(tasks.PermanentDocumentError, match="corrupto o protegido"):
        tasks._extract_pdf_text_safely(tmp_path / "corrupt.pdf")


def test_empty_document_is_marked_failed_with_a_cause(monkeypatch):
    """Ensure a PDF with no extractable text fails immediately and stores its reason."""
    document = SimpleNamespace(status=tasks.DocumentStatus.PENDING, checksum=None, filename="empty.pdf")
    set_status = AsyncMock()
    monkeypatch.setattr(tasks, "_fetch_document", AsyncMock(return_value=document))
    monkeypatch.setattr(tasks, "_set_document_status", set_status)
    monkeypatch.setattr(tasks, "read_document_file", lambda *_: b"empty pdf")
    monkeypatch.setattr(tasks, "_completed_checksum_exists", AsyncMock(return_value=False))
    monkeypatch.setattr(tasks, "_extract_text_for_document", lambda *_: "")

    with pytest.raises(tasks.PermanentDocumentError, match="vacío o no contiene texto"):
        tasks._process_document_task_impl("doc-empty")

    assert set_status.await_args_list[-1].args == (
        "doc-empty",
        tasks.DocumentStatus.FAILED,
        "El documento está vacío o no contiene texto extraíble.",
    )


def test_embedding_failure_uses_exponential_celery_retry(monkeypatch):
    """Check transient embedding failures are rescheduled by Celery with exponential delay."""
    document = SimpleNamespace(status=tasks.DocumentStatus.PENDING, checksum=None, filename="policy.pdf")
    set_status = AsyncMock()
    retry = Mock(side_effect=Retry())
    task = SimpleNamespace(request=SimpleNamespace(retries=2), max_retries=4, retry=retry)
    monkeypatch.setattr(tasks, "_fetch_document", AsyncMock(return_value=document))
    monkeypatch.setattr(tasks, "_set_document_status", set_status)
    monkeypatch.setattr(tasks, "read_document_file", lambda *_: b"pdf")
    monkeypatch.setattr(tasks, "_completed_checksum_exists", AsyncMock(return_value=False))
    monkeypatch.setattr(tasks, "_extract_text_for_document", lambda *_: "contenido")
    monkeypatch.setattr(tasks, "chunk_text", lambda *args, **kwargs: [{"text": "contenido"}])
    monkeypatch.setattr(
        tasks,
        "get_embedding_provider",
        lambda: SimpleNamespace(embed=Mock(side_effect=TimeoutError("embedding timeout"))),
    )

    with pytest.raises(Retry):
        tasks._process_document_task_impl("doc-retry", task=task)

    retry.assert_called_once()
    assert retry.call_args.kwargs["countdown"] == 4
    assert set_status.await_args_list[-1].args == (
        "doc-retry",
        tasks.DocumentStatus.PROCESSING,
        "El proveedor de embeddings falló: embedding timeout",
    )


def test_exhausted_embedding_retries_mark_document_failed(monkeypatch):
    """Persist the provider failure as the final document cause after Celery exhausts retries."""
    document = SimpleNamespace(status=tasks.DocumentStatus.PENDING, checksum=None, filename="policy.pdf")
    set_status = AsyncMock()
    retry = Mock()
    task = SimpleNamespace(request=SimpleNamespace(retries=4), max_retries=4, retry=retry)
    monkeypatch.setattr(tasks, "_fetch_document", AsyncMock(return_value=document))
    monkeypatch.setattr(tasks, "_set_document_status", set_status)
    monkeypatch.setattr(tasks, "read_document_file", lambda *_: b"pdf")
    monkeypatch.setattr(tasks, "_completed_checksum_exists", AsyncMock(return_value=False))
    monkeypatch.setattr(tasks, "_extract_text_for_document", lambda *_: "contenido")
    monkeypatch.setattr(tasks, "chunk_text", lambda *args, **kwargs: [{"text": "contenido"}])
    monkeypatch.setattr(
        tasks,
        "get_embedding_provider",
        lambda: SimpleNamespace(embed=Mock(side_effect=TimeoutError("embedding timeout"))),
    )

    with pytest.raises(RuntimeError, match="embedding timeout"):
        tasks._process_document_task_impl("doc-failed", task=task)

    retry.assert_not_called()
    assert set_status.await_args_list[-1].args == (
        "doc-failed",
        tasks.DocumentStatus.FAILED,
        "El proveedor de embeddings falló: embedding timeout",
    )
