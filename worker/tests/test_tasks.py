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
