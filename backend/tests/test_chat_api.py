from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import _build_chat_prompt, app
from app.schemas import ChatRequest, ChatResponse


def test_context_window_is_at_least_32k():
    assert settings.CONTEXT_WINDOW_TOKENS >= 32768


def test_chat_request_supports_max_fragments_alias():
    payload = ChatRequest.model_validate({"question": "¿Qué cubre la póliza?", "max_fragments": 4})
    assert payload.k == 4
    assert payload.question == "¿Qué cubre la póliza?"


def test_build_chat_prompt_includes_question_and_context():
    prompt = _build_chat_prompt("¿Qué cubre la póliza?", "Fragmento relevante sobre daños materiales.")

    assert "¿Qué cubre la póliza?" in prompt
    assert "Fragmento relevante sobre daños materiales." in prompt
    assert "answer, citations, has_sufficient_evidence" in prompt


def test_chat_endpoint_returns_structured_response(monkeypatch):
    async def fake_retrieve(*args, **kwargs):
        return [
            {
                "document_id": "doc-1",
                "chunk": "La póliza cubre daños materiales.",
                "page": 3,
                "score": 0.92,
            }
        ]

    def fake_llm_generate(prompt: str, **kwargs):
        return '{"answer": "La póliza cubre daños materiales.", "citations": [{"document_id": "doc-1", "chunk": "La póliza cubre daños materiales.", "page": 3, "score": 0.92}], "has_sufficient_evidence": true}'

    monkeypatch.setattr("app.main.retrieve_chunks", fake_retrieve)
    monkeypatch.setattr("app.main.get_llm_provider", lambda: type("Llm", (), {"generate": staticmethod(fake_llm_generate)})())

    client = TestClient(app)
    response = client.post("/api/v1/chat", json={"question": "¿Qué cubre la póliza?", "k": 2})

    assert response.status_code == 200
    data = response.json()
    assert data["has_sufficient_evidence"] is True
    assert data["answer"]
    assert data["citations"][0]["document_id"] == "doc-1"
    ChatResponse.model_validate(data)


@pytest.mark.parametrize(
    ("provider_error", "expected_status", "expected_code"),
    [
        (httpx.TimeoutException("provider timeout"), 504, "llm_timeout"),
        (httpx.ConnectError("connection refused"), 502, "llm_provider_error"),
        (RuntimeError("unexpected provider failure"), 502, "llm_provider_error"),
    ],
)
def test_chat_endpoint_returns_structured_provider_errors(
    monkeypatch,
    provider_error,
    expected_status,
    expected_code,
):
    """Keep provider outages and timeouts in a stable API error envelope."""
    monkeypatch.setattr(
        "app.main.retrieve_chunks",
        AsyncMock(return_value=[{"document_id": "doc-1", "chunk": "Cobertura relevante.", "page": 1, "score": 0.9}]),
    )
    failing_llm = SimpleNamespace(generate=Mock(side_effect=provider_error))
    monkeypatch.setattr("app.main.get_llm_provider", lambda: failing_llm)

    response = TestClient(app).post("/api/v1/chat", json={"question": "¿Qué cubre?"})

    assert response.status_code == expected_status
    assert response.json()["detail"]["code"] == expected_code
    assert response.json()["detail"]["message"]


def test_fake_provider_end_to_end_upload_and_chat_flow(monkeypatch):
    class FakeResult:
        def __init__(self, rows):
            self._rows = rows

        def scalar_one_or_none(self):
            return self._rows[0] if len(self._rows) == 1 else None

        def scalars(self):
            scalar_result = type(
                "Scalars",
                (),
                {
                    "__init__": lambda self, rows: setattr(self, "_rows", rows),
                    "all": lambda self: self._rows,
                },
            )
            return scalar_result(self._rows)

    class FakeSession:
        documents = []

        def __init__(self):
            self.documents = self.__class__.documents

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def execute(self, stmt):
            return FakeResult(self.documents)

        async def get(self, model, document_id):
            for doc in self.documents:
                if doc.id == document_id:
                    return doc
            return None

        def add(self, document):
            self.documents.append(document)

        async def commit(self):
            return None

        async def refresh(self, document):
            if not document.id:
                document.id = "doc-123"

    async def fake_allow(*args, **kwargs):
        return True

    async def fake_retrieve(**kwargs):
        return [{"document_id": "doc-1", "chunk": "La póliza cubre daños materiales.", "page": 3, "score": 0.92}]

    monkeypatch.setattr("app.main.AsyncSessionLocal", lambda: FakeSession())
    monkeypatch.setattr("app.main.save_document_file", lambda *args, **kwargs: None)
    monkeypatch.setattr("app.main.rate_limiter.allow", fake_allow)
    monkeypatch.setattr("app.main.process_document_task", type("TaskStub", (), {"delay": staticmethod(lambda document_id: document_id)})())
    monkeypatch.setattr("app.main.retrieve_chunks", fake_retrieve)
    monkeypatch.setattr("app.main.get_llm_provider", lambda: type("Llm", (), {"generate": staticmethod(lambda prompt, **kwargs: '{"answer": "La póliza cubre daños materiales.", "citations": [{"document_id": "doc-1", "chunk": "La póliza cubre daños materiales.", "page": 3, "score": 0.92}], "has_sufficient_evidence": true}')})())

    client = TestClient(app)
    upload_response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("policy.pdf", b"%PDF-1.4", "application/pdf")},
    )
    assert upload_response.status_code == 200

    chat_response = client.post("/api/v1/chat", json={"question": "¿Qué cubre la póliza?", "k": 2})
    assert chat_response.status_code == 200
    payload = chat_response.json()
    assert payload["has_sufficient_evidence"] is True
    assert payload["citations"][0]["document_id"] == "doc-1"


def test_document_upload_and_status_endpoints(monkeypatch):
    class FakeResult:
        def __init__(self, rows):
            self._rows = rows

        def scalar_one_or_none(self):
            return self._rows[0] if len(self._rows) == 1 else None

        def scalars(self):
            scalar_result = type(
                "Scalars",
                (),
                {
                    "__init__": lambda self, rows: setattr(self, "_rows", rows),
                    "all": lambda self: self._rows,
                },
            )
            return scalar_result(self._rows)

    class FakeSession:
        documents = []

        def __init__(self):
            self.documents = self.__class__.documents

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def execute(self, stmt):
            return FakeResult(self.documents)

        async def get(self, model, document_id):
            for doc in self.documents:
                if doc.id == document_id:
                    return doc
            return None

        def add(self, document):
            self.documents.append(document)

        async def commit(self):
            return None

        async def refresh(self, document):
            if not document.id:
                document.id = "doc-123"

    async def fake_allow(*args, **kwargs):
        return True

    monkeypatch.setattr("app.main.AsyncSessionLocal", lambda: FakeSession())
    monkeypatch.setattr("app.main.save_document_file", lambda *args, **kwargs: None)
    monkeypatch.setattr("app.main.rate_limiter.allow", fake_allow)
    monkeypatch.setattr(
        "app.main.process_document_task",
        type("TaskStub", (), {"delay": staticmethod(lambda document_id: document_id)})(),
    )

    client = TestClient(app)
    upload_response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("policy.pdf", b"%PDF-1.4", "application/pdf")},
    )

    assert upload_response.status_code == 200
    payload = upload_response.json()
    assert payload["status"] == "pending"
    assert payload["filename"] == "policy.pdf"

    status_response = client.get(f"/api/v1/documents/{payload['id']}/status")
    assert status_response.status_code == 200
    assert status_response.json()["status"] == "pending"

    list_response = client.get("/api/v1/documents")
    assert list_response.status_code == 200
    assert len(list_response.json()) >= 1
