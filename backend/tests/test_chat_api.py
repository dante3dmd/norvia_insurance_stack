from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.schemas import ChatRequest, ChatResponse


def test_context_window_is_at_least_32k():
    assert settings.CONTEXT_WINDOW_TOKENS >= 32768


def test_chat_request_supports_max_fragments_alias():
    payload = ChatRequest.model_validate({"question": "¿Qué cubre la póliza?", "max_fragments": 4})
    assert payload.k == 4
    assert payload.question == "¿Qué cubre la póliza?"


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
