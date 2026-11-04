import json
import logging
from types import SimpleNamespace

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

from app.observability import StructuredLogFormatter
from app.retriever import RetrievalSpan


def test_structured_log_includes_active_trace_ids_without_message_content():
    """Verify JSON logs include trace correlation while omitting arbitrary message text."""
    provider = TracerProvider()
    tracer = provider.get_tracer("test")
    record = logging.LogRecord("norvia", logging.INFO, "", 0, "safe event", (), None)
    record.event_name = "safe.event"
    record.safe_fields = {"result_count": 2}

    with tracer.start_as_current_span("request") as span:
        formatted = StructuredLogFormatter().format(record)
        payload = json.loads(formatted)

    assert payload["trace_id"] == f"{span.get_span_context().trace_id:032x}"
    assert payload["span_id"] == f"{span.get_span_context().span_id:016x}"
    assert payload["event"] == "safe.event"
    assert payload["result_count"] == 2
    assert "safe event" not in formatted
    provider.shutdown()


def test_retrieval_span_records_only_aggregate_results(monkeypatch):
    """Ensure retrieval telemetry stores numeric aggregates instead of sensitive query and chunk data."""
    class CapturedSpan:
        """Capture span attributes without creating an exported trace during the unit test."""

        def __init__(self, attributes=None):
            """Initialize the fake span with attributes supplied at creation."""
            self.attributes = dict(attributes or {})

        def __enter__(self):
            """Return this fake span as the active context manager."""
            return self

        def __exit__(self, *_args):
            """Allow the instrumented block to close without side effects."""
            return None

        def set_attribute(self, key, value):
            """Store the attribute so the test can inspect exported telemetry."""
            self.attributes[key] = value

    captured = CapturedSpan()
    monkeypatch.setattr(
        "app.retriever.tracer",
        SimpleNamespace(start_span=lambda *args, **kwargs: CapturedSpan(kwargs.get("attributes"))),
    )

    with RetrievalSpan("private question", ["private-document-id"], k=4) as retrieval_span:
        retrieval_span.add_documents(
            [{"document_id": "private-document-id", "chunk": "private policy text", "score": 0.9}]
        )

    attributes = retrieval_span._span.attributes
    assert attributes["retrieval.query_chars"] == len("private question")
    assert attributes["retrieval.filter_count"] == 1
    assert attributes["retrieval.result_count"] == 1
    assert attributes["retrieval.score_max"] == 0.9
    assert attributes["retrieval.score_avg"] == 0.9
    assert not any("private" in str(value) for value in attributes.values())
    assert "retrieval.query" not in attributes
    assert "retrieval.documents" not in attributes