from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from opentelemetry import trace


class StructuredLogFormatter(logging.Formatter):
    """Format application events as JSON with identifiers from the active trace context."""

    def format(self, record: logging.LogRecord) -> str:
        """Serialize the log event and safe structured fields without formatting arbitrary extras."""
        span_context = trace.get_current_span().get_span_context()
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": getattr(record, "event_name", record.getMessage()),
            "trace_id": f"{span_context.trace_id:032x}" if span_context.is_valid else None,
            "span_id": f"{span_context.span_id:016x}" if span_context.is_valid else None,
        }
        payload.update(getattr(record, "safe_fields", {}))
        return json.dumps(payload, ensure_ascii=False, default=str)


def get_observability_logger() -> logging.Logger:
    """Return the configured JSON application logger without changing root logging behavior."""
    logger = logging.getLogger("norvia")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not any(getattr(handler, "_norvia_json_handler", False) for handler in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(StructuredLogFormatter())
        handler._norvia_json_handler = True  # type: ignore[attr-defined]
        logger.addHandler(handler)
    return logger


def log_event(logger: logging.Logger, event: str, **safe_fields: Any) -> None:
    """Write a JSON event with explicitly selected fields and current trace correlation identifiers."""
    logger.info(event, extra={"event_name": event, "safe_fields": safe_fields})