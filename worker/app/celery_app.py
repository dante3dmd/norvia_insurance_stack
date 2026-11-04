import os
import sys
from pathlib import Path

from celery import Celery
from celery.signals import worker_init
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


def setup_worker_telemetry(**_signal_kwargs) -> TracerProvider:
    """Export worker spans to Phoenix with a distinct service identity for distributed traces."""
    provider = TracerProvider(
        resource=Resource.create({"service.name": "norvia-worker", "service.namespace": "norvia"})
    )
    exporter = OTLPSpanExporter(
        endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317"),
        insecure=True,
    )
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    return provider


worker_init.connect(setup_worker_telemetry, weak=False)

# Celery acts as the async ingest worker; Redis is the broker and result backend hosting job dispatch and retry coordination.
celery_app = Celery(
    "norvia_worker",
    broker=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
    backend=os.getenv("REDIS_RESULT_BACKEND", os.getenv("REDIS_URL", "redis://localhost:6379/0")),
    include=["worker.app.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_time_limit=600,
    task_soft_time_limit=300,
    worker_prefetch_multiplier=1,
    broker_connection_retry_on_startup=True,
)
