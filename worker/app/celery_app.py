import os
import sys
from pathlib import Path

from celery import Celery

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

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
