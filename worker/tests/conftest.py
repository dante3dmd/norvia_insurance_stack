import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
WORKER = ROOT / "worker"
for path in (str(ROOT), str(BACKEND), str(WORKER)):
    if path not in sys.path:
        sys.path.insert(0, path)
