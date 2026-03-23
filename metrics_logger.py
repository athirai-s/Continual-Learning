import json
import time
from pathlib import Path
from typing import Any

from run_artifacts import metrics_root


METRICS_EVENTS_FILENAME = "events.jsonl"
METRICS_SCHEMA_VERSION = 1


class MetricsLogger:
    def __init__(self, run_root: str | Path):
        self.run_root = Path(run_root)
        self.events_path = metrics_root(run_root) / METRICS_EVENTS_FILENAME
        self.events_path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event_type: str, **fields: Any) -> None:
        payload = {
            "schema_version": METRICS_SCHEMA_VERSION,
            "event_type": event_type,
            "timestamp_unix": time.time(),
            **fields,
        }
        with open(self.events_path, "a") as f:
            f.write(json.dumps(payload, sort_keys=True) + "\n")
