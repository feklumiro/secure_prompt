import json
from pathlib import Path
from typing import Protocol
from secure_prompt.audit.models import SecurityEvent


class StorageBackend(Protocol):
    def write(self, event: SecurityEvent) -> None:
        ...


class JsonlStorage:
    def __init__(self, path: str = "security.log.jsonl"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: SecurityEvent) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event.__dict__, ensure_ascii=False) + "\n")
