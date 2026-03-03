from secure_prompt.audit.models import SecurityEvent
from secure_prompt.audit.storage import StorageBackend, JsonlStorage
from typing import Optional


class SecurityLogger:
    def __init__(self, storage: Optional[StorageBackend] = None):
        self.storage = storage or JsonlStorage()

    def log_input_check(
        self,
        raw_prompt: str,
        decision: str,
        probability: float,
        score: float,
        features: list[float]
    ) -> None:
        event = SecurityEvent(
            timestamp=self._now(),
            event_type="input_check",
            decision=decision,
            probability=probability,
            score=score,
            features=features,
            prompt_hash=SecurityEvent.hash_text(raw_prompt)
        )
        self.storage.write(event)

    def log_response_check(
        self,
        response_text: str,
        decision: str,
        probability: float,
        score: int,
        features: list[float]
    ) -> None:
        event = SecurityEvent(
            timestamp=self._now(),
            event_type="response_check",
            decision=decision,
            probability=probability,
            score=score,
            features=features,
            prompt_hash=SecurityEvent.hash_text(response_text),
        )
        self.storage.write(event)

    @staticmethod
    def _now() -> str:
        from datetime import datetime
        return datetime.utcnow().isoformat()
