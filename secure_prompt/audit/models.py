from dataclasses import dataclass
import hashlib


@dataclass
class SecurityEvent:
    timestamp: str
    event_type: str
    decision: str
    probability: float
    score: float
    features: list[float]

    prompt_hash: str = ""

    @staticmethod
    def hash_text(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
