from dataclasses import dataclass, field
from typing import List
import hashlib


@dataclass
class SecurityEvent:
    timestamp: str
    event_type: str           # input_check | response_check
    decision: str             # allow | block
    score: int

    matched_guards: List[str] = field(default_factory=list)
    matched_rules: List[str] = field(default_factory=list)
    matched_variants: List[str] = field(default_factory=list)

    prompt_hash: str = ""

    @staticmethod
    def hash_text(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
