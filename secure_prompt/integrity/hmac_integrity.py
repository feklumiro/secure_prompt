import hmac
import hashlib
from typing import Final


class SystemPromptIntegrity:
    HASH_ALGO: Final = hashlib.sha256

    def __init__(self, secret_key: bytes):
        if len(secret_key) < 32:
            raise ValueError("HMAC key must be at least 256 bits")
        self._key = secret_key

    def sign(self, system_prompt: str) -> str:
        mac = hmac.new(
            self._key,
            system_prompt.encode("utf-8"),
            self.HASH_ALGO
        )
        return mac.hexdigest()

    def verify(self, system_prompt: str, signature: str) -> bool:
        expected = self.sign(system_prompt)
        return hmac.compare_digest(expected, signature)
