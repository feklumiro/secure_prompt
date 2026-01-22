from dataclasses import dataclass
from typing import Optional


class SecurityError(Exception):
    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = None


@dataclass
class BaseResult:
    is_jailbreak: Optional[bool]
    rules: Optional[list[tuple[str, str]]]
