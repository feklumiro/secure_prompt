from enum import Enum
from dataclasses import dataclass

from secure_prompt.core.preprocess import preprocess
from secure_prompt.guards.regex_guard import regex_guard
from secure_prompt.core.scoring import score_regex


class Verdict(Enum):
    ALLOW = "allow"
    REVIEW = "review"
    BLOCK = "block"


@dataclass
class Decision:
    verdict: Verdict
    score: int
    reason: list[str]


def apply_policy(score: int) -> Verdict:
    if score >= 70:
        return Verdict.BLOCK
    return Verdict.ALLOW


def decide(user_prompt: str) -> Decision:
    score = 0
    normalized = preprocess(user_prompt)
    regex_results = regex_guard(normalized)

    regex_matched_rules = []
    for matched in regex_results:
        score += score_regex(matched)
        regex_matched_rules.append(matched.rule)
    verdict = apply_policy(score)

    return Decision(
        verdict=verdict,
        score=score,
        reason=regex_matched_rules,
    )
