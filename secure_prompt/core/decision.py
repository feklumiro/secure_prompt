from enum import Enum
from dataclasses import dataclass

from secure_prompt.audit.logger import SecurityLogger

from secure_prompt.core.preprocess import preprocess

from secure_prompt.guards.regex_guard import regex_guard
from secure_prompt.guards.ml_guard import MLGuard
from secure_prompt.core.scoring import score_regex, score_ml


@dataclass
class Decision:
    verdict: str
    score: int
    reason: list[str]


def apply_policy(score: int) -> str:
    if score >= 40:
        return "BLOCK"
    return "ALLOW"


def decide(user_prompt: str) -> Decision:
    logger = SecurityLogger()

    variants = preprocess(user_prompt)
    scores = [0]

    regex_matched_rules = set()
    matched_guards = set()
    matched_variants = []

    for var, text in variants.items():
        score = 0

        regex_result = regex_guard(text)
        score += score_regex(regex_result)
        for matched in regex_result.rules:
            regex_matched_rules.add(matched[0])
        if regex_result.is_jailbreak:
            matched_guards.add("regex")

        ml = MLGuard()
        ml_result = ml.predict(text)
        score += score_ml(ml_result)
        scores.append(score)
        if score_ml(ml_result) > 0:
            matched_guards.add("ml")

        if regex_result.is_jailbreak or score_ml(ml_result) > 0:
            matched_variants.append(var)

    score = max(scores)
    verdict = apply_policy(score)

    logger.log_input_check(
        raw_prompt=user_prompt,
        decision=verdict,
        score=score,
        matched_guards=list(matched_guards),
        matched_rules=list(regex_matched_rules),
        matched_variants=matched_variants
    )

    return Decision(
        verdict=verdict,
        score=score,
        reason=list(regex_matched_rules),
    )
