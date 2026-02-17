from dataclasses import dataclass

from secure_prompt.audit.logger import SecurityLogger

from secure_prompt.core.preprocess import preprocess
from secure_prompt.guards.ml_guard import MLGuard
from secure_prompt.core.scoring import PIPELINE_POLICY


@dataclass
class DecisionResult:
    verdict: str
    score: float
    reason: list[str]


class DecisionCore:
    def __init__(self, use_vector: bool = True):
        self.jail_score = PIPELINE_POLICY[bool(use_vector)]
        self.logger = SecurityLogger()
        self.guard = MLGuard(use_vector=use_vector)

    def _apply_policy(self, score: float) -> str:
        if score >= self.jail_score:
            return "BLOCK"
        return "ALLOW"

    def decide(self, prompts: list[str]) -> list[DecisionResult]:
        normalized = preprocess(prompts)
        g_result = self.guard.detect(prompts + normalized)
        result = []

        for i in range(len(prompts)):
            score_raw, score_norm = g_result[i].score, g_result[i+len(prompts)].score
            score = max(score_raw, score_norm)
            reason = g_result[i].features if score == score_raw else g_result[i+len(prompts)].features
            verdict = self._apply_policy(score)
            if verdict == "ALLOW":
                reason = []
            result.append(DecisionResult(
                verdict=verdict,
                score=score,
                reason=reason,
            ))
            self.logger.log_input_check(
                raw_prompt=prompts[i],
                decision=verdict,
                score=score,
                reason=reason
            )

        return result
