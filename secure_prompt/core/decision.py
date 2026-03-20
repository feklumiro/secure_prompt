from dataclasses import dataclass

from secure_prompt.audit.logger import SecurityLogger

from secure_prompt.core.preprocess import preprocess
from secure_prompt.ML.ml_guard import MLGuard
from secure_prompt.core.scoring import PIPELINE_POLICY


@dataclass
class DecisionResult:
    verdict: str
    probability: float
    score: float
    features: list[float]


class DecisionCore:
    def __init__(self, use_vector: bool = True):
        self.jail_score = PIPELINE_POLICY[bool(use_vector)]
        self.logger = SecurityLogger()
        self.guard = MLGuard(threshold=self.jail_score, use_vector=use_vector)
        if use_vector and not self.guard.use_vector:
            print("Exception while connecting to HuggingFace Hub, check your internet connection")
            print("! Using non-vector model because no connection to HuggingFace Hub")

    def _apply_policy(self, prob: float) -> str:
        if prob >= self.jail_score:
            return "BLOCK"
        return "ALLOW"

    def decide(self, prompts: list[str]) -> list[DecisionResult]:
        normalized = preprocess(prompts)
        g_result = self.guard.detect(prompts + normalized)
        result = []

        for i in range(len(prompts)):
            score_raw, score_norm = g_result[i].score, g_result[i+len(prompts)].score
            ans = g_result[i+len(prompts)]
            if score_raw > score_norm:
                ans = g_result[i]
            score = ans.score
            prob = ans.probability
            features = ans.features
            verdict = self._apply_policy(prob)
            result.append(DecisionResult(
                verdict=verdict,
                probability=prob,
                score=score,
                features=features,
            ))
            self.logger.log_input_check(
                raw_prompt=prompts[i],
                decision=verdict,
                probability=prob,
                score=score,
                features=features
            )

        return result
