from dataclasses import dataclass

from secure_prompt.audit.logger import SecurityLogger

from secure_prompt.core.preprocess import preprocess

from secure_prompt.guards.regex_guard import RegexGuard
from secure_prompt.guards.ml_guard import MLGuard
from secure_prompt.guards.vector_guard import VectorGuard
from secure_prompt.core.scoring import PIPELINE_POLICY
from secure_prompt.core.scoring import score_regex, score_ml, score_vector


@dataclass
class Decision:
    verdict: str
    score: int
    reason: list[str]


class HybridGuard:
    def __init__(self,
                 use_regex: bool = True,
                 use_ml: bool = True,
                 use_vector: bool = True
                 ):
        if not (use_regex or use_ml or use_vector):
            raise ValueError("At least one guard is needed")
        self.scores = PIPELINE_POLICY
        self.regex_guard = self.ml_guard = self.vector_guard = None
        self.guards = []
        if use_regex:
            self.regex_guard = RegexGuard()
            self.guards.append((self.regex_guard, score_regex, "regex"))
        if use_ml:
            self.ml_guard = MLGuard()
            self.guards.append((self.ml_guard, score_ml, "ML"))
        if use_vector:
            self.vector_guard = VectorGuard()
            self.guards.append((self.vector_guard, score_vector, "vector"))

    # [(0, 0), (1, 1635), (40, 1918), (23, 1936), (17, 1292), (1, 1647), (40, 1920), (23, 1937)]
    def _apply_policy(self, score: int) -> str:
        if score >= self.scores[bool(self.regex_guard)*4 + bool(self.ml_guard)*2 + bool(self.vector_guard)]:
            return "BLOCK"
        return "ALLOW"

    def decide(self, user_prompt: str) -> Decision:
        logger = SecurityLogger()

        variants = preprocess(user_prompt)
        scores = [0]

        regex_matched_rules = set()
        matched_guards = set()
        matched_variants = set()

        for var, text in variants.items():
            score = 0
            for guard in self.guards:
                g_result = guard[0].detect(text)
                print(g_result)
                score += guard[1](g_result)
                if g_result.is_jailbreak:
                    matched_guards.add(guard[2])
                    matched_variants.add(var)
                if guard[2] == "regex" and g_result.rules:
                    for matched in g_result.rules:
                        regex_matched_rules.add(matched[0])
            scores.append(score)

        score = max(scores)
        verdict = self._apply_policy(score)
        if verdict == "ALLOW":
            regex_matched_rules.clear()

        logger.log_input_check(
            raw_prompt=user_prompt,
            decision=verdict,
            score=score,
            matched_guards=list(matched_guards),
            matched_rules=list(regex_matched_rules),
            matched_variants=list(matched_variants)
        )

        return Decision(
            verdict=verdict,
            score=score,
            reason=list(regex_matched_rules),
        )
