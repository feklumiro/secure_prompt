from dataclasses import dataclass
import math
import pickle
from pathlib import Path

from secure_prompt.core.scoring import ML_JAIL_SCORE
from secure_prompt.core.base import BaseResult
from ML.dataset import extract_features


MODEL_PATH = Path(__file__).resolve().parents[2] / "ml" / "model.pkl"


@dataclass
class MLResult(BaseResult):
    probability: float
    score: float
    features: list[float]


class MLGuard:
    def __init__(self, model_path: Path = MODEL_PATH, threshold: float = ML_JAIL_SCORE):
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        self.threshold = threshold

    def predict(self, x: list[list[float]]) -> float:
        return float(self.model.predict_proba(x)[0][1])

    def detect(self, text: str) -> MLResult:
        feats = extract_features(text)
        X = [feats]
        prob = self.predict(X)
        return MLResult(
            is_jailbreak=-math.log(1 - prob + 1e-6) >= self.threshold,
            rules=None,
            probability=prob,
            score=-math.log(1 - prob + 1e-6),
            features=feats
        )
