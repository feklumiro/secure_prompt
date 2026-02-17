from dataclasses import dataclass
import math
import pickle
from pathlib import Path

from secure_prompt.core.scoring import ML_JAIL_SCORE
from secure_prompt.core.base import BaseResult
from ML.dataset import FeatureExtractor


MODEL_PATH_VECTOR = Path(__file__).resolve().parents[2] / "ml" / "model_vector.pkl"
MODEL_PATH = Path(__file__).resolve().parents[2] / "ml" / "model.pkl"


@dataclass
class MLResult(BaseResult):
    probability: float
    score: float
    features: list[float]


class MLGuard:
    def __init__(self, model_path: Path = MODEL_PATH, use_vector: bool = False, threshold: float = ML_JAIL_SCORE):
        self.use_vector = use_vector
        if use_vector:
            model_path = MODEL_PATH_VECTOR
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        self.threshold = threshold
        self.feature_extractor = FeatureExtractor()

    def predict(self, x: list[list[float]]) -> list[list[float]]:
        return self.model.predict_proba(x)

    def detect(self, texts: list[str]) -> list[MLResult]:
        feats = self.feature_extractor.extract_features(texts, self.use_vector)
        x = feats
        probs = self.predict(x)
        return [MLResult(
            is_jailbreak=-math.log(1 - prob[1] + 1e-6) >= self.threshold,
            rules=None,
            probability=prob[1],
            score=-math.log(1 - prob[1] + 1e-6),
            features=feats[i]
        ) for i, prob in enumerate(probs)]
