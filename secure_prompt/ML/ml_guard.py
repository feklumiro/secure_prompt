from dataclasses import dataclass
import math
import pickle
from pathlib import Path
from requests.exceptions import ConnectionError

from secure_prompt.core.scoring import PIPELINE_POLICY
from secure_prompt.core.base import BaseResult
from secure_prompt.features.feature_extractor import FeatureExtractor


MODEL_PATH_VECTOR = Path(__file__).resolve().parents[0] / "models" / "model_vector.pkl"
MODEL_PATH = Path(__file__).resolve().parents[0] / "models" / "model.pkl"


@dataclass
class MLResult(BaseResult):
    probability: float
    score: float
    features: list[float]


class MLGuard:
    def __init__(self, model_path: Path = None, use_vector: bool = False, threshold: float = None):
        self.use_vector = use_vector
        if not threshold:
            threshold = PIPELINE_POLICY[bool(use_vector)]
        self.threshold = threshold
        try:
            self.feature_extractor = FeatureExtractor()
        except ConnectionError:
            self.use_vector = False
            self.feature_extractor = FeatureExtractor(init_vector=False)
        if not model_path and self.use_vector:
            model_path = MODEL_PATH_VECTOR
        elif not model_path:
            model_path = MODEL_PATH
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

    def predict(self, x: list[list[float]]) -> list[list[float]]:
        return self.model.predict_proba(x)

    def detect(self, texts: list[str]) -> list[MLResult]:
        feats = self.feature_extractor.extract_features(texts, self.use_vector)
        x = feats
        probs = self.predict(x)
        return [MLResult(
            is_jailbreak=prob[1] >= self.threshold,
            rules=None,
            probability=prob[1],
            score=-math.log(1 - prob[1] + 1e-6),
            features=feats[i]
        ) for i, prob in enumerate(probs)]
