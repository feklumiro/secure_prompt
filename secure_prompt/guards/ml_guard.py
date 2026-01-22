from dataclasses import dataclass
from typing import Dict
import math
import pickle
from pathlib import Path

from secure_prompt.core.preprocess import preprocess
from secure_prompt.core.base import BaseResult
from secure_prompt.guards.regex_guard import LEXEMES


MODEL_PATH = Path(__file__).resolve().parents[2] / "ml" / "model.pkl"


@dataclass
class MLResult(BaseResult):
    probability: float
    score: float
    features: Dict[str, float]


class MLGuard:
    def __init__(self, model_path: Path = MODEL_PATH):
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

    @staticmethod
    def extract_features(text: str) -> Dict[str, float]:
        words = text.split()

        meta_words = {
            "override2": LEXEMES["override2"],
            "roles2": LEXEMES["roles2"],
            "freedom2": LEXEMES["freedom2"],
            "system_prompt2": LEXEMES["system_prompt2"]
        }
        meta_words_values = {
            "override2": 3,
            "roles2": 1,
            "freedom2": 1,
            "system_prompt2": 5
        }
        imperatives = set(
            LEXEMES["override1"]
            + LEXEMES["freedom1"]
            + LEXEMES["roles1"]
            + LEXEMES["system_prompt1"]
        )
        meta_words_hits = 0
        for meta in meta_words:
            for meta_w in meta_words[meta]:
                if meta_w in text:
                    meta_words_hits += meta_words_values[meta]

        features = {
            "len_chars": len(text),
            "len_words": len(words),
            "avg_word_len": sum(len(w) for w in words) / max(len(words), 1),
            "non_alpha_ratio": sum(not c.isalpha() for c in text) / max(len(text), 1),
            "meta_word_hits": meta_words_hits,
            "imperative_hits": sum(imp in text for imp in imperatives)
        }

        return features

    def predict(self, text: str) -> MLResult:
        feats = self.extract_features(text)

        X = [list(feats.values())]
        prob = float(self.model.predict_proba(X)[0][1])

        return MLResult(
            is_jailbreak=None,
            rules=None,
            probability=prob,
            score=-math.log(1 - prob + 1e-6),
            features=feats
        )
