import csv
from pathlib import Path
from typing import List, Tuple

from secure_prompt.core.preprocess import preprocess
from secure_prompt.guards.regex_guard import LEXEMES


DATA_DIR = Path(__file__).resolve().parents[1] / "data"


class DatasetLoader:
    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = data_dir

    def load_file(self, filename: str) -> List[str]:
        path = self.data_dir / filename
        samples = []

        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if row:
                    samples.append(row[0])

        return samples

    def extract_features(self, text: str) -> List[float]:
        text = preprocess(text)["normalized"]
        words = text.split()

        meta_words = {
            "override1": LEXEMES["override1"],
            "override2": LEXEMES["override2"],
            "roles2": LEXEMES["roles2"],
            "freedom2": LEXEMES["freedom2"],
            "system_prompt2": LEXEMES["system_prompt2"]
        }
        meta_words_values = {
            "override1": 2,
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

        features = [
            len(text),                                                                                                  # len_chars
            len(words),                                                                                                 # len_words
            sum(len(w) for w in words) / max(len(words), 1),                                                            # avg_word_len
            sum(not c.isalpha() for c in text) / max(len(text), 1),                                                     # non_alpha_ratio
            meta_words_hits,                                                                                            # meta_word_hits
            sum(imp in text for imp in imperatives),                                                                    # imperative_hits
        ]

        return features

    def load_dataset(self) -> Tuple[List[List[float]], List[int]]:
        benign = self.load_file("benign.csv")
        jailbreak = self.load_file("jailbreak.csv")

        X, y = [], []

        for text in benign:
            X.append(self.extract_features(text))
            y.append(0)

        for text in jailbreak:
            X.append(self.extract_features(text))
            y.append(1)

        return X, y
