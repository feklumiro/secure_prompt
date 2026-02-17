import csv
import os

import numpy as np
from collections import Counter

from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords

from pathlib import Path
from typing import List, Tuple

from secure_prompt.core.preprocess import preprocess
from secure_prompt.guards.vector_features import VectorFeatureExtractor
from data.lexical import LEXEMES

from dotenv import load_dotenv

load_dotenv()
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def extract_features_static(text: str) -> List[float]:
    # -------- FEATS #1 -----------------

    char_counter = Counter(text.lower())
    char_probs = [count / len(text) for count in char_counter.values()]
    char_entropy = -sum(p * np.log2(p) for p in char_probs) if char_probs else 0

    stops_en = set(stopwords.words('english'))
    stops_ru = set(stopwords.words('russian'))
    words = word_tokenize(text.lower()) if text else []
    stopword_ratio_en = sum(1 for w in words if w in stops_en) / len(words) if words else 0
    stopword_ratio_ru = sum(1 for w in words if w in stops_ru) / len(words) if words else 0

    translation_markers = [
        'chatgpt', 'openai', 'gpt', 'ai', 'model', 'assistant',
        'please', 'could', 'would', 'should', 'generate', 'create',
        'content', 'response', 'output', 'instructions', 'ethical', 'input'
    ]
    translation_marker_count = sum(
        1 for marker in translation_markers if marker in text.lower()
    )

    # -------- FEATS #2 -----------------

    if words:
        lexical_diversity = len(set(words)) / len(words)
    else:
        lexical_diversity = 0

    tagged = pos_tag(words)

    pos_tags = [tag for word, tag in tagged]

    verb_ratio = sum(1 for tag in pos_tags if tag.startswith('VB')) / len(pos_tags) if pos_tags else 0
    noun_ratio = sum(1 for tag in pos_tags if tag.startswith('NN')) / len(pos_tags) if pos_tags else 0
    pronoun_ratio = sum(1 for tag in pos_tags if tag.startswith('PR')) / len(pos_tags) if pos_tags else 0

    # -------- FEATS #3 -----------------

    meta_category = ["override", "freedom", "roleplay", "system"]

    c = ("actions", "targets")
    meta_words_hits = [0] * (len(c) * len(meta_category))
    for vt in range(len(c)):
        for meta in range(len(meta_category)):
            for word in LEXEMES[c[vt]][meta_category[meta]]:
                if word in text:
                    meta_words_hits[vt * len(meta_category) + meta] += 1
    m1, m2, m3, m4, m5, m6, m7, m8 = meta_words_hits

    features = [
        sum(len(w) for w in words) / max(len(words), 1),
        sum(not c.isalpha() for c in text) / max(len(text), 1),
        m1, m2, m3, m4, m5, m6, m7, m8,
        char_entropy,
        stopword_ratio_en,
        stopword_ratio_ru,
        translation_marker_count,
        lexical_diversity,
        verb_ratio,
        noun_ratio,
        pronoun_ratio
    ]

    return features

class FeatureExtractor:
    def __init__(self, init_vector=True):
        self.vector_feats_extractor = None
        if init_vector:
            self.vector_feats_extractor = VectorFeatureExtractor()

    def extract_features(self, texts: List[str], use_vector: bool = False) -> List[List[float]]:
        result = []
        # -------- FEATS #4 -----------------
        v_feats = [[]] * len(texts)
        if use_vector:
            if not self.vector_feats_extractor:
                self.vector_feats_extractor = VectorFeatureExtractor()
            v_feats = self.vector_feats_extractor.extract_features_batch(texts)
        for i, text in enumerate(texts):
            result.append(extract_features_static(text) + v_feats[i])
        return result


class DatasetLoader:
    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = data_dir

    def load_file(self, filename: str) -> List[str]:
        path = self.data_dir / filename
        samples = []

        with open(path) as f:
            reader = csv.reader(f)
            for row in reader:
                if row:
                    samples.append(row[0])

        return samples

    def load_dataset(self) -> Tuple[List[List[float]], List[int], List[List[float]], List[int]]:
        benign = preprocess(self.load_file(os.getenv("BENIGN_DATA_PATH")))
        jailbreak = preprocess(self.load_file(os.getenv("JAILBREAK_DATA_PATH")))
        data = benign + jailbreak

        extractor = FeatureExtractor()

        X_v = extractor.extract_features(data, use_vector=True)
        y_v = [0] * len(benign) + [1] * len(jailbreak)
        X = extractor.extract_features(data)
        y = [0] * len(benign) + [1] * len(jailbreak)

        return X, y, X_v, y_v
