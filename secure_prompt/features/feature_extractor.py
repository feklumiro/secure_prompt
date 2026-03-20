import csv
import os

from pathlib import Path
from typing import List, Tuple
from requests.exceptions import ConnectionError

from secure_prompt.core.preprocess import preprocess
from secure_prompt.features.vector_features import VectorFeatureExtractor
from secure_prompt.features.linear_features import LinearFeatureExtractor

from dotenv import load_dotenv

load_dotenv()
DATA_DIR = Path(__file__).resolve().parents[2] / "data"


class FeatureExtractor:
    def __init__(self, init_vector=True):
        self.vector_feats_extractor = None
        if init_vector:
            self.vector_feats_extractor = VectorFeatureExtractor()
        self.linear_feats_extractor = LinearFeatureExtractor()

    def extract_features(self, texts: List[str], use_vector: bool = False) -> List[List[float]]:
        # -------- FEATS #4 -----------------
        l_feats = self.linear_feats_extractor.extract_features_static(texts)
        if use_vector:
            if not self.vector_feats_extractor:
                self.vector_feats_extractor = VectorFeatureExtractor()
            v_feats = self.vector_feats_extractor.extract_features_batch(texts)
            result = []
            for i in range(len(texts)):
                result.append(l_feats[i] + v_feats[i])
            return result
        return l_feats


class DatasetLoader:
    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = data_dir
        self.vector_import = True
        try:
            self.extractor = FeatureExtractor()
        except ConnectionError:
            self.vector_import = False
            self.extractor = FeatureExtractor(init_vector=False)

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

        X = self.extractor.extract_features(data)
        y = [0] * len(benign) + [1] * len(jailbreak)
        X_v, y_v = [], []
        if self.vector_import:
            X_v = self.extractor.extract_features(data, use_vector=True)
            y_v = [0] * len(benign) + [1] * len(jailbreak)

        return X, y, X_v, y_v
