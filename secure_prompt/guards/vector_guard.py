import os
import pickle
import logging
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Optional, List

import faiss
from sentence_transformers import SentenceTransformer

from secure_prompt.core.scoring import VECTOR_JAIL_SCORE
from data.lexical import VECTOR_TEMPLATES

@dataclass
class VectorResult:
    is_jailbreak: bool
    score: float
    matched_template: Optional[str]
    method: str
    metadata: Optional[Dict[str, Any]] = None


class VectorGuard:
    """
    Vector-based jailbreak detector v3.
    Core idea:
    - attack vs context vectors
    - signed similarity
    - aggregation, not max()
    """

    MODEL_NAME = "intfloat/multilingual-e5-small"

    def __init__(
        self,
        templates: List[Dict] = VECTOR_TEMPLATES,
        threshold: float = VECTOR_JAIL_SCORE,
        use_faiss: bool = True,
        cache_dir: str = "./vector_cache"
    ):
        self.templates = templates
        self.threshold = threshold
        self.use_faiss = use_faiss
        self.cache_dir = cache_dir

        self.logger = logging.getLogger(__name__)
        os.makedirs(cache_dir, exist_ok=True)

        self.model = SentenceTransformer(self.MODEL_NAME)
        self.embeddings = None
        self.index = None

        self._init_embeddings()
        self._init_index()

    # ---------- INIT ----------

    def _init_embeddings(self):
        cache_path = os.path.join(self.cache_dir, "templates_v3.pkl")

        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
                self.embeddings = data["embeddings"]
                self.template_texts = data["texts"]
        else:
            texts = [t["text"] for t in self.templates]
            self.embeddings = self.model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            self.template_texts = texts
            with open(cache_path, "wb") as f:
                pickle.dump(
                    {"embeddings": self.embeddings, "texts": texts},
                    f
                )

    def _init_index(self):
        if not self.use_faiss:
            return
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings.astype("float32"))

    # ---------- CORE ----------

    def _score(self, query_emb: np.ndarray) -> Dict[str, Any]:
        if self.use_faiss:
            sims, idxs = self.index.search(
                query_emb.reshape(1, -1).astype("float32"),
                k=6
            )
            sims, idxs = sims[0], idxs[0]
        else:
            sims = np.dot(self.embeddings, query_emb)
            idxs = np.argsort(sims)[-6:][::-1]

        attack_score = 0.0
        context_score = 0.0
        categories = set()

        for sim, idx in zip(sims, idxs):
            tpl = self.templates[idx]
            weighted = sim * tpl["weight"]
            categories.add(tpl["category"])

            if tpl["type"] == "attack":
                attack_score += weighted
            else:
                context_score += weighted

        final = attack_score + context_score

        return {
            "final": final,
            "attack": attack_score,
            "context": context_score,
            "categories": categories,
            "best": self.templates[idxs[0]]["text"]
        }

    # ---------- PUBLIC ----------

    def detect(self, text: str, return_metadata: bool = False) -> VectorResult:
        if not text or len(text.strip()) < 5:
            return VectorResult(False, 0.0, None, "vector_v3")

        emb = self.model.encode(
            text,
            normalize_embeddings=True
        )

        score_pack = self._score(emb)

        # diversity bonus: multi-signal attack
        if len(score_pack["categories"]) >= 2:
            score_pack["final"] += 0.05

        is_jailbreak = score_pack["final"] >= self.threshold

        meta = None
        if return_metadata:
            meta = score_pack

        return VectorResult(
            is_jailbreak=is_jailbreak,
            score=score_pack["final"],
            matched_template=score_pack["best"],
            method="vector_v3",
            metadata=meta
        )
