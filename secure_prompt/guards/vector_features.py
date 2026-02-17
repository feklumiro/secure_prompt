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
    template_scores: Optional[np.ndarray] = None


class VectorFeatureExtractor:
    MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    def __init__(
            self,
            templates: List[Dict] = VECTOR_TEMPLATES,
            threshold: float = VECTOR_JAIL_SCORE,
            use_faiss: bool = True,
            cache_dir: str = "./vector_cache",
            feature_config: Dict[str, bool] = None
    ):
        self.templates = templates
        self.threshold = threshold
        self.use_faiss = use_faiss
        self.cache_dir = cache_dir

        # Конфигурация признаков
        self.feature_config = feature_config or {
            'include_template_scores': True,
            'include_stats': True,
            'include_top_k': True,
            'include_category_scores': True
        }

        self.logger = logging.getLogger(__name__)
        os.makedirs(cache_dir, exist_ok=True)

        # Инициализация модели
        self.model = SentenceTransformer(self.MODEL_NAME)

        # Загружаем или вычисляем эмбеддинги шаблонов
        self.template_embeddings = None
        self.template_texts = None
        self.template_categories = None
        self.template_weights = None
        self._init_embeddings()

        # Получаем уникальные категории для консистентности
        self.unique_categories = sorted(set(self.template_categories))

        # Индекс для быстрого поиска (опционально)
        self.index = None
        if use_faiss:
            self._init_index()

        # Кэш для эмбеддингов текстов
        self.text_embedding_cache = {}
        self.max_cache_size = 1000

        # Предвычисляем размерность признаков
        self.feature_dim = self._compute_feature_dim()
        self.logger.info(f"Размерность признаков: {self.feature_dim}")

    # ---------- INIT ----------

    def _init_embeddings(self):
        """Инициализация эмбеддингов шаблонов"""
        cache_path = os.path.join(self.cache_dir, "template_embeddings_v4.pkl")

        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
                self.template_embeddings = data["embeddings"]
                self.template_texts = data["texts"]
                self.template_categories = data.get("categories", [])
                self.template_weights = data.get("weights", [])
        else:
            # Извлекаем текст, категорию и вес из шаблонов
            self.template_texts = [t["text"] for t in self.templates]
            self.template_categories = [t.get("category", "unknown") for t in self.templates]
            self.template_weights = [t.get("weight", 1.0) for t in self.templates]

            # Вычисляем эмбеддинги
            self.template_embeddings = self.model.encode(
                self.template_texts,
                normalize_embeddings=True,
                show_progress_bar=False
            )

            # Сохраняем в кэш
            with open(cache_path, "wb") as f:
                pickle.dump({
                    "embeddings": self.template_embeddings,
                    "texts": self.template_texts,
                    "categories": self.template_categories,
                    "weights": self.template_weights
                }, f)

        self.logger.info(f"Загружено {len(self.template_texts)} шаблонов для извлечения признаков")

    def _init_index(self):
        """Инициализация FAISS индекса для быстрого поиска"""
        dim = self.template_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.template_embeddings.astype("float32"))

    def _compute_feature_dim(self) -> int:
        """Вычисляет размерность вектора признаков"""
        dim = 0

        # 1. Похожесть на каждый паттерн
        if self.feature_config.get('include_template_scores', True):
            dim += len(self.template_texts)

        # 2. Статистики
        if self.feature_config.get('include_stats', True):
            dim += 3  # max, mean, std

        # 3. Топ-K значения
        if self.feature_config.get('include_top_k', True):
            dim += 3  # топ-3

        # 4. Оценки по категориям
        if self.feature_config.get('include_category_scores', True):
            dim += len(self.unique_categories)

        return dim

    # ---------- FEATURE EXTRACTION ----------

    def get_text_embedding(self, text: str) -> np.ndarray:
        """Получает эмбеддинг текста с кэшированием"""
        if text in self.text_embedding_cache:
            return self.text_embedding_cache[text]

        embedding = self.model.encode(
            text,
            normalize_embeddings=True
        )

        if len(self.text_embedding_cache) >= self.max_cache_size:
            self.text_embedding_cache.pop(next(iter(self.text_embedding_cache)))
        self.text_embedding_cache[text] = embedding

        return embedding

    def _compute_similarities(self, text_emb: np.ndarray) -> np.ndarray:
        """Вычисляет похожесть со всеми шаблонами"""
        if self.index is not None and self.use_faiss:
            # Получаем все похожести через FAISS
            if len(self.template_embeddings) <= 5000:
                sims, idxs = self.index.search(
                    text_emb.reshape(1, -1).astype("float32"),
                    k=len(self.template_embeddings)
                )
                similarities = np.zeros(len(self.template_embeddings))
                similarities[idxs[0]] = sims[0]
            else:
                # Для большого количества используем батчи
                similarities = np.zeros(len(self.template_embeddings))
                batch_size = 1000
                for i in range(0, len(self.template_embeddings), batch_size):
                    end = min(i + batch_size, len(self.template_embeddings))
                    batch_sims = np.dot(self.template_embeddings[i:end], text_emb)
                    similarities[i:end] = batch_sims
        else:
            similarities = np.dot(self.template_embeddings, text_emb)

        return similarities

    def extract_features_vector(self, text: str) -> List[float]:
        """
        Извлекает признаки для ML модели в виде списка ФИКСИРОВАННОЙ длины.

        Args:
            text: Текст для анализа

        Returns:
            list[float] признаков фиксированной длины
        """
        if not text or len(text.strip()) < 3:
            # Возвращаем нулевой список фиксированной длины
            return [0.0] * self.feature_dim

        # Получаем эмбеддинг и вычисляем похожести
        text_emb = self.get_text_embedding(text)
        similarities = self._compute_similarities(text_emb)

        # Применяем веса
        weighted_similarities = similarities * np.array(self.template_weights)

        feature_list = []

        # 1. Похожесть на каждый паттерн
        if self.feature_config.get('include_template_scores', True):
            feature_list.extend(weighted_similarities.tolist())

        # 2. Статистики
        if self.feature_config.get('include_stats', True):
            feature_list.append(float(np.max(weighted_similarities)))
            feature_list.append(float(np.mean(weighted_similarities)))
            feature_list.append(float(np.std(weighted_similarities)))

        # 3. Топ-3 значения
        if self.feature_config.get('include_top_k', True):
            top_3 = sorted(weighted_similarities, reverse=True)[:3]
            # Дополняем нулями до 3 элементов
            top_3_padded = top_3 + [0.0] * (3 - len(top_3))
            feature_list.extend(top_3_padded)

        # 4. Оценки по категориям
        if self.feature_config.get('include_category_scores', True):
            # Инициализируем словари для всех категорий
            cat_scores = {cat: 0.0 for cat in self.unique_categories}
            cat_counts = {cat: 0 for cat in self.unique_categories}

            # Суммируем по категориям
            for i, cat in enumerate(self.template_categories):
                cat_scores[cat] += weighted_similarities[i]
                cat_counts[cat] += 1

            # Добавляем средние значения в порядке категорий
            for cat in self.unique_categories:
                if cat_counts[cat] > 0:
                    feature_list.append(cat_scores[cat] / cat_counts[cat])
                else:
                    feature_list.append(0.0)

        # Проверяем размерность
        assert len(feature_list) == self.feature_dim, \
            f"Ожидаемая размерность {self.feature_dim}, получена {len(feature_list)}"

        return feature_list

    def extract_features_batch(
            self,
            texts: List[str],
            show_progress: bool = False,
            batch_size: int = 32
    ) -> List[List[float]]:
        """
        Извлекает признаки для батча текстов.

        Returns:
            List[List[float]] - список списков признаков фиксированной длины
        """
        if not texts:
            return []

        all_features = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            # Получаем эмбеддинги для всего батча
            batch_embeddings = self.model.encode(
                batch,
                normalize_embeddings=True,
                show_progress_bar=show_progress
            )

            for emb in batch_embeddings:
                # Вычисляем похожести
                similarities = self._compute_similarities(emb)
                weighted = similarities * np.array(self.template_weights)

                feature_list = []

                # 1. Похожесть на паттерны
                if self.feature_config.get('include_template_scores', True):
                    feature_list.extend(weighted.tolist())

                # 2. Статистики
                if self.feature_config.get('include_stats', True):
                    feature_list.append(float(np.max(weighted)))
                    feature_list.append(float(np.mean(weighted)))
                    feature_list.append(float(np.std(weighted)))

                # 3. Топ-3
                if self.feature_config.get('include_top_k', True):
                    top_3 = sorted(weighted, reverse=True)[:3]
                    top_3_padded = top_3 + [0.0] * (3 - len(top_3))
                    feature_list.extend(top_3_padded)

                # 4. Категории
                if self.feature_config.get('include_category_scores', True):
                    cat_scores = {cat: 0.0 for cat in self.unique_categories}
                    cat_counts = {cat: 0 for cat in self.unique_categories}

                    for j, cat in enumerate(self.template_categories):
                        cat_scores[cat] += weighted[j]
                        cat_counts[cat] += 1

                    for cat in self.unique_categories:
                        if cat_counts[cat] > 0:
                            feature_list.append(cat_scores[cat] / cat_counts[cat])
                        else:
                            feature_list.append(0.0)

                # Проверяем размерность
                assert len(feature_list) == self.feature_dim, \
                    f"Ожидаемая размерность {self.feature_dim}, получена {len(feature_list)}"

                all_features.append(feature_list)

        return all_features

    # ---------- UTILITY ----------

    def get_feature_names(self) -> List[str]:
        """Возвращает имена признаков для интерпретации"""
        names = []

        # 1. Имена шаблонов
        if self.feature_config.get('include_template_scores', True):
            for i, (text, cat) in enumerate(zip(self.template_texts, self.template_categories)):
                short_text = text[:30] + "..." if len(text) > 30 else text
                names.append(f"template_{i}_{cat}_{short_text}")

        # 2. Статистики
        if self.feature_config.get('include_stats', True):
            names.append("max_similarity")
            names.append("mean_similarity")
            names.append("std_similarity")

        # 3. Топ-3
        if self.feature_config.get('include_top_k', True):
            names.append("top_1_similarity")
            names.append("top_2_similarity")
            names.append("top_3_similarity")

        # 4. Категории
        if self.feature_config.get('include_category_scores', True):
            for cat in self.unique_categories:
                names.append(f"category_{cat}_score")

        return names

    def get_template_summary(self) -> Dict[str, Any]:
        """Возвращает сводку по шаблонам"""
        categories = {}
        for cat in self.template_categories:
            categories[cat] = categories.get(cat, 0) + 1

        return {
            'total_templates': len(self.template_texts),
            'categories': categories,
            'embedding_dim': self.template_embeddings.shape[1],
            'feature_dim': self.feature_dim,
            'model': self.MODEL_NAME,
            'feature_config': self.feature_config
        }


class VectorGuard(VectorFeatureExtractor):
    """Класс для обратной совместимости"""

    def detect(self, text: str, return_metadata: bool = False) -> VectorResult:
        if not text or len(text.strip()) < 5:
            return VectorResult(False, 0.0, None, "vector_v3")

        # Используем extract_features_vector для получения признаков
        features = self.extract_features_vector(text)

        # Для совместимости вычисляем максимальную похожесть
        n_templates = len(self.template_texts)
        template_scores = features[:n_templates] if self.feature_config.get('include_template_scores', True) else [
                                                                                                                      0.0] * n_templates
        max_similarity = float(max(template_scores)) if template_scores else 0.0

        is_jailbreak = max_similarity >= self.threshold

        meta = None
        if return_metadata:
            meta = {
                'max_similarity': max_similarity,
                'feature_dim': len(features),
                'template_scores': template_scores[:5]  # только топ-5
            }

        # Находим лучший шаблон
        if template_scores:
            best_idx = template_scores.index(max(template_scores))
            matched_template = self.template_texts[best_idx] if best_idx >= 0 else None
        else:
            matched_template = None

        return VectorResult(
            is_jailbreak=is_jailbreak,
            score=max_similarity,
            matched_template=matched_template,
            method="vector_v3",
            metadata=meta,
            template_scores=np.array(template_scores) if template_scores else None
        )