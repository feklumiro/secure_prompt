import re
from dataclasses import dataclass
from secure_prompt.core.base import BaseResult
from data.lexical import LEXEMES


@dataclass
class RegexResult(BaseResult):
    pass


class RegexGuard:
    def __init__(self):
        self.gap_config = {
            "override": 6,
            "freedom": 4,
            "roleplay": 5,
            "system": 7
        }

        # Веса для разных типов связок
        self.pattern_weights = {
            "override": 1.0,
            "system": 0.9,
            "roleplay": 0.8,
            "freedom": 0.7
        }
        self.patterns = self._build_patterns()
        self.dangerous = self._build_dangerous()

    def _build_patterns(self):
        """Строит оптимизированные regex паттерны"""
        patterns = {}

        # 1. Основные связки по категориям
        for category in LEXEMES["actions"].keys():
            actions = '|'.join(re.escape(word) for word in LEXEMES["actions"][category])
            targets = '|'.join(re.escape(word) for word in LEXEMES["targets"][category])
            gap = self.gap_config[category]

            # Основной паттерн: действие + [gap] + цель
            patterns[f"{category}_main"] = re.compile(
                rf'\b({actions})\b'  # Действие
                rf'{self._simple_gap(gap)}'  # Динамический gap
                rf'.*?\b({targets})\b',  # Цель
                re.IGNORECASE | re.DOTALL
            )

            # Обратный паттерн: цель + [gap] + действие
            patterns[f"{category}_reverse"] = re.compile(
                rf'\b({targets})\b'  # Цель
                rf'{self._simple_gap(gap)}'  # Динамический gap
                rf'.*?\b({actions})\b',  # Действие
                re.IGNORECASE | re.DOTALL
            )

        # 2. Комбинированные паттерны (межкатегориальные)
        # Например: "ignore" + "system prompt" (override + system)
        patterns["override_system"] = re.compile(
            rf'\b({"|".join(re.escape(w) for w in LEXEMES["actions"]["override"])})\b'
            rf'{self._simple_gap(8)}'
            rf'.*?\b({"|".join(re.escape(w) for w in LEXEMES["targets"]["system"])})\b',
            re.IGNORECASE | re.DOTALL
        )

        # "act as" + "without limits" (roleplay + freedom)
        patterns["roleplay_freedom"] = re.compile(
            rf'\b({"|".join(re.escape(w) for w in LEXEMES["actions"]["roleplay"])})\b'
            rf'{self._simple_gap(6)}'
            rf'.*?\b({"|".join(re.escape(w) for w in LEXEMES["targets"]["freedom"])})\b',
            re.IGNORECASE | re.DOTALL
        )
        return patterns

    @staticmethod
    def _build_dangerous():
        dangerous = []
        for category_dict in LEXEMES["targets"].values():
            dangerous.extend(category_dict)
        return dangerous

    @staticmethod
    def _simple_gap(max_words: int = 4) -> str:
        """Упрощенный gap для совместимости с вашим кодом"""
        return rf'(?:\W+\w+){{0,{max_words}}}'

    def detect(self, text: str) -> RegexResult:
        """Основная функция детекции"""
        if not text or len(text.strip()) < 3:
            return RegexResult(is_jailbreak=False, rules=[])

        rules_detected = []
        is_jailbreak = False

        # 1. Проверка основных паттернов
        for pattern_name, pattern in self.patterns.items():
            match = pattern.search(text)
            if match:
                rules_detected.append((pattern_name, match.group(0)))

        # 2. Опасные слова
        found_words = []
        for word in set(self.dangerous):  # Уникальные слова
            if word in text:
                found_words.append(word)

        if len(found_words) >= 4:
            rules_detected.append(("MULTIPLE_DANGEROUS_WORDS", ",".join(found_words)))

        if len(rules_detected) > 0:
            is_jailbreak = True

        return RegexResult(
            is_jailbreak=is_jailbreak,
            rules=rules_detected,
        )
