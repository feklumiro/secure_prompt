from dataclasses import dataclass
from secure_prompt.core.base import BaseResult
import re


GAP_WORDS = 6


@dataclass
class ResponseGuardResult(BaseResult):
    pass


def gap(max_words: int = 4) -> str:
    return rf"\s+(?:\S+\s+){{0,{max_words}}}"


def group(words: list[str]) -> str:
    return "(" + "|".join(words) + ")"


LEXEMES = {
    # =========================
    # SYSTEM / POLICY LEAKAGE
    # =========================
    "system_leak_1": [
        "system", "internal", "hidden", "initial", "base", "core",
        "private", "backend", "underlying", "embedded",
        "внутренний", "внутренние", "внутренняя", "скрыт", "скрытый", "скрытые", "скрытая",
        "системный", "системные", "системная", "начальный", "начальные", "начальная",
        "базовый", "базовая", "базовые", "служебные", "служебный", "служебная", "правила"
        "конфиденциальный", "конфиденциальные", "конфиденциальная", "приватный", "приватная", "приватные", "ядро"
    ],
    "system_leak_2": [
        "prompt", "instruction", "rule", "policy", "configuration",
        "setup", "message", "guideline", "directive", "система", "системы",
        "промпт", "промпты", "инструкции", "инструкция", "правил", "правила", "правило", "политик", "политика", "политики",
        "конфигурация", "конфигурации", "конфигураций", "настройка", "настройку", "настройки",
        "сообщение", "сообщения", "сообщений", "директива", "директивы"
    ],

    # =========================
    # META DISCLOSURE
    # =========================
    "meta_1": [
        "policy", "policies", "rules", "guidelines",
        "moderation", "safety", "filters",
        "guardrails", "restrictions", "limitations",
        "openai", "safety system",
        "правила", "политики", "ограничения", "инструкции", "политика"
        "модерация", "фильтры", "безопасность",
        "защита", "запреты", "ограничения модели"
    ],
    "meta_2": [
        "prevent", "block", "restrict", "allow",
        "detect", "stop", "handle",
        "предотвращает", "блокирует", "ограничивает",
        "разрешает", "разрешают", "детектит", "детектят", "останавливает", "запрещает"
        "останавливают", "запрещают", "обрабатывают", "следят", "контролируют"
        "обрабатывает", "следит", "контролирует", "прерывает", "прерывают"
    ],

    # =========================
    # SAFETY / POLICY JUSTIFICATION
    # =========================
    "policy_1": [
        "policy", "policies", "guidelines", "rules",
        "safety", "moderation", "compliance",
        "политик", "правила", "рекомендации", "безопасности"
        "безопасность", "модерация", "соответствие", "соответствует"
    ],
    "policy_2": [
        "prevent", "ensure", "protect", "avoid",
        "enforce", "comply", "follow",
        "предотвращать", "обеспечивать", "защищает",
        "избегать", "соблюдать",
    ],

    # =========================
    # UNRESTRICTED / UNFILTERED CLAIMS
    # =========================
    "freedom_1": [
        "without", "no", "zero", "fully"
        "completely", "totally", "absolutely",
        "без", "никаких", "полностью", "игнорируя",
        "абсолютно", "совершенно", "отключая", "отключив"
    ],
    "freedom_2": [
        "restrictions", "filters", "limits", "constraints",
        "moderation", "rules", "boundaries",
        "ограничений", "фильтров", "лимитов",
        "цензуры", "правил", "рамок"
    ],

    # =========================
    # GENERIC HARMFUL INSTRUCTION FRAMING
    # =========================
    "harmful_1": [
        "how to", "steps", "instructions", "guide",
        "procedure", "method", "process",
        "как", "способ", "способы", "способов", "инструкции", "шаг",
        "шаги", "шагов", "алгоритмы", "процессы", "гайд", "метод"
        "алгоритм", "методы", "процесс", "процедура"
    ],
    "harmful_2": [
        "illegal", "harmful", "dangerous", "forbidden",
        "restricted", "prohibited",
        "незаконное", "незаконные", "незаконнный", "опасное", "опасные", "опасных", "опасный",
        "вредное", "вредные", "вредных", "вредный", "ограниченных",
        "запрещенный", "запрещенное", "запрещенные", "запрещенных", "ограниченный", "ограниченное", "ограниченные"
    ],
}


RESPONSE_PATTERNS = {
    # System / policy leakage
    "system_policy_leak": re.compile(
        rf"{group(LEXEMES['system_leak_1'])}"
        rf"{gap(GAP_WORDS)}"
        rf"{group(LEXEMES['system_leak_2'])}",
        re.I
    ),

    # Meta self-disclosure
    "meta_disclosure": re.compile(
        rf"{group(LEXEMES['meta_1'])}"
        rf"{gap(GAP_WORDS)}"
        rf"{group(LEXEMES['meta_2'])}",
        re.I
    ),

    # Safety justification leakage
    "policy_justification": re.compile(
        rf"{group(LEXEMES['policy_1'])}"
        rf"{gap(GAP_WORDS)}"
        rf"{group(LEXEMES['policy_2'])}",
        re.I
    ),

    # Claims of no restrictions
    "unrestricted_claim": re.compile(
        rf"{group(LEXEMES['freedom_1'])}"
        rf"{gap(GAP_WORDS)}"
        rf"{group(LEXEMES['freedom_2'])}",
        re.I
    ),

    # Generic harmful instructional framing
    "harmful_instruction": re.compile(
        rf"{group(LEXEMES['harmful_1'])}"
        rf"{gap(GAP_WORDS)}"
        rf"{group(LEXEMES['harmful_2'])}",
        re.I
    ),
}


def response_guard(llm_response: str) -> ResponseGuardResult:
    text = llm_response.lower()
    rules = []

    for rule_name, pattern in RESPONSE_PATTERNS.items():
        match = pattern.search(text)
        if match:
            rules.append((rule_name, str(match.group(0))))
    if not rules:
        return ResponseGuardResult(
            is_jailbreak=False,
            rules=None
        )
    return ResponseGuardResult(
        is_jailbreak=True,
        rules=rules
    )
