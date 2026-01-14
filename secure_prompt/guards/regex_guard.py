from dataclasses import dataclass
from typing import Optional
import re


GAP_WORDS = 6

@dataclass
class RegexResult:
    is_jailbreak: bool
    rule: Optional[str]
    matched_text: Optional[str]


LEXEMES = {
    "override1": [
        "ignore", "bypass", "disable", "forget", "skip", "override", "neglect",
        "circumvent", "avoid", "disregard", "omit", "suspend", "ignor", "erase",
        "игнор", "игнорируй", "проигнорируй", "обойди", "обходи", "действуй",
        "забудь", "не учитывай", "пропусти", "отмени", "сними", "отключи", "не обращай внимания"
    ],
    "override2": [
        "rules", "instructions", "policies", "safety", "limitations",
        "restrictions", "constraints", "filters", "guardrails",
        "правила", "инструкции", "ограничения", "фильтры", "блокировки", "политику",
        "политики", "рамки", "запреты", "всё выше", "все выше", "безопасность", "безопасности"
    ],

    "freedom1": [
        "say", "tell", "mean", "reveal", "show", "display", "print", "dump", "describe", "explain", "list", "output",
        "обойди", "обходи", "действуй",
        "ответь", "объясни", "расскажи", "опиши", "отвечай"
    ],
    "freedom2": [
        "свободно", "честно", "без ограничений",
        "без фильтров", "откровенно", "freely", "honestly", "without limits", "without restrictions",
        "unfiltered", "openly", "no constraints"
    ],

    "roles1": [
        "act", "behave", "pretend", "simulate", "assume",
        "roleplay", "imagine", "consider yourself", "play",
        "действуй", "веди себя", "притворись",
        "считай себя", "сыграй", "стань", "представь", "поиграй"
    ],
    "roles2": [
        "system", "developer", "admin", "administrator", "owner",
        "root", "maintainer", "engineer", "hacker", "bad", "evil"
        "система", "разработчик", "плохим", "злым", "негативным", "злодей", "злодеем"
        "владелец", "создатель", "хакер", "администратор"
    ],

    "system_prompt1": [
        "reveal", "show", "display", "print", "dump", "give",
        "describe", "explain", "list", "output", "say", "tell",
        "покажи", "раскрой", "выведи", "опиши", "расскажи"
        "объясни", "перечисли", "дай"
    ],
    "system_prompt2": [
        "system prompt", "initial prompt", "internal prompt",
        "system instructions", "rules",
        "hidden prompt", "configuration",
        "системный промпт", "системные инструкции",
        "правила", "начальные инструкции", "ограничения", "конфигурацию",
    ],
}


def gap(max_words: int = 4) -> str:
    return rf"\s+(?:\S+\s+){{0,{max_words}}}"


def group(words: list[str]) -> str:
    escaped = words  # map(re.escape, words)
    return "(" + "|".join(escaped) + ")"


JAILBREAK_PATTERNS = {
    # Override / ignore instructions
    "override_rules": re.compile(
        rf"{group(LEXEMES['override1'])}"
        rf"{gap(GAP_WORDS)}"
        rf"{group(LEXEMES['override2'])}",
        re.I
    ),

    # Freedom framing / no limits
    "freedom_farming": re.compile(
        rf"{group(LEXEMES['freedom1'])}"
        rf"{gap(GAP_WORDS)}"
        rf"{group(LEXEMES['freedom2'])}",
        re.I
    ),

    # Role manipulation
    "role_manipulation": re.compile(
        rf"{group(LEXEMES['roles1'])}"
        rf"{gap(GAP_WORDS)}"
        rf"{group(LEXEMES['roles2'])}",
        re.I
    ),

    # System prompt / internals extraction
    "system_prompt_extraction": re.compile(
        rf"{group(LEXEMES['system_prompt1'])}"
        rf"{gap(GAP_WORDS)}"
        rf"{group(LEXEMES['system_prompt2'])}",
        re.I
    ),
}

def regex_guard(user_prompt: str) -> list[RegexResult]:
    text = user_prompt
    results = []

    for rule_name, pattern in JAILBREAK_PATTERNS.items():
        match = pattern.search(text)
        if match:
            results.append(RegexResult(
                is_jailbreak=True,
                rule=rule_name,
                matched_text=match.group(0)
            ))

    return results

