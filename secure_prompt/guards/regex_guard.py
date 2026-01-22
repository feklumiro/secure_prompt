from dataclasses import dataclass
from secure_prompt.core.base import BaseResult
import re


GAP_WORDS = 6

@dataclass
class RegexResult(BaseResult):
    pass


LEXEMES = {
    "override1": [
        "ignore", "bypass", "disable", "forget", "skip", "override", "neglect", "comply",
        "circumvent", "avoid", "disregard", "omit", "suspend", "ignor", "erase",
        "игнор", "игнорируй", "проигнорируй", "обойди", "обходи", "действуй",
        "забудь", "не учитывай", "пропусти", "отмени", "сними", "отключи", "не обращай внимания"
    ],
    "override2": [
        "rules", "instructions", "policies", "safety", "limitations", "moderation",
        "restrictions", "constraints", "filters", "guardrails", "policy",
        "правила", "инструкции", "ограничения", "фильтры", "блокировки", "политику",
        "политики", "рамки", "запреты", "всё выше", "все выше", "безопасность", "безопасности"
    ],

    "freedom1": [
        "say", "tell", "mean", "reveal", "show", "display", "print", "dump", "describe", "explain", "list", "output",
        "обойди", "обходи", "действуй", "act", "do", "делать", "отвечать", "говорить", "ответить",
        "ответь", "объясни", "расскажи", "опиши", "отвечай", "говори"
    ],
    "freedom2": [
        "свободно", "честно", "без ограничений",
        "без фильтров", "откровенно", "freely", "honestly", "without limits", "without restrictions", "without limitations",
        "unfiltered", "openly", "without constraints", "without filters", "fully", "completely", "totally", "absolutely",
        "no limits", "no restrictions", "no limitations", "no filters", "no constraints"
    ],

    "roles1": [
        "act", "behave", "pretend", "simulate", "assume", "become",
        "roleplay", "imagine", "consider yourself", "play",
        "действуй", "веди себя", "притворись", "отвечай",
        "считай себя", "сыграй", "стань", "представь", "поиграй", "предположи"
    ],
    "roles2": [
        "developer", "admin", "administrator", "owner", "ии без ограничений", "ии без фильтров",
        "root", "maintainer", "engineer", "hacker", "bad", "evil", "unresponsive", "unrestricted",
        "система", "разработчик", "плохим", "злым", "негативным", "злодей",
        "владелец", "создатель", "хакер", "администратор"
    ],

    "system_prompt1": [
        "reveal", "show", "display", "print", "dump", "give",
        "describe", "explain", "list", "output", "say", "tell",
        "покажи", "раскрой", "выведи", "опиши", "расскажи", "напиши", "отправь",
        "объясни", "перечисли", "дай", "скажи", "выдай"
    ],
    "system_prompt2": [
        "system prompt", "initial prompt", "internal prompt", "initial message", "safety",
        "system instructions", "rules", "prompt", "system message", "internal setup",
        "hidden prompt", "configuration", "промпт", "системных инструкций", "модерация", "модерацию",
        "системный промпт", "системные инструкции", "внутренние инструкции",
        "правила", "начальные инструкции", "ограничения", "конфигурацию",
        "скрытое сообщение", "скрытый промпт", "скрытые инструкции"
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

def regex_guard(user_prompt: str) -> RegexResult:
    text = user_prompt
    rules = []

    for rule_name, pattern in JAILBREAK_PATTERNS.items():
        match = pattern.search(text)
        if match:
            rules.append((rule_name, str(match.group(0))))
    if not rules:
        return RegexResult(
            is_jailbreak=False,
            rules=[]
        )
    return RegexResult(
        is_jailbreak=True,
        rules=rules
    )
