import re
import unicodedata
import base64


ZERO_WIDTH_CHARS = [
    "\u200b",  # zero width space
    "\u200c",  # zero width non-joiner
    "\u200d",  # zero width joiner
    "\ufeff",  # zero width no-break space
]


HOMOGLYPH_MAP_K_L = {
    "і": "i", "ӏ": "l", "а": "a", "с": "c", "е": "e", "о": "o", "у": "y", "к": "k", "х": "x", "р": "p"
}
HOMOGLYPH_MAP_L_K = {
    "a": "а", "c": "с", "e": "е", "o": "о", "y": "у", "k": "к", "x": "х", "p": "р"
}


def try_decode_base64(text: str) -> str:
    stripped = text.strip()

    if not re.fullmatch(r"[A-Za-z0-9+/=]+", stripped):
        return text

    try:
        decoded = base64.b64decode(stripped, validate=True).decode("utf-8")
        # Возвращает декодированный текст при количестве печатаемых символов >= 90% от всего текста
        if sum(c.isprintable() for c in decoded) / len(decoded) >= 0.9:
            return decoded
    except ValueError:
        pass

    return text


def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def remove_zero_width(text: str) -> str:
    for ch in ZERO_WIDTH_CHARS:
        text = text.replace(ch, "")
    return text


def normalize_punctuation(text: str) -> str:
    out = []

    for ch in text:
        cat = unicodedata.category(ch)

        if cat.startswith(("P", "S")):
            out.append(" ")
        else:
            out.append(ch)

    return "".join(out)


def normalize_spacing(text: str) -> str:
    while True:
        new = re.sub(r"(?<!\w)([a-zа-я])\s+(?=[a-zа-я](?!\w))", r"\1", text)
        if new == text:
            break
        text = new
    return text


def normalize_whitespace(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_homoglyphs_k_l(text: str) -> str:
    return "".join(HOMOGLYPH_MAP_K_L.get(c, c) for c in text)


def normalize_homoglyphs_l_k(text: str) -> str:
    return "".join(HOMOGLYPH_MAP_L_K.get(c, c) for c in text)


def preprocess(text: str) -> dict:
    raw = text
    text = try_decode_base64(text)
    text = text.lower()
    text = normalize_unicode(text)
    text = remove_zero_width(text)
    text = normalize_punctuation(text)
    text = normalize_spacing(text)
    text = normalize_whitespace(text)
    norm = text
    lat_canon = normalize_homoglyphs_k_l(norm)
    cyr_canon = normalize_homoglyphs_l_k(norm)
    return {"raw": raw, "normalized": norm, "latin_canonical": lat_canon, "cyrillic_canon": cyr_canon}
