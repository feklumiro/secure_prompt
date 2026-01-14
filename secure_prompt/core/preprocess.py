import binascii
import re
import unicodedata
import base64


ZERO_WIDTH_CHARS = [
    "\u200b",  # zero width space
    "\u200c",  # zero width non-joiner
    "\u200d",  # zero width joiner
    "\ufeff",  # zero width no-break space
]


HOMOGLYPH_MAP = {
    "і": "i", "ӏ": "l", "І": "i", "3": "e", "0": "o"
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


def normalize_homoglyphs(text: str) -> str:
    return "".join(HOMOGLYPH_MAP.get(c, c) for c in text)


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
        print(text)
        new = re.sub(r"(?<!\w)([a-zа-я])\s+(?=[a-zа-я](?!\w))", r"\1", text)
        if new == text:
            break
        text = new
    return text


def normalize_whitespace(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def preprocess(text: str) -> str:
    text = try_decode_base64(text)
    text = text.lower()
    text = normalize_unicode(text)
    text = remove_zero_width(text)
    text = normalize_homoglyphs(text)
    text = normalize_punctuation(text)
    text = normalize_spacing(text)
    text = normalize_whitespace(text)
    print(text)
    return text
