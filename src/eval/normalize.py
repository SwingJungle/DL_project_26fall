"""Answer normalization (P6.1)."""
from __future__ import annotations

import re
import unicodedata

_ARTICLES = {"a", "an", "the"}
_NUM_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
_PUNCT_RE = re.compile(r"[^\w\s]|_", re.UNICODE)
_WS_RE = re.compile(r"\s+")
_NUM_RE = re.compile(r"-?\d+(?:[.,]\d+)?")

_UNKNOWN = {
    "unknown", "not mentioned", "not stated", "not specified", "unclear",
    "insufficient information", "cannot be determined", "can not be determined",
    "no information", "not available", "none", "n a", "na", "no answer",
}


def normalize(text: str | None) -> str:
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", str(text)).lower()
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    # strip thousands separators before punctuation removal so 1,200 -> 1200
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)
    text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    words = []
    for word in text.split():
        if word in _ARTICLES:
            continue
        if word in _NUM_WORDS:
            words.append(str(_NUM_WORDS[word]))
        elif re.fullmatch(r"\d{1,3}(?:,\d{3})+", word):
            words.append(word.replace(",", ""))
        else:
            words.append(word)
    return " ".join(words)


def tokens(text: str | None) -> list[str]:
    norm = normalize(text)
    return norm.split() if norm else []


def numeric_value(text: str | None) -> float | None:
    if text is None:
        return None
    norm = normalize(text)
    match = _NUM_RE.search(norm)
    if not match:
        return None
    value = match.group(0).replace(",", "")
    if "." in value:
        # "2.000" is more likely a thousands separator than a decimal here
        head, tail = value.split(".", 1)
        if len(tail) == 3:
            value = head + tail
    try:
        return float(value)
    except ValueError:
        return None


def is_unknown(text: str | None) -> bool:
    norm = normalize(text)
    if not norm:
        return False
    if norm in _UNKNOWN:
        return True
    return any(
        phrase in norm
        for phrase in (
            "not mentioned",
            "not stated",
            "no information",
            "insufficient information",
            "does not mention",
            "does not say",
            "not provided",
            "no mention",
        )
    )


def _contiguous(haystack: list[str], needle: list[str]) -> bool:
    if not needle or len(needle) > len(haystack):
        return False
    n = len(needle)
    return any(haystack[i : i + n] == needle for i in range(len(haystack) - n + 1))
