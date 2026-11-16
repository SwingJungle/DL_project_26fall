"""Fact-retention proxies for P0.9 (in-message retention).

Definition used for the G3 band: a **fact survives** if every key element of
the fact (numbers, rare proper nouns, and the literal gold span) is present in
the message. Key elements are the identity of a fact; the surrounding prose is
what compression is allowed to remove. This avoids punishing legitimate terse
rewrites (a token-recall threshold would score a perfectly good 8-word
statement as "lost" because the source sentence was padded).

``mean_fact_recall`` is kept as a lexical diagnostic only. The decisive P0.9
criterion remains receiver accuracy, which is robust to paraphrase.
"""
from __future__ import annotations

import json
import math
import re

_NUM_RE = re.compile(r"\d+(?:[.,]\d+)?")
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z\-']+|\d+(?:[.,]\d+)?")
_SPLIT_RE = re.compile(r"(?<=[.!?;])\s+|\n+")

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "than", "that", "this",
    "these", "those", "is", "are", "was", "were", "be", "been", "being", "to",
    "of", "in", "on", "at", "for", "from", "with", "by", "as", "it", "its",
    "has", "have", "had", "do", "does", "did", "not", "no", "never", "any",
    "all", "each", "every", "some", "such", "which", "who", "whom", "whose",
    "when", "where", "why", "how", "there", "their", "they", "them", "he",
    "she", "his", "her", "we", "our", "you", "your", "i", "my", "me", "us",
    "since", "until", "after", "before", "during", "between", "into", "over",
    "under", "above", "below", "more", "most", "less", "least", "only", "also",
    "one", "two", "three", "four", "five", "would", "could", "should", "may",
    "might", "must", "can", "will", "shall", "about", "up", "out", "so",
    "because", "while", "through", "per", "own", "same", "other", "both",
}


def _norm_number(value: str) -> str:
    return value.replace(",", "")


def _numbers(text: str) -> set[str]:
    return {_norm_number(m) for m in _NUM_RE.findall(text)}


def content_tokens(text: str) -> list[str]:
    return [
        t for t in re.findall(r"[a-z0-9]+", text.lower())
        if (t not in STOPWORDS) and (len(t) > 1 or t.isdigit())
    ]


def key_entities(text: str) -> list[str]:
    """Numbers plus proper nouns inside a fact sentence.

    Only mid-sentence capitalized tokens count: sentence-initial capitals are
    ambiguous ("Every", "Any", "Responsibility"), while fabricated names such
    as "Kulgel" or "Tiathban" appear capitalized inside the clause.
    """
    out: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        for i, tok in enumerate(_TOKEN_RE.findall(sentence)):
            if _NUM_RE.fullmatch(tok):
                out.append(_norm_number(tok))
            elif i > 0 and tok[0].isupper() and len(tok) > 2:
                out.append(tok.lower())
    return list(dict.fromkeys(out))


def _literal_gold_tokens(fact: dict) -> list[str]:
    gold = content_tokens(fact.get("gold") or "")
    fact_tokens = content_tokens(fact.get("text") or "")
    if not gold or not fact_tokens:
        return []
    n = len(gold)
    for i in range(len(fact_tokens) - n + 1):
        if fact_tokens[i : i + n] == gold:
            return gold
    return []


def message_tokens(message: str) -> tuple[set[str], set[str]]:
    """(all lowercase alphanumeric tokens, normalized numbers)."""
    tokens = {t for t in re.findall(r"[a-z0-9]+", message.lower())}
    return tokens, _numbers(message)


def message_segments(message: str, method: str) -> list[str]:
    if method == "structured":
        try:
            data = json.loads(message)
            return [str(item.get("x", "")) for item in data.get("f", [])] or [message]
        except (json.JSONDecodeError, AttributeError):
            return [message]
    if method == "compact":
        try:
            data = json.loads(message)
            return [str(item[1]) for item in data if isinstance(item, list) and len(item) == 2] or [message]
        except (json.JSONDecodeError, AttributeError, TypeError, IndexError):
            return [message]
    parts = [p.strip() for p in _SPLIT_RE.split(message) if p.strip()]
    return parts or [message]


def fact_keys(fact: dict) -> list[str]:
    keys = key_entities(fact.get("text", ""))
    for tok in _literal_gold_tokens(fact):
        if tok not in keys:
            keys.append(tok)
    return keys


def fact_scores(doc: dict, message: str, method: str) -> list[dict]:
    tokens, numbers = message_tokens(message)
    segments = message_segments(message, method)
    seg_tokens = [set(content_tokens(s)) for s in segments]
    out = []
    for fact in doc.get("facts", []):
        fact_tokens = set(content_tokens(fact.get("text", "")))
        best = 0.0
        if fact_tokens:
            best = max((len(fact_tokens & st) / len(fact_tokens) for st in seg_tokens), default=0.0)

        fact_numbers = _numbers(fact.get("text", ""))
        gold_tokens = _literal_gold_tokens(fact)
        if gold_tokens:
            # the queried span is literal in the source: it must survive verbatim
            survived = all(t in tokens for t in gold_tokens) and fact_numbers <= numbers
            keys = gold_tokens
        else:
            # clause-level gold: require all numbers and most names to survive
            keys = key_entities(fact.get("text", ""))
            present = sum(1 for k in keys if (k in numbers if _NUM_RE.fullmatch(k) else k in tokens))
            survived = bool(keys) and fact_numbers <= numbers and present >= max(1, math.ceil(0.6 * len(keys)))
        out.append(
            {
                "fid": fact.get("fid"),
                "type": fact.get("type"),
                "keys": keys,
                "recall": round(best, 3),
                "numbers_ok": fact_numbers <= numbers,
                "survived": survived,
            }
        )
    return out


def question_evidence(doc: dict, message: str, questions: list[dict], method: str) -> list[dict]:
    facts = {f["fid"]: f for f in doc.get("facts", [])}
    scores = {s["fid"]: s for s in fact_scores(doc, message, method)}
    out = []
    for q in questions:
        if not q.get("answerable", True):
            continue
        fact_ids = [fid for fid in q.get("fact_ids", []) if fid in facts]
        if not fact_ids:
            out.append({"qid": q["qid"], "evidence": False, "reason": "no_fact_ids"})
            continue
        facts_ok = all(scores.get(fid, {}).get("survived", False) for fid in fact_ids)
        out.append(
            {
                "qid": q["qid"],
                "demand": q.get("demand"),
                "evidence": facts_ok,
                "fact_ids": fact_ids,
            }
        )
    return out


def retention_summary(doc: dict, message: str, questions: list[dict], method: str) -> dict:
    fs = fact_scores(doc, message, method)
    qe = question_evidence(doc, message, questions, method)
    n_facts = len(fs)
    n_survived = sum(1 for s in fs if s["survived"])
    q_total = len(qe)
    q_ok = sum(1 for r in qe if r["evidence"])
    return {
        "n_facts": n_facts,
        "n_facts_survived": n_survived,
        "fact_retention": round(n_survived / n_facts, 4) if n_facts else 0.0,
        "mean_fact_recall": round(sum(s["recall"] for s in fs) / n_facts, 4) if n_facts else 0.0,
        "n_numbers_missing": sum(1 for s in fs if not s["numbers_ok"]),
        "n_evidence_questions": q_total,
        "n_evidence_present": q_ok,
        "question_evidence_rate": round(q_ok / q_total, 4) if q_total else 0.0,
        "facts": fs,
        "questions": qe,
    }
