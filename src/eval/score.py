"""Answer scoring (P6.2 / P6.3).

``correct`` is deliberately lenient about surface form (the corpus contains
clause-level golds, see primary_set_spec.md §12.6) but strict about not
rewarding fabricated answers: a prediction only matches if it contains the gold
span, the gold contains the prediction with >= 60% coverage, or the two
numeric values are equal.

``unknown`` is only correct for questions whose gold is ``unknown``; for
answerable questions an ``unknown`` prediction is wrong by definition.
"""
from __future__ import annotations

from collections import Counter

from .normalize import is_unknown, normalize, numeric_value, tokens


def token_f1(pred: str | None, gold: str | None) -> float:
    p, g = tokens(pred), tokens(gold)
    if not p or not g:
        return 1.0 if p == g else 0.0
    common = Counter(p) & Counter(g)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(p)
    recall = overlap / len(g)
    return 2 * precision * recall / (precision + recall)


def match(pred: str | None, gold: str | None, answerable: bool) -> dict:
    if not answerable or normalize(gold) == "unknown":
        ok = is_unknown(pred)
        return {"correct": ok, "match_type": "unknown" if ok else "wrong_unknown", "f1": 1.0 if ok else 0.0}

    if is_unknown(pred):
        return {"correct": False, "match_type": "predicted_unknown", "f1": 0.0}

    npred, ngold = normalize(pred), normalize(gold)
    if npred and npred == ngold:
        return {"correct": True, "match_type": "exact", "f1": 1.0}

    pv, gv = numeric_value(pred), numeric_value(gold)
    if pv is not None and gv is not None and abs(pv - gv) < 1e-9:
        return {"correct": True, "match_type": "numeric", "f1": token_f1(pred, gold)}

    pt, gt = tokens(pred), tokens(gold)
    if pt and gt:
        if _contains(pt, gt):
            return {"correct": True, "match_type": "gold_in_pred", "f1": token_f1(pred, gold)}
        if _contains(gt, pt) and len(pt) >= 0.6 * len(gt):
            return {"correct": True, "match_type": "pred_in_gold", "f1": token_f1(pred, gold)}
    return {"correct": False, "match_type": "mismatch", "f1": token_f1(pred, gold)}


def _contains(haystack: list[str], needle: list[str]) -> bool:
    if not needle or len(needle) > len(haystack):
        return False
    n = len(needle)
    return any(haystack[i : i + n] == needle for i in range(len(haystack) - n + 1))


def score_answers(questions: list[dict], answers: dict[str, str]) -> dict:
    rows = []
    for q in questions:
        pred = answers.get(q["qid"])
        result = match(pred, q.get("gold"), bool(q.get("answerable", True)))
        rows.append(
            {
                "qid": q["qid"],
                "demand": q.get("demand"),
                "answerable": q.get("answerable", True),
                "gold": q.get("gold"),
                "pred": pred,
                "correct": result["correct"],
                "match_type": result["match_type"],
                "f1": result["f1"],
            }
        )
    n = len(rows)
    correct = sum(1 for r in rows if r["correct"])
    lookup = [r for r in rows if r["demand"] == "lookup"]
    multihop = [r for r in rows if r["demand"] == "multi-hop"]
    return {
        "rows": rows,
        "accuracy": correct / n if n else 0.0,
        "n": n,
        "lookup_accuracy": sum(r["correct"] for r in lookup) / len(lookup) if lookup else 0.0,
        "multihop_accuracy": sum(r["correct"] for r in multihop) / len(multihop) if multihop else 0.0,
        "mean_f1": sum(r["f1"] for r in rows) / n if n else 0.0,
    }
