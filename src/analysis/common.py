"""Analysis helpers shared by the zero-call and probe reports.

Everything here is pure standard library: figures (which need matplotlib) live
in ``src.analysis.figures`` and are the only optional dependency.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict


def read_jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def accuracy(rows: list[dict]) -> tuple[int, int, float | None]:
    n = len(rows)
    if not n:
        return 0, 0, None
    k = sum(int(bool(r.get("correct"))) for r in rows)
    return k, n, k / n


def flat_questions(
    predictions: list[dict],
    *,
    methods: set[str] | None = None,
    levels: set[str] | None = None,
    pair_types: set[str] | None = None,
) -> list[dict]:
    """Flatten prediction records to one row per (record, question)."""
    out = []
    for rec in predictions:
        if methods is not None and rec["method"] not in methods:
            continue
        if levels is not None and rec["level"] not in levels:
            continue
        if pair_types is not None and rec["pair_type"] not in pair_types:
            continue
        for q in rec.get("per_question") or []:
            out.append(
                {
                    "doc_id": rec["doc_id"],
                    "sender_model": rec.get("sender_model"),
                    "receiver_model": rec.get("receiver_model"),
                    "method": rec["method"],
                    "level": rec["level"],
                    "pair_type": rec["pair_type"],
                    "message_bytes": rec.get("message_bytes"),
                    "qid": q["qid"],
                    "demand": q.get("demand"),
                    "answerable": q.get("answerable"),
                    "correct": bool(q.get("correct")),
                }
            )
    return out


def group_accuracy(rows: list[dict], *keys: str) -> dict[tuple, dict]:
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        buckets[tuple(row[k] for k in keys)].append(row)
    out = {}
    for key, items in buckets.items():
        k, n, acc = accuracy(items)
        out[key] = {"n_correct": k, "n_questions": n, "accuracy": acc}
    return out


def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value (binomial on discordant pairs).

    ``b`` = A correct / B wrong, ``c`` = A wrong / B correct. No continuity
    correction: with these sample sizes the exact test is preferable to chi2.
    """
    n = b + c
    if n == 0:
        return 1.0
    lo = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, lo + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def paired_questions(
    rows_a: list[dict],
    rows_b: list[dict],
    *,
    keys: tuple[str, ...] = ("doc_id", "sender_model", "receiver_model", "qid"),
) -> list[tuple[str, dict, dict]]:
    """Pair two conditions at the question level on ``keys``.

    Returns (doc_id, row_a, row_b) for questions present in both conditions.
    Duplicate keys (e.g. two levels of one method) raise: pairing must be
    unambiguous, otherwise a silent cross-product would bias the test.
    """
    def index(rows):
        out = {}
        for r in rows:
            key = tuple(r[k] for k in keys)
            if key in out:
                raise ValueError(f"duplicate pairing key {key}")
            out[key] = r
        return out

    a, b = index(rows_a), index(rows_b)
    shared = sorted(set(a) & set(b))
    return [(a[key]["doc_id"], a[key], b[key]) for key in shared]


def paired_summary(pairs: list[tuple[str, dict, dict]]) -> dict:
    """Agreement table + discordant counts + naive accuracy difference."""
    both = sum(1 for _, a, b in pairs if a["correct"] and b["correct"])
    neither = sum(1 for _, a, b in pairs if not a["correct"] and not b["correct"])
    b_only = sum(1 for _, a, b in pairs if a["correct"] and not b["correct"])
    c_only = sum(1 for _, a, b in pairs if not a["correct"] and b["correct"])
    n = len(pairs)
    acc_a = (both + b_only) / n if n else None
    acc_b = (both + c_only) / n if n else None
    return {
        "n_pairs": n,
        "both_correct": both,
        "both_wrong": neither,
        "a_only_correct": b_only,
        "b_only_correct": c_only,
        "accuracy_a": acc_a,
        "accuracy_b": acc_b,
        "diff_a_minus_b": (acc_a - acc_b) if n else None,
        "mcnemar_p": mcnemar_exact(b_only, c_only),
    }


def bootstrap_paired(
    pairs: list[tuple[str, dict, dict]],
    *,
    n_boot: int = 4000,
    seed: int = 11,
) -> tuple[float | None, float | None]:
    """Doc-cluster bootstrap CI for accuracy(a) - accuracy(b).

    Questions inside one document share a message and a fact set, so the
    document is the resampling unit; a question-level bootstrap would be
    anticonservative.
    """
    by_doc: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
    for doc, a, b in pairs:
        by_doc[doc].append((a, b))
    docs = sorted(by_doc)
    if not docs:
        return None, None
    import random

    rng = random.Random(seed)
    diffs = []
    for _ in range(n_boot):
        sample = [rng.choice(docs) for _ in docs]
        na = nb = k = 0
        for doc in sample:
            for a, b in by_doc[doc]:
                na += int(a["correct"])
                nb += int(b["correct"])
                k += 1
        if k:
            diffs.append((na - nb) / k)
    diffs.sort()
    return diffs[int(0.025 * len(diffs))], diffs[int(0.975 * len(diffs))]


def bootstrap_interaction(
    pairs_abs: list[tuple[str, dict, dict]],
    pairs_struct: list[tuple[str, dict, dict]],
    *,
    n_boot: int = 4000,
    seed: int = 11,
) -> tuple[float | None, float | None, float | None]:
    """Doc-cluster bootstrap for (struct_a - struct_b) - (abs_a - abs_b).

    ``pairs_*`` are (doc, dense/method-X, sparse/method-Y) pairs from the *same*
    documents; resampling docs keeps the two methods paired.
    """
    def by_doc(pairs):
        out: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
        for doc, a, b in pairs:
            out[doc].append((a, b))
        return out

    abs_doc, struct_doc = by_doc(pairs_abs), by_doc(pairs_struct)
    docs = sorted(set(abs_doc) & set(struct_doc))
    if not docs:
        return None, None, None

    def stat(docs_sample):
        def delta(by_doc):
            na = nb = k = 0
            for d in docs_sample:
                for a, b in by_doc[d]:
                    na += int(a["correct"])
                    nb += int(b["correct"])
                    k += 1
            return (na - nb) / k if k else 0.0

        return delta(struct_doc) - delta(abs_doc)

    point = stat(docs)
    import random

    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        sample = [rng.choice(docs) for _ in docs]
        vals.append(stat(sample))
    vals.sort()
    return point, vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals))]


def fmt_pp(value, digits: int = 1) -> str:
    if value is None:
        return "-"
    return f"{100 * value:+.{digits}f}pp"


def write_json(path, obj) -> None:
    import os

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
