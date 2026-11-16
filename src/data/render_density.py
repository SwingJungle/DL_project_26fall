"""Re-render the same documents at a different source density (paired probe).

The base corpus must have been generated with ``--save-roles``. Questions are
role-based and therefore identical across densities; only the sentence
realisations change. This gives a within-document density comparison: same
doc_id, same facts, same questions, same gold, different words/bytes per fact.

Usage:
    python -m src.data.render_density \
        --base data/_density_probe_paired/base \
        --bpf 100 --out data/_density_probe_paired/b100
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
from pathlib import Path

from . import frameworks as F
from . import generate_v3 as G


def stable_seed(doc_id: str, bpf: float, attempt: int = 0) -> int:
    payload = f"{doc_id}|{bpf:g}|{attempt}".encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:8], 16)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def render_one(
    base_doc: dict,
    roles: dict,
    slots: list[str],
    questions: list[dict],
    bpf: float,
    usage: dict[str, collections.Counter],
    attempt: int,
) -> tuple[dict | None, list[str]]:
    G.configure_density(bpf)
    rng = random.Random(stable_seed(base_doc["doc_id"], bpf, attempt))
    chosen = G.pick_texts(slots, roles, rng, usage)
    facts = []
    for pos, (slot, idx) in enumerate(zip(slots, chosen)):
        template = F.TEMPLATES[slot][idx]
        facts.append(
            {
                "fid": f"f{pos + 1}",
                "renderer": slot,
                "type": F.SLOT_TYPE[slot],
                "sentence_idx": pos,
                "text": template.format(**roles),
                "gold": F.GOLD[slot].format(**roles),
            }
        )
    text = " ".join(f["text"] for f in facts)
    doc = {
        "doc_id": base_doc["doc_id"],
        "split": base_doc["split"],
        "domain": base_doc["domain"],
        "text": text,
        "words": len(text.split()),
        "bytes": len(text.encode("utf-8")),
        "n_facts": len(facts),
        "bytes_per_fact": round(len(text.encode("utf-8")) / len(facts), 1),
        "facts": facts,
        "questions": questions,
    }
    return doc, G.validate(doc)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="base corpus dir (with roles.jsonl)")
    ap.add_argument("--bpf", type=float, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-attempts", type=int, default=40)
    args = ap.parse_args()

    base_dir = Path(args.base)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    docs = read_jsonl(base_dir / "docs.jsonl")
    roles_rows = {r["doc_id"]: r for r in read_jsonl(base_dir / "roles.jsonl")}
    questions = collections.defaultdict(list)
    for q in read_jsonl(base_dir / "questions.jsonl"):
        questions[q["doc_id"]].append(q)

    G.configure_density(args.bpf)
    usage: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    rendered: list[dict] = []
    failed: list[tuple[str, list[str]]] = []
    attempts_total = 0

    for base_doc in docs:
        meta = roles_rows[base_doc["doc_id"]]
        doc = None
        problems: list[str] = []
        for attempt in range(args.max_attempts):
            attempts_total += 1
            doc, problems = render_one(
                base_doc, meta["roles"], meta["slots"], questions[base_doc["doc_id"]],
                args.bpf, usage, attempt,
            )
            if not problems:
                break
        if problems:
            failed.append((base_doc["doc_id"], problems))
            doc = None
        if doc is not None:
            rendered.append(doc)

    with (out_dir / "docs.jsonl").open("w", encoding="utf-8") as fh:
        for d in rendered:
            payload = {k: v for k, v in d.items() if k not in ("questions", "roles")}
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
    # questions are role-based: copy the base file unchanged
    (out_dir / "questions.jsonl").write_text(
        (base_dir / "questions.jsonl").read_text(encoding="utf-8"), encoding="utf-8"
    )

    import statistics

    wpf = [d["words"] / d["n_facts"] for d in rendered]
    bpf = [d["bytes_per_fact"] for d in rendered]
    print(f"base={base_dir}  target bpf={args.bpf:g}")
    print(f"rendered {len(rendered)}/{len(docs)} docs; failed {len(failed)}; attempts {attempts_total}")
    if rendered:
        print(f"words/fact : min {min(wpf):.1f} mean {statistics.mean(wpf):.1f} max {max(wpf):.1f}")
        print(f"bytes/fact : min {min(bpf):.1f} mean {statistics.mean(bpf):.1f} max {max(bpf):.1f}")
    for doc_id, problems in failed[:10]:
        print(f"  FAIL {doc_id}: {problems[:4]}")
    print(f"wrote {out_dir}/docs.jsonl and questions.jsonl")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
