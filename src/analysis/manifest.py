"""Artifact manifest for the freeze / reproducibility check.

Hashes the frozen data and result files and summarises API usage per run. The
point is that a reader can tell *which* messages and predictions the reported
numbers came from, and how many provider calls each run cost.

Usage:
    python -m src.analysis.manifest
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from ..common import now_iso

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = [
    "data/_density_probe_paired/base/docs.jsonl",
    "data/_density_probe_paired/base/questions.jsonl",
    "data/_density_probe_paired/b100/docs.jsonl",
    "data/_density_probe_paired/b100/questions.jsonl",
    "data/_v3_candidate/b100/docs.jsonl",
    "data/_v3_candidate/b100/questions.jsonl",
    "data/_v3_paired/base/docs.jsonl",
    "data/_v3_paired/base/questions.jsonl",
    "data/_v3_paired/base/roles.jsonl",
    "data/_v3_paired/b100/docs.jsonl",
    "data/_v3_paired/b100/questions.jsonl",
    "data/_v3_paired/b100/roles.jsonl",
    "results/v3probe4/messages.jsonl",
    "results/v3probe4/predictions.jsonl",
    "results/v3probe4/predictions_4x4.jsonl",
    "results/v3base4/messages.jsonl",
    "results/v3base4/predictions.jsonl",
    "results/compact_probe/messages.jsonl",
    "results/compact_probe/predictions.jsonl",
    "results/compact_reserialize/messages.jsonl",
    "results/compact_reserialize/predictions.jsonl",
    "results/repeats/rep2/messages.jsonl",
    "results/repeats/rep2/predictions.jsonl",
    "results/repeats/rep3/messages.jsonl",
    "results/repeats/rep3/predictions.jsonl",
    "results/pilot_trans/messages.jsonl",
    "results/pilot_trans/predictions.jsonl",
    "results/messages_p0.9.jsonl",
    "results/predictions_p0.9.jsonl",
    "reports/density_2x2.json",
    "reports/variance_decomposition.json",
    "reports/compliance_analysis.json",
    "reports/compact_probe.json",
    "reports/error_examples.json",
    "reports/repeatability.json",
    "reports/token_budget.json",
    "reports/paired_corpus.json",
    "reports/main_plan.md",
    "results/main_v3/doc_manifest.json",
    "results/main_v3/messages_base.jsonl",
    "results/main_v3/messages_b100.jsonl",
    "results/main_v3/messages_b100_compact.jsonl",
    "results/main_v3/predictions_base.jsonl",
    "results/main_v3/predictions_b100.jsonl",
    "results/main_v3/predictions_base_cross.jsonl",
    "results/main_v3/predictions_b100_cross.jsonl",
    "results/main_v3/predictions_b100_compact.jsonl",
    "reports/main_results.json",
    "reports/compact_compare.json",
    "reports/main_base_matrix.json",
    "reports/main_b100_matrix.json",
    "results/main_v3/repeats/rep1/messages.jsonl",
    "results/main_v3/repeats/rep1/predictions.jsonl",
    "results/main_v3/repeats/rep2/messages.jsonl",
    "results/main_v3/repeats/rep2/predictions.jsonl",
    "results/main_v3/repeats/rep3/messages.jsonl",
    "results/main_v3/repeats/rep3/predictions.jsonl",
    "reports/repeatability_main.json",
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _count_records(path: Path) -> int | None:
    if path.suffix != ".jsonl":
        return None
    n = 0
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                n += 1
    return n


def _runs() -> dict:
    out = {}
    runs_dir = ROOT / "runs"
    for run_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        log = run_dir / "calls.jsonl"
        if not log.exists():
            continue
        n = cached = 0
        models = set()
        with log.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                rec = json.loads(line)
                n += 1
                cached += int(bool(rec.get("cached")))
                models.add(rec.get("model_key"))
        out[run_dir.name] = {
            "calls_logged": n,
            "fresh_calls": n - cached,
            "cached_hits": cached,
            "models": sorted(m for m in models if m),
        }
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="reports/artifact_manifest.json")
    args = parser.parse_args(argv)
    artifacts = []
    for rel in ARTIFACTS:
        path = ROOT / rel
        if not path.exists():
            artifacts.append({"path": rel, "exists": False})
            continue
        artifacts.append(
            {
                "path": rel,
                "exists": True,
                "bytes": path.stat().st_size,
                "n_records": _count_records(path),
                "sha256": _sha256(path),
                "mtime": path.stat().st_mtime,
            }
        )
    runs = _runs()
    manifest = {
        "generated_at": now_iso(),
        "note": "Hashes of the artefacts behind docs/preliminary_results.md; "
        "fresh_calls excludes local cache hits but includes retries.",
        "runs": runs,
        "total_fresh_calls_logged": sum(r["fresh_calls"] for r in runs.values()),
        "artifacts": artifacts,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    print(
        json.dumps(
            {
                "out": args.out,
                "runs": len(runs),
                "fresh_calls_logged": manifest["total_fresh_calls_logged"],
                "artifacts": sum(1 for a in artifacts if a["exists"]),
                "missing": [a["path"] for a in artifacts if not a["exists"]],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
