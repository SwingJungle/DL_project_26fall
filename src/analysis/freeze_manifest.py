"""Freeze manifest: hash everything the confirmatory run depends on.

The repository has no git, so the freeze point is recorded as a digest manifest
instead of a commit. It covers the paired corpora, the prompt/receiver code (the
prompt text lives in ``src/compress/prompts.py``, not in a config file), the
analysis scripts, the signed plan and the doc manifest.

``combined_digest`` is a single sha256 over the sorted ``path: sha256`` lines, so
two checkouts can be proven identical without diffing every file.

Usage:
    python -m src.analysis.freeze_manifest            # write frozen/freeze_v3_manifest.json
    python -m src.analysis.freeze_manifest --check    # verify the tree still matches
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ..common import now_iso

ROOT = Path(__file__).resolve().parents[2]

#: data collection: changing any of this means the main run is not comparable
CORPORA = [
    "data/_v3_paired/base/docs.jsonl",
    "data/_v3_paired/base/questions.jsonl",
    "data/_v3_paired/base/roles.jsonl",
    "data/_v3_paired/b100/docs.jsonl",
    "data/_v3_paired/b100/questions.jsonl",
    "data/_v3_paired/b100/roles.jsonl",
]
CONFIG = [
    "config/models.json",
    "config/budget.json",
    "config/schema.json",
]
#: prompt and receiver text lives here; a change invalidates the pre-registration
PROMPT_CODE = ["src/compress/prompts.py", "src/compress/pipeline.py", "src/run/receiver.py"]
RUN_CODE = [
    "src/api/adapters.py", "src/api/cache.py", "src/api/calls.py",
    "src/api/config.py", "src/api/cost.py", "src/api/retry.py",
    "src/run/sender.py", "src/run/data.py", "src/run/cli.py",
]
#: documents the exact eval selection used by the run
SINGLE = ["results/main_v3/doc_manifest.json"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collection_paths() -> list[str]:
    return sorted(dict.fromkeys(list(CORPORA) + list(CONFIG) + list(PROMPT_CODE)
                                + list(RUN_CODE) + list(SINGLE)))


def analysis_paths() -> list[str]:
    paths = [str(p.relative_to(ROOT)) for p in sorted((ROOT / "src").rglob("*.py"))]
    paths += [str(p.relative_to(ROOT)) for p in sorted((ROOT / "tests").rglob("*.py"))]
    paths += ["docs/main_runbook.md", "docs/main_experiment_results.md",
              "docs/freeze_v3_plan.md",
              "reports/paired_corpus.md", "reports/main_plan.md",
              "reports/main_results.md", "reports/compact_compare.md",
              "reports/compliance_analysis.md", "reports/variance_decomposition.md"]
    return sorted(dict.fromkeys(paths))


def collect() -> list[str]:
    return sorted(dict.fromkeys(collection_paths() + analysis_paths()))


def _digest(entries: list[dict]) -> str:
    lines = [f"{e['path']}: {e['sha256']}" for e in entries if e["sha256"]]
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def build() -> dict:
    entries = []
    for rel in collect():
        path = ROOT / rel
        if not path.exists():
            entries.append({"path": rel, "sha256": None, "bytes": None, "missing": True})
            continue
        entries.append({"path": rel, "sha256": sha256_file(path),
                        "bytes": path.stat().st_size, "missing": False})
    collection = [e for e in entries if e["path"] in set(collection_paths())]
    analysis = [e for e in entries if e["path"] in set(analysis_paths())]
    combined = _digest(entries)
    from ..compress.prompts import (
        PROMPT_VERSION_ABS,
        PROMPT_VERSION_COMPACT,
        PROMPT_VERSION_RECEIVER,
        PROMPT_VERSION_RECEIVER_COMPACT,
        PROMPT_VERSION_RECEIVER_FULL,
        PROMPT_VERSION_STRUCT,
    )
    return {
        "generated_at": now_iso(),
        "combined_digest": combined,
        "collection_digest": _digest(collection),
        "analysis_digest": _digest(analysis),
        "n_collection_files": len(collection),
        "n_analysis_files": len(analysis),
        "n_files": len(entries),
        "n_missing": sum(1 for e in entries if e["missing"]),
        "prompt_versions": {
            "sender_abstractive": PROMPT_VERSION_ABS,
            "sender_structured": PROMPT_VERSION_STRUCT,
            "sender_compact": PROMPT_VERSION_COMPACT,
            "receiver": PROMPT_VERSION_RECEIVER,
            "receiver_compact": PROMPT_VERSION_RECEIVER_COMPACT,
            "receiver_full": PROMPT_VERSION_RECEIVER_FULL,
        },
        "corpora": list(CORPORA),
        "files": entries,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="frozen/freeze_v3_manifest.json")
    parser.add_argument("--check", action="store_true",
                        help="compare the current tree with the stored manifest")
    args = parser.parse_args(argv)

    report = build()
    if args.check:
        stored = json.loads((ROOT / args.out).read_text(encoding="utf-8"))
        if stored["combined_digest"] == report["combined_digest"]:
            print(f"OK: tree matches {args.out} ({report['combined_digest'][:16]}…)")
            return 0
        now = {e["path"]: e["sha256"] for e in report["files"]}
        then = {e["path"]: e["sha256"] for e in stored["files"]}
        changed = sorted(p for p in now.keys() | then.keys() if now.get(p) != then.get(p))
        print(f"DRIFT ({len(changed)}):")
        for path in changed:
            print(f"  {path}")
        return 1

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}: {report['n_files']} files, "
          f"{report['n_missing']} missing")
    print(f"  collection {report['collection_digest'][:16]}… "
          f"({report['n_collection_files']} files)")
    print(f"  analysis   {report['analysis_digest'][:16]}… "
          f"({report['n_analysis_files']} files)")
    print("prompt versions:", report["prompt_versions"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
