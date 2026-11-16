"""Zero-call planner for the main experiment (call counts, doc manifest, budget).

The main run is staged (sender -> self receiver -> 4x4 subset -> compact) and each
stage is capped with ``--max-calls``; this script computes what those numbers are
for a given corpus pair *before* any paid call is made, and records the frozen
document manifest (eval doc_ids + corpus hashes + seed) that
``docs/freeze_v3_plan.md`` §10 asks for.

It does not touch the API and does not read ``runs/`` -- only the corpora and the
measured per-call token averages in ``reports/token_budget.json``.

Usage
-----
    python -m src.run.plan_main \
        --densities data/_v3_paired/base,data/_v3_paired/b100 \
        --split eval --docs 100 --cross-subset 30 --include-compact
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ..api.config import all_model_keys
from .data import load_jsonl

#: read from config/models.json so the plan cannot drift from the runner
MODELS = tuple(all_model_keys())
METHODS = ("abstractive", "structured")
#: the compact ablation is one extra representation, not a third main method
COMPACT = ("compact",)

#: fallback per-call token averages if reports/token_budget.json is missing
FALLBACK_TOKENS = {"sender": 1043, "receiver": 841, "full": 899}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def corpus_summary(path: Path) -> dict:
    docs = load_jsonl(path / "docs.jsonl")
    questions = load_jsonl(path / "questions.jsonl")
    bpf = [d["bytes_per_fact"] for d in docs]
    return {
        "path": str(path),
        "n_docs": len(docs),
        "n_questions": len(questions),
        "bpf_mean": round(sum(bpf) / len(bpf), 1) if bpf else None,
        "sha256_docs": sha256_file(path / "docs.jsonl"),
        "sha256_questions": sha256_file(path / "questions.jsonl"),
        "docs": docs,
    }


def tokens_per_call() -> dict:
    path = Path("reports/token_budget.json")
    if not path.exists():
        return dict(FALLBACK_TOKENS)
    measured = json.loads(path.read_text(encoding="utf-8")).get("measured", {}).get("by_type", {})
    out = {}
    for kind, fallback in FALLBACK_TOKENS.items():
        block = measured.get(kind) or {}
        per_call = (block.get("prompt_per_call") or 0) + (block.get("completion_per_call") or 0)
        out[kind] = round(per_call) if per_call else fallback
    return out


def build_plan(
    corpora: list[dict],
    *,
    n_docs: int,
    cross_subset: int,
    include_compact: bool,
    include_full: bool,
    retry_factor: float,
    tokens: dict,
) -> dict:
    stages = []
    total_calls = 0
    total_tokens = 0

    def add(stage: str, calls: int, kind: str, note: str) -> None:
        nonlocal total_calls, total_tokens
        tok = calls * tokens[kind]
        total_calls += calls
        total_tokens += tok
        stages.append({
            "stage": stage, "calls": calls, "tokens": tok, "kind": kind, "note": note,
        })

    n_models = len(MODELS)
    n_methods = len(METHODS)
    n_densities = len(corpora)

    # 1) sender: every document x density x method x model
    sender = n_docs * n_densities * n_methods * n_models
    add("sender", sender, "sender", f"{n_docs} docs x {n_densities} densities x {n_methods} methods x {n_models} models")

    # 2) self receiver: one prediction per message (receiver = sender)
    add("receiver_self", sender, "receiver", "self receiver, 1 per message")

    # 3) full-document upper bound baseline: every doc x density x 4 receivers
    if include_full:
        add("receiver_full", n_docs * n_densities * n_models, "full",
            f"{n_docs} docs x {n_densities} densities x {n_models} receivers")

    # 4) 4x4 subset: same messages, three extra receivers each
    if cross_subset:
        cross = cross_subset * n_densities * n_methods * n_models * (n_models - 1)
        add("receiver_4x4", cross, "receiver",
            f"{cross_subset} docs x {n_densities} densities x {n_methods} methods x "
            f"{n_models} senders x {n_models - 1} extra receivers")

    # 5) compact ablation: b100 only, self receiver
    if include_compact:
        b100 = next(c for c in corpora if c["is_dense"])
        add("compact_sender", n_docs * n_models, "sender",
            f"compact, b100 only: {n_docs} docs x {n_models} senders")
        add("compact_receiver", n_docs * n_models, "receiver",
            f"compact, b100 only: {n_docs} messages x self receiver")

    nominal = total_calls
    budgeted = int(round(nominal * retry_factor))
    per_stage = []
    for s in stages:
        cap = int(round(s["calls"] * retry_factor / 100.0) * 100)
        per_stage.append({**s, "max_calls": max(100, cap)})
    return {
        "stages": per_stage,
        "nominal_calls": nominal,
        "retry_factor": retry_factor,
        "budgeted_calls": budgeted,
        "tokens_nominal": total_tokens,
        "tokens_budgeted": int(round(total_tokens * retry_factor)),
        "tokens_per_call": tokens,
    }


def render_markdown(plan: dict, manifest: dict) -> str:
    lines = [
        "# 主实验 0 调用排期（plan_main）",
        "",
        "> 脚本：`src/run/plan_main.py`；不联网、不读 `runs/`，只读语料与",
        "> `reports/token_budget.json` 的实测均值。所有数字都是**计划值**，",
        "> 实际调用以 `runs/<run_id>/calls.jsonl` 为准。",
        "",
        f"- 语料：`{'`, `'.join(c['path'] for c in manifest['corpora'])}`",
        f"- 选中文档：{manifest['n_docs']} 篇 / 密度（split={manifest['split']}，"
        f"每档共 {manifest['corpora'][0]['n_docs']} 篇）",
        f"- 模型：{', '.join(MODELS)}",
        "",
        "## 阶段与封顶",
        "",
        "| 阶段 | 名义 calls | 预估 tokens | 建议 `--max-calls` | 说明 |",
        "|---|---:|---:|---:|---|",
    ]
    for s in plan["stages"]:
        lines.append(f"| `{s['stage']}` | {s['calls']:,} | {s['tokens']:,} | "
                     f"{s['max_calls']:,} | {s['note']} |")
    lines += [
        "",
        f"**合计**：名义 {plan['nominal_calls']:,} calls / {plan['tokens_nominal']:,} tokens；"
        f"按 retry×{plan['retry_factor']:g} 计 {plan['budgeted_calls']:,} calls / "
        f"{plan['tokens_budgeted']:,} tokens。",
        "",
        "单次调用 token 均值（实测）：sender "
        f"{plan['tokens_per_call']['sender']:,}，receiver "
        f"{plan['tokens_per_call']['receiver']:,}，full "
        f"{plan['tokens_per_call']['full']:,}。",
        "",
        "## 口径说明（与旧文档对账）",
        "",
        "- **全文档基线按密度各跑一次**：配对设计要求 base 与 b100 的源文本都做上界，",
        "  所以是 100 × 2 × 4 = 800 次；`docs/freeze_v3_plan.md` §9 里写的 400 只算了",
        "  一档密度（pilot 的 95.1% / 95.8% 就是两档都测的）。",
        "- **`reports/token_budget.md` 的 B/C 方案把 retries 折进了 calls**，本表分开列",
        "  “名义” 与 “按 retry×1.4 计”。名义 6,240（含 compact）≈ 该表 C 的 6,640（含",
        "  retries），不是两套设计。",
        "- **必须带 `--corpus-id base|b100`**：`message_key` 只有 "
        "`doc_id|model|method|level`，两档密度的 key 会相同；不带 tag 的断点续跑会把",
        "  第二档当成已完成而静默跳过（见 `tests/test_corpus_id.py` 与 `docs/handoff_next_session_v4.md` §2.1）。",
        "",
        "## 语料哈希（冻结用）",
        "",
        "| 语料 | docs | questions | 平均 B/fact | sha256(docs) | sha256(questions) |",
        "|---|---:|---:|---:|---|---|",
    ]
    for c in manifest["corpora"]:
        lines.append(f"| `{c['path']}` | {c['n_docs']} | {c['n_questions']} | "
                     f"{c['bpf_mean']} | `{c['sha256_docs'][:16]}…` | "
                     f"`{c['sha256_questions'][:16]}…` |")
    lines += [
        "",
        "## eval 文档清单",
        "",
        f"seed=20261111，`p-*` 编号连续；取 `{manifest['split']}` 前 "
        f"{manifest['n_docs']} 篇：",
        "",
        "```",
        ", ".join(manifest["doc_ids"]),
        "```",
        "",
        "> 完整清单与每篇字数在 `results/main_v3/doc_manifest.json`。",
        "",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--densities", default="data/_v3_paired/base,data/_v3_paired/b100",
                        help="comma-separated corpus dirs; the last one is the dense one")
    parser.add_argument("--split", default="eval")
    parser.add_argument("--docs", type=int, default=100)
    parser.add_argument("--cross-subset", type=int, default=30,
                        help="docs in the 4x4 receiver subset; 0 disables it")
    parser.add_argument("--include-compact", action="store_true")
    parser.add_argument("--no-full", action="store_true", help="skip the full-document baseline")
    parser.add_argument("--retry-factor", type=float, default=1.4,
                        help="call multiplier for retries (measured: ~1.4x)")
    parser.add_argument("--out", default="results/main_v3/doc_manifest.json")
    parser.add_argument("--report", default="reports/main_plan.md")
    args = parser.parse_args(argv)

    corpora = []
    for i, raw in enumerate(args.densities.split(",")):
        path = Path(raw.strip())
        summary = corpus_summary(path)
        summary["is_dense"] = i == len(args.densities.split(",")) - 1
        corpora.append(summary)

    if len({len(c["docs"]) for c in corpora}) != 1:
        raise SystemExit("density corpora have different document counts")
    first_ids = [d["doc_id"] for d in corpora[0]["docs"]]
    for c in corpora[1:]:
        if [d["doc_id"] for d in c["docs"]] != first_ids:
            raise SystemExit(f"{c['path']} does not share the doc_id order of {corpora[0]['path']}")

    selected = [d for d in corpora[0]["docs"] if d["split"] == args.split][: args.docs]
    if len(selected) < args.docs:
        raise SystemExit(f"split {args.split!r} has {len(selected)} docs, {args.docs} requested")
    doc_ids = [d["doc_id"] for d in selected]

    tokens = tokens_per_call()
    plan = build_plan(corpora,
                      n_docs=len(doc_ids),
                      cross_subset=args.cross_subset,
                      include_compact=args.include_compact,
                      include_full=not args.no_full,
                      retry_factor=args.retry_factor,
                      tokens=tokens)

    manifest = {
        "split": args.split,
        "n_docs": len(doc_ids),
        "doc_ids": doc_ids,
        "seed": 20261111,
        "corpora": [{k: v for k, v in c.items() if k != "docs"} for c in corpora],
        "plan": plan,
        "cross_subset_doc_ids": doc_ids[: args.cross_subset] if args.cross_subset else [],
        "models": list(MODELS),
        "methods": list(METHODS),
        "notes": [
            "plan only: no provider calls were made",
            "retry factor is an allowance, not a promise; each stage is capped with --max-calls",
            "doc order is the corpus order; eval docs are the first --docs of split=eval",
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(render_markdown(plan, manifest) + "\n", encoding="utf-8")

    print(f"wrote {args.out} and {args.report}")
    print(f"docs={len(doc_ids)}  nominal={plan['nominal_calls']:,} calls  "
          f"budgeted={plan['budgeted_calls']:,}  tokens~{plan['tokens_budgeted']:,}")
    for s in plan["stages"]:
        print(f"  {s['stage']:<20} {s['calls']:>6,} calls  cap {s['max_calls']:>6,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
