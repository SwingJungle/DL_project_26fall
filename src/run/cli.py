"""Resumable command line entry point (P1.7).

Examples
--------
    python -m src.run.cli smoke --models all
    python -m src.run.cli messages --split pilot --limit 10 --levels b57,b69,b82 --dry-run
    python -m src.run.cli messages --split pilot --limit 10 --levels b57,b69,b82 --run-id p0.9
    python -m src.run.cli receive --messages results/messages_p0.9.jsonl --include-full --run-id p0.9
    python -m src.run.cli calibrate --limit 10
    python -m src.run.cli report --messages results/messages_p0.9.jsonl \\
        --predictions results/predictions_p0.9.jsonl --out docs/density_ladder.md
"""
from __future__ import annotations

import argparse
import json
import sys

from ..api.adapters import RunContext, smoke_test
from ..api.config import all_model_keys
from ..common import read_jsonl
from . import report as report_mod
from .data import load_docs
from .receiver import run_receivers
from .sender import parse_levels, plan_messages, run_messages

DEFAULT_LEVELS = "b57,b69,b82"
DEFAULT_METHODS = "abstractive,structured"


def _models(spec: str) -> list[str]:
    if not spec or spec == "all":
        return all_model_keys()
    return [m.strip() for m in spec.split(",") if m.strip()]


def _methods(spec: str) -> list[str]:
    return [m.strip() for m in spec.split(",") if m.strip()]


def _print(obj: dict) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2))


def cmd_smoke(args: argparse.Namespace) -> int:
    ctx = RunContext.create("smoke", max_calls=len(_models(args.models)) + 1)
    results = smoke_test(_models(args.models), ctx=ctx)
    _print({"results": results, "guard": ctx.guard.snapshot()})
    return 0 if all(r.get("ok") for r in results) else 1


def cmd_messages(args: argparse.Namespace) -> int:
    docs = load_docs(args.split, args.limit, args.doc_ids.split(",") if args.doc_ids else None)
    levels = parse_levels(args.levels)
    models = _models(args.models)
    methods = _methods(args.methods)
    out = args.out or f"results/messages_{args.run_id}.jsonl"
    ctx = RunContext.create(args.run_id, max_calls=args.max_calls)
    stats = run_messages(
        run_id=args.run_id,
        docs=docs,
        model_keys=models,
        methods=methods,
        levels=levels,
        out_path=out,
        ctx=ctx,
        concurrency=args.concurrency,
        max_retries=args.max_retries,
        dry_run=args.dry_run,
        style=getattr(args, "style", "readable"),
        cache_nonce=getattr(args, "cache_nonce", ""),
        corpus_id=getattr(args, "corpus_id", ""),
    )
    _print({**stats, "guard": ctx.guard.snapshot()})
    return 0


def cmd_receive(args: argparse.Namespace) -> int:
    messages = read_jsonl(args.messages)
    if not messages:
        raise SystemExit(f"no messages in {args.messages}")
    out = args.out or f"results/predictions_{args.run_id}.jsonl"
    ctx = RunContext.create(args.run_id, max_calls=args.max_calls)
    stats = run_receivers(
        run_id=args.run_id,
        messages=messages,
        model_keys=_models(args.models),
        out_path=out,
        ctx=ctx,
        mode=args.mode,
        include_full=args.include_full,
        levels=args.levels.split(",") if args.levels else None,
        concurrency=args.concurrency,
        dry_run=args.dry_run,
        cache_nonce=getattr(args, "cache_nonce", ""),
        corpus_id=getattr(args, "corpus_id", ""),
    )
    _print({**stats, "guard": ctx.guard.snapshot()})
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    docs = load_docs(args.split, args.limit, args.doc_ids.split(",") if args.doc_ids else None)
    levels = parse_levels(args.levels)
    models = _models(args.models)
    methods = _methods(args.methods)
    msg_path = args.messages_out or f"results/messages_{args.run_id}.jsonl"
    pred_path = args.predictions_out or f"results/predictions_{args.run_id}.jsonl"

    print(
        f"[plan] docs={len(docs)} models={len(models)} methods={len(methods)} "
        f"levels={[l.name for l in levels]} -> sender tasks={len(docs) * len(models) * len(methods) * len(levels)}",
        flush=True,
    )
    print(f"[plan] sender output: {msg_path}", flush=True)

    if args.dry_run:
        ctx = RunContext.create(f"{args.run_id}-send")
        stats = run_messages(
            run_id=args.run_id,
            docs=docs,
            model_keys=models,
            methods=methods,
            levels=levels,
            out_path=msg_path,
            ctx=ctx,
            dry_run=True,
        )
        messages = read_jsonl(msg_path)
        recv_tasks = run_receivers(
            run_id=f"{args.run_id}-recv",
            messages=messages,
            model_keys=models,
            out_path=pred_path,
            ctx=ctx,
            mode="self",
            include_full=True,
            dry_run=True,
        )
        _print({"messages": stats, "receivers": recv_tasks})
        return 0

    ctx_send = RunContext.create(f"{args.run_id}-send", max_calls=args.max_calls)
    send_stats = run_messages(
        run_id=args.run_id,
        docs=docs,
        model_keys=models,
        methods=methods,
        levels=levels,
        out_path=msg_path,
        ctx=ctx_send,
        concurrency=args.concurrency,
        max_retries=args.max_retries,
    )
    print("[sender done]", json.dumps(send_stats, ensure_ascii=False), flush=True)

    messages = read_jsonl(msg_path)
    ctx_recv = RunContext.create(f"{args.run_id}-recv", max_calls=args.max_recv_calls)
    recv_stats = run_receivers(
        run_id=f"{args.run_id}-recv",
        messages=messages,
        model_keys=models,
        out_path=pred_path,
        ctx=ctx_recv,
        mode="self",
        include_full=True,
        concurrency=args.concurrency,
    )
    print("[receiver done]", json.dumps(recv_stats, ensure_ascii=False), flush=True)

    summary = report_mod.build_and_write(
        run_id=args.run_id,
        messages_path=msg_path,
        predictions_path=pred_path,
        out_path=args.report_out,
        summary_path=args.summary_out,
    )
    _print(
        {
            "report": args.report_out,
            "summary": args.summary_out,
            "decision_status": summary["decision"]["status"],
            "selected_level": (summary["decision"]["selected"] or {}).get("level"),
        }
    )
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    summary = report_mod.build_and_write(
        run_id=args.run_id,
        messages_path=args.messages,
        predictions_path=args.predictions,
        out_path=args.out,
        summary_path=args.summary_out,
    )
    _print(
        {
            "report": args.out,
            "summary": args.summary_out,
            "decision_status": summary["decision"]["status"],
            "selected_level": (summary["decision"]["selected"] or {}).get("level"),
        }
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.run.cli")
    sub = parser.add_subparsers(dest="cmd", required=True)

    smoke = sub.add_parser("smoke", help="one JSON call per model")
    smoke.add_argument("--models", default="all")
    smoke.set_defaults(func=cmd_smoke)

    messages = sub.add_parser("messages", help="generate compressed messages (resumable)")
    messages.add_argument("--split", default="pilot")
    messages.add_argument("--limit", type=int, default=None)
    messages.add_argument("--doc-ids", default=None)
    messages.add_argument("--levels", default=DEFAULT_LEVELS)
    messages.add_argument("--models", default="all")
    messages.add_argument("--methods", default=DEFAULT_METHODS)
    messages.add_argument("--out", default=None)
    messages.add_argument("--run-id", default="p0.9")
    messages.add_argument("--concurrency", type=int, default=4)
    messages.add_argument("--max-retries", type=int, default=2)
    messages.add_argument("--max-calls", type=int, default=600)
    messages.add_argument("--data-dir", default=None, help="alternate corpus directory")
    messages.add_argument("--style", choices=["readable", "transmission"], default="readable")
    messages.add_argument("--cache-nonce", default="", help="re-issue identical requests (repetition study)")
    messages.add_argument("--corpus-id", default="",
                          help="density tag folded into record keys (paired corpora: base, b100)")
    messages.add_argument("--dry-run", action="store_true")
    messages.set_defaults(func=cmd_messages)

    receive = sub.add_parser("receive", help="ask receivers and score (resumable)")
    receive.add_argument("--messages", required=True)
    receive.add_argument("--out", default=None)
    receive.add_argument("--run-id", default="p0.9-recv")
    receive.add_argument("--models", default="all")
    receive.add_argument("--mode", choices=["self", "all"], default="self")
    receive.add_argument("--include-full", action="store_true")
    receive.add_argument("--levels", default=None)
    receive.add_argument("--concurrency", type=int, default=4)
    receive.add_argument("--data-dir", default=None, help="alternate corpus directory")
    receive.add_argument("--max-calls", type=int, default=500)
    receive.add_argument("--cache-nonce", default="", help="re-issue identical requests (repetition study)")
    receive.add_argument("--corpus-id", default="",
                         help="density tag for full-document baseline keys (paired corpora: base, b100)")
    receive.add_argument("--dry-run", action="store_true")
    receive.set_defaults(func=cmd_receive)

    calibrate = sub.add_parser("calibrate", help="P0.9 end-to-end: messages -> receivers -> report")
    calibrate.add_argument("--split", default="pilot")
    calibrate.add_argument("--limit", type=int, default=10)
    calibrate.add_argument("--doc-ids", default=None)
    calibrate.add_argument("--levels", default=DEFAULT_LEVELS)
    calibrate.add_argument("--models", default="all")
    calibrate.add_argument("--methods", default=DEFAULT_METHODS)
    calibrate.add_argument("--run-id", default="p0.9")
    calibrate.add_argument("--messages-out", default=None)
    calibrate.add_argument("--predictions-out", default=None)
    calibrate.add_argument("--report-out", default="docs/density_ladder.md")
    calibrate.add_argument("--summary-out", default="results/p0.9_summary.json")
    calibrate.add_argument("--concurrency", type=int, default=4)
    calibrate.add_argument("--max-retries", type=int, default=2)
    calibrate.add_argument("--max-calls", type=int, default=600)
    calibrate.add_argument("--max-recv-calls", type=int, default=500)
    calibrate.add_argument("--data-dir", default=None, help="alternate corpus directory")
    calibrate.add_argument("--dry-run", action="store_true")
    calibrate.set_defaults(func=cmd_calibrate)

    report = sub.add_parser("report", help="rebuild the P0.9 markdown report from JSONL")
    report.add_argument("--messages", default="results/messages_p0.9.jsonl")
    report.add_argument("--predictions", default="results/predictions_p0.9.jsonl")
    report.add_argument("--out", default="docs/density_ladder.md")
    report.add_argument("--summary-out", default="results/p0.9_summary.json")
    report.add_argument("--data-dir", default=None, help="alternate corpus directory")
    report.add_argument("--run-id", default="p0.9")
    report.set_defaults(func=cmd_report)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "data_dir", None):
        import os
        os.environ["DL_DATA_DIR"] = str(args.data_dir)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
