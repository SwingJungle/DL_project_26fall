"""Sender-side orchestration and resumable message generation (P1.7 / P5.1)."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from ..api.adapters import RunContext
from ..common import JsonlWriter, now_iso, read_jsonl
from ..compress.budget import BudgetWindow, window_from_b, window_from_ratio
from ..compress.pipeline import compress_document
from ..eval.retention import retention_summary
from .data import doc_bytes, load_docs, load_questions_by_doc


@dataclass(frozen=True)
class Level:
    name: str
    kind: str  # "ratio" | "b"
    value: float

    def window(self, doc: dict) -> BudgetWindow:
        if self.kind == "ratio":
            return window_from_ratio(doc_bytes(doc), self.value, 0.02)
        return window_from_b(doc_bytes(doc), int(doc["n_facts"]), self.value, 0.02)

    def describe(self) -> str:
        if self.kind == "ratio":
            return f"{int(self.value * 100)}% of source bytes"
        return f"{self.value:g} B/fact (b)"


def parse_levels(spec: str) -> list[Level]:
    levels = []
    for raw in spec.split(","):
        token = raw.strip().lower()
        if not token:
            continue
        if token.startswith("b"):
            levels.append(Level(token, "b", float(token[1:])))
        elif token.startswith("r"):
            levels.append(Level(token, "ratio", float(token[1:]) / 100.0))
        else:
            raise ValueError(f"level {raw!r}: use 'b69' (bytes/fact) or 'r50' (ratio)")
    if not levels:
        raise ValueError("no levels given")
    return levels


def message_key(doc_id: str, model_key: str, method: str, level: str,
                corpus_id: str = "") -> str:
    """Record key for one sender message.

    ``corpus_id`` disambiguates the same document rendered at another density.
    The paired-density design re-uses ``doc_id``/``method``/``level`` across
    densities, so without a corpus tag the two runs would collide: a resumed run
    (or a merged file) would silently skip the second density. It is optional so
    every existing run keeps its keys.
    """
    base = f"{doc_id}|{model_key}|{method}|{level}"
    return f"{corpus_id}|{base}" if corpus_id else base


def plan_messages(
    docs: list[dict],
    model_keys: list[str],
    methods: list[str],
    levels: list[Level],
    corpus_id: str = "",
) -> list[dict]:
    tasks = []
    for doc in docs:
        for model_key in model_keys:
            for method in methods:
                for level in levels:
                    tasks.append(
                        {
                            "doc": doc,
                            "model_key": model_key,
                            "method": method,
                            "level": level,
                            "window": level.window(doc),
                            "key": message_key(doc["doc_id"], model_key, method,
                                               level.name, corpus_id),
                        }
                    )
    return tasks


def run_messages(
    *,
    run_id: str,
    docs: list[dict],
    model_keys: list[str],
    methods: list[str],
    levels: list[Level],
    out_path: str,
    ctx: RunContext,
    concurrency: int = 3,
    max_retries: int = 2,
    dry_run: bool = False,
    verbose: bool = True,
    style: str = "readable",
    cache_nonce: str = "",
    corpus_id: str = "",
) -> dict:
    all_tasks = plan_messages(docs, model_keys, methods, levels, corpus_id)
    existing = {rec.get("key") for rec in read_jsonl(out_path)}
    tasks = [t for t in all_tasks if t["key"] not in existing]

    if dry_run:
        return {
            "dry_run": True,
            "planned_total": len(all_tasks),
            "already_done": len(existing),
            "would_run": len(tasks),
            "out_path": str(out_path),
        }

    questions = load_questions_by_doc()
    writer = JsonlWriter(out_path)
    stats = {"ok": 0, "error": 0, "cache_hits": 0, "noncompliant": 0, "calls": 0}
    errors: list[str] = []

    def work(task: dict) -> tuple[dict, dict | None, str | None]:
        doc = task["doc"]
        try:
            result = compress_document(
                doc,
                task["model_key"],
                task["method"],
                task["window"],
                ctx=ctx,
                level=task["level"].name,
                max_retries=max_retries,
                style=style,
                cache_nonce=cache_nonce,
            )
        except Exception as exc:  # noqa: BLE001 - reported, not fatal
            return task, None, f"{type(exc).__name__}: {exc}"
        rec = result.to_dict()
        rec.update(
            {
                "key": task["key"],
                "corpus_id": corpus_id or None,
                "run_id": run_id,
                "created_at": now_iso(),
                "doc_bytes": doc_bytes(doc),
                "n_facts": int(doc["n_facts"]),
                "level_kind": task["level"].kind,
                "level_value": task["level"].value,
                "style": style,
                "first_attempt_compliant": bool(
                    result.attempts and result.attempts[0].compliant
                ),
                "retention": retention_summary(
                    doc, result.message, questions.get(doc["doc_id"], []), task["method"]
                ),
            }
        )
        return task, rec, None

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = {pool.submit(work, task): task for task in tasks}
        for i, fut in enumerate(as_completed(futures), start=1):
            task, rec, error = fut.result()
            if error is not None:
                stats["error"] += 1
                errors.append(f"{task['key']}: {error}")
            else:
                writer.append(rec)
                stats["ok"] += 1
                stats["cache_hits"] += sum(1 for a in rec["attempts"] if a["cached"])
                if not rec["compliant"]:
                    stats["noncompliant"] += 1
            if verbose and (i % 20 == 0 or i == len(tasks)):
                print(f"  messages {i}/{len(tasks)} ok={stats['ok']} err={stats['error']}", flush=True)

    stats["calls"] = ctx.guard.calls
    stats["elapsed_guard"] = ctx.guard.snapshot()
    stats["errors"] = errors[:20]
    stats["out_path"] = str(out_path)
    stats["planned"] = len(tasks)
    return stats
