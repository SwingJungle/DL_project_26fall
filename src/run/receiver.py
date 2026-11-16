"""Receiver side: one batched request per (message, receiver) (P5.2 shape)."""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..api.adapters import RunContext, complete
from ..api.config import model_spec
from ..common import JsonlWriter, now_iso, read_jsonl
from ..compress.prompts import (
    PROMPT_VERSION_RECEIVER,
    PROMPT_VERSION_RECEIVER_COMPACT,
    PROMPT_VERSION_RECEIVER_FULL,
    build_receiver_prompt,
)
from ..eval.score import score_answers
from .data import load_docs, load_questions_by_doc

_RECEIVER_MAX_TOKENS = 1536
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def parse_answers(text: str) -> tuple[dict[str, str], str | None]:
    stripped = _FENCE_RE.sub("", text.strip())
    data = None
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        start, end = stripped.find("{"), stripped.rfind("}")
        if start >= 0 and end > start:
            try:
                data = json.loads(stripped[start : end + 1])
            except json.JSONDecodeError:
                return {}, "invalid_json"
        else:
            return {}, "invalid_json"

    items = None
    if isinstance(data, dict) and isinstance(data.get("answers"), list):
        items = data["answers"]
    elif isinstance(data, list):
        items = data
    if items is None:
        return {}, "no_answers_key"

    answers: dict[str, str] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        qid = item.get("qid") or item.get("id")
        if qid is None:
            continue
        value = item.get("a", item.get("answer"))
        answers[str(qid)] = "" if value is None else str(value)
    if not answers:
        return {}, "empty_answers"
    return answers, None


def ask_receiver(
    model_key: str,
    message: str,
    questions: list[dict],
    *,
    source: str,
    ctx: RunContext,
    tag: str,
    message_format: str | None = None,
    cache_nonce: str = "",
):
    prompt = build_receiver_prompt(message, questions, source=source, message_format=message_format)
    if source != "compressed":
        prompt_version = PROMPT_VERSION_RECEIVER_FULL
    elif message_format == "compact":
        prompt_version = PROMPT_VERSION_RECEIVER_COMPACT
    else:
        prompt_version = PROMPT_VERSION_RECEIVER
    completion = complete(
        model_key,
        [{"role": "user", "content": prompt}],
        json_schema={"type": "object", "required": ["answers"]},
        max_tokens=_RECEIVER_MAX_TOKENS,
        tag=tag,
        prompt_version=prompt_version,
        ctx=ctx,
        timeout_s=60.0,
        cache_nonce=cache_nonce,
    )
    answers, error = parse_answers(completion.text)
    return completion, answers, error, prompt


def receiver_record(
    *,
    key: str,
    doc: dict,
    questions: list[dict],
    sender_model: str | None,
    receiver_model: str,
    method: str,
    level: str,
    pair_type: str,
    message_format: str | None,
    message: str,
    message_bytes: int,
    completion,
    answers: dict,
    parse_error: str | None,
) -> dict:
    scored = score_answers(questions, answers)
    return {
        "key": key,
        "doc_id": doc["doc_id"],
        "sender_model": sender_model,
        "receiver_model": receiver_model,
        "method": method,
        "level": level,
        "pair_type": pair_type,
        "message_format": message_format,
        "message_bytes": message_bytes,
        "n_questions": scored["n"],
        "n_answered": sum(1 for v in answers.values() if v),
        "accuracy": round(scored["accuracy"], 4),
        "lookup_accuracy": round(scored["lookup_accuracy"], 4),
        "multihop_accuracy": round(scored["multihop_accuracy"], 4),
        "mean_f1": round(scored["mean_f1"], 4),
        "parse_error": parse_error,
        "answers": answers,
        "per_question": scored["rows"],
        "usage": completion.usage,
        "latency_ms": completion.latency_ms,
        "cached": completion.cached,
        "receiver_text": completion.text,
    }


def pair_type(sender_model: str | None, receiver_model: str) -> str:
    if sender_model is None:
        return "baseline_full"
    if sender_model == receiver_model:
        return "self"
    sender = model_spec(sender_model)
    receiver = model_spec(receiver_model)
    if sender["family"] == receiver["family"]:
        return "same"
    if sender["tier"] == receiver["tier"]:
        return "cross_matched"
    return "cross_unmatched"


def plan_receiver_tasks(
    messages: list[dict],
    docs_by_id: dict[str, dict],
    model_keys: list[str],
    *,
    mode: str = "self",
    include_full: bool = False,
    corpus_id: str = "",
) -> list[dict]:
    tasks = []
    for msg in messages:
        if msg.get("status") == "error" or not msg.get("message"):
            continue
        receivers = [msg["model_key"]] if mode == "self" else list(model_keys)
        for receiver in receivers:
            tasks.append(
                {
                    "kind": "compressed",
                    "key": f"{msg['key']}|recv:{receiver}",
                    "doc": docs_by_id[msg["doc_id"]],
                    "sender_model": msg["model_key"],
                    "receiver_model": receiver,
                    "method": msg["method"],
                    "level": msg["level"],
                    "message_format": "compact" if msg["method"] == "compact" else None,
                    "message": msg["message"],
                    "message_bytes": int(msg["final_bytes"]),
                    "pair_type": pair_type(msg["model_key"], receiver),
                }
            )
    if include_full:
        doc_ids = sorted({m["doc_id"] for m in messages if m.get("message")})
        for doc_id in doc_ids:
            doc = docs_by_id[doc_id]
            for receiver in model_keys:
                tasks.append(
                    {
                        "kind": "full",
                        "key": (
                            f"{corpus_id}|{doc_id}|full|recv:{receiver}"
                            if corpus_id
                            else f"{doc_id}|full|recv:{receiver}"
                        ),
                        "doc": doc,
                        "sender_model": None,
                        "receiver_model": receiver,
                        "method": "full",
                        "level": "full",
                        "message_format": None,
                        "message": doc["text"],
                        "message_bytes": len(doc["text"].encode("utf-8")),
                        "pair_type": "baseline_full",
                    }
                )
    return tasks


def run_receivers(
    *,
    run_id: str,
    messages: list[dict],
    model_keys: list[str],
    out_path: str,
    ctx: RunContext,
    mode: str = "self",
    include_full: bool = False,
    levels: list[str] | None = None,
    concurrency: int = 3,
    dry_run: bool = False,
    verbose: bool = True,
    cache_nonce: str = "",
    corpus_id: str = "",
) -> dict:
    docs_by_id = {doc["doc_id"]: doc for doc in load_docs()}
    if levels:
        wanted = set(levels)
        messages = [m for m in messages if m.get("level") in wanted]
    tasks = plan_receiver_tasks(
        messages, docs_by_id, model_keys, mode=mode, include_full=include_full,
        corpus_id=corpus_id,
    )
    existing = {rec.get("key") for rec in read_jsonl(out_path)}
    tasks = [t for t in tasks if t["key"] not in existing]

    if dry_run:
        return {
            "dry_run": True,
            "planned_total": len(existing) + len(tasks),
            "already_done": len(existing),
            "would_run": len(tasks),
            "out_path": str(out_path),
        }

    questions_by_doc = load_questions_by_doc()
    writer = JsonlWriter(out_path)
    stats = {"ok": 0, "error": 0, "cache_hits": 0, "parse_errors": 0, "calls": 0}
    errors: list[str] = []

    def work(task: dict) -> tuple[dict, dict | None, str | None]:
        doc = task["doc"]
        questions = questions_by_doc[doc["doc_id"]]
        source = "full" if task["kind"] == "full" else "compressed"
        try:
            completion, answers, parse_error, _prompt = ask_receiver(
                task["receiver_model"],
                task["message"],
                questions,
                source=source,
                ctx=ctx,
                tag=f"receiver:{task['key']}",
                message_format=task.get("message_format"),
                cache_nonce=cache_nonce,
            )
        except Exception as exc:  # noqa: BLE001
            return task, None, f"{type(exc).__name__}: {exc}"
        rec = receiver_record(
            key=task["key"],
            doc=doc,
            questions=questions,
            sender_model=task["sender_model"],
            receiver_model=task["receiver_model"],
            method=task["method"],
            level=task["level"],
            pair_type=task["pair_type"],
            message_format=task.get("message_format"),
            message=task["message"],
            message_bytes=task["message_bytes"],
            completion=completion,
            answers=answers,
            parse_error=parse_error,
        )
        rec.update({"run_id": run_id, "created_at": now_iso(),
                    "corpus_id": corpus_id or None})
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
                stats["cache_hits"] += int(bool(rec["cached"]))
                stats["parse_errors"] += int(rec["parse_error"] is not None)
            if verbose and (i % 20 == 0 or i == len(tasks)):
                print(f"  receivers {i}/{len(tasks)} ok={stats['ok']} err={stats['error']}", flush=True)

    stats["calls"] = ctx.guard.calls
    stats["elapsed_guard"] = ctx.guard.snapshot()
    stats["errors"] = errors[:20]
    stats["out_path"] = str(out_path)
    stats["planned"] = len(tasks)
    return stats
