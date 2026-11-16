"""Budget-closed compression pipeline (P3.4).

One call per attempt; at most ``max_retries`` corrective regenerations after
the first attempt. Never truncates. A message that is still out of band is
returned with ``compliant=False`` and every attempt preserved -- failures are
data, and the runner decides about document-level exclusion.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..api.adapters import RunContext, complete
from . import abstractive, compact, structured
from .budget import BudgetWindow, count_bytes
from .prompts import (
    PROMPT_VERSION_ABS,
    PROMPT_VERSION_ABS_TRANS,
    PROMPT_VERSION_COMPACT,
    PROMPT_VERSION_STRUCT,
    PROMPT_VERSION_STRUCT_TRANS,
    build_sender_prompt,
    build_transmission_prompt,
)

_SENDER_MAX_TOKENS = 1536


@dataclass
class Attempt:
    index: int
    prompt_chars: int
    text: str
    bytes: int
    direction: str
    compliant: bool
    error: str | None
    usage: dict = field(default_factory=dict)
    latency_ms: int = 0
    cached: bool = False
    finish_reason: str | None = None

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "prompt_chars": self.prompt_chars,
            "text": self.text,
            "bytes": self.bytes,
            "direction": self.direction,
            "compliant": self.compliant,
            "error": self.error,
            "usage": self.usage,
            "latency_ms": self.latency_ms,
            "cached": self.cached,
            "finish_reason": self.finish_reason,
        }


@dataclass
class CompressionResult:
    doc_id: str
    model_key: str
    method: str
    level: str
    window: BudgetWindow
    message: str
    final_bytes: int
    ratio_source: float
    compliant: bool
    direction: str
    attempts: list[Attempt]
    prompt_version: str
    n_items: int | None = None
    json_valid: bool = False
    elapsed_s: float = 0.0

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "model_key": self.model_key,
            "method": self.method,
            "level": self.level,
            "window": self.window.to_dict(),
            "message": self.message,
            "final_bytes": self.final_bytes,
            "ratio_source": round(self.ratio_source, 4),
            "compliant": self.compliant,
            "direction": self.direction,
            "n_attempts": len(self.attempts),
            "n_items": self.n_items,
            "json_valid": self.json_valid,
            "prompt_version": self.prompt_version,
            "elapsed_s": round(self.elapsed_s, 2),
            "attempts": [a.to_dict() for a in self.attempts],
        }


def compress_document(
    doc: dict,
    model_key: str,
    method: str,
    window: BudgetWindow,
    *,
    ctx: RunContext,
    level: str,
    max_retries: int = 2,
    tag_prefix: str = "sender",
    style: str = "readable",
    cache_nonce: str = "",
) -> CompressionResult:
    started = time.monotonic()
    document = doc["text"]
    if style == "transmission":
        if method not in ("abstractive", "structured"):
            raise ValueError(f"unknown transmission method {method!r}")
        prompt_version = (
            PROMPT_VERSION_ABS_TRANS
            if method == "abstractive"
            else PROMPT_VERSION_STRUCT_TRANS
        )
    elif style == "readable":
        if method == "abstractive":
            prompt_version = PROMPT_VERSION_ABS
        elif method == "structured":
            prompt_version = PROMPT_VERSION_STRUCT
        elif method == "compact":
            prompt_version = PROMPT_VERSION_COMPACT
        else:
            raise ValueError(f"unknown method {method!r}")
    else:
        raise ValueError(f"unknown style {style!r}")
    attempts: list[Attempt] = []
    previous_text: str | None = None
    previous_bytes: int | None = None
    problem: str | None = None
    last_error: str | None = None
    previous_words: int | None = None
    message = ""
    final_bytes = 0
    direction = "under"
    n_items: int | None = None
    json_valid = False

    for index in range(max_retries + 1):
        builder = build_transmission_prompt if style == "transmission" else build_sender_prompt
        prompt = builder(
            method,
            document,
            window,
            previous_text=previous_text,
            previous_bytes=previous_bytes,
            problem=problem,
            error_detail=last_error,
            previous_words=previous_words,
        )
        completion = complete(
            model_key,
            [{"role": "user", "content": prompt}],
            max_tokens=_SENDER_MAX_TOKENS,
            tag=f"{tag_prefix}:{doc['doc_id']}:{model_key}:{method}:{level}:a{index}",
            prompt_version=prompt_version,
            ctx=ctx,
            timeout_s=45.0,
            cache_nonce=cache_nonce,
        )
        raw = completion.text
        error: str | None = None
        parsed = None
        try:
            if method == "structured":
                message, parsed = structured.canonicalize(raw)
                n_items = len(parsed["f"])
                json_valid = True
            elif method == "compact":
                message, parsed = compact.canonicalize(raw)
                n_items = len(parsed["f"])
                json_valid = True
            elif method == "abstractive":
                message, _flags = abstractive.finalize(raw)
            else:
                raise ValueError(f"unknown method {method!r}")
        except (
            structured.StructuredFormatError,
            compact.CompactFormatError,
            abstractive.AbstractiveFormatError,
        ) as exc:
            error = str(exc)
            attempts.append(
                Attempt(
                    index=index,
                    prompt_chars=len(prompt),
                    text=raw,
                    bytes=count_bytes(raw),
                    direction="invalid",
                    compliant=False,
                    error=error,
                    usage=completion.usage,
                    latency_ms=completion.latency_ms,
                    cached=completion.cached,
                    finish_reason=completion.finish_reason,
                )
            )
            previous_text = raw
            previous_bytes = count_bytes(raw)
            previous_words = max(1, len(raw.split()))
            problem = "invalid_json" if method == "structured" else "format"
            last_error = error
            continue

        final_bytes = count_bytes(message)
        direction = window.direction(final_bytes)
        compliant = direction == "ok"
        attempts.append(
            Attempt(
                index=index,
                prompt_chars=len(prompt),
                text=message,
                bytes=final_bytes,
                direction=direction,
                compliant=compliant,
                error=None,
                usage=completion.usage,
                latency_ms=completion.latency_ms,
                cached=completion.cached,
                finish_reason=completion.finish_reason,
            )
        )
        if compliant:
            break
        previous_text = message
        previous_bytes = final_bytes
        if method == "structured":
            previous_words = sum(len(item["x"].split()) for item in parsed["f"])
        elif method == "compact":
            previous_words = sum(len(item["x"].split()) for item in parsed["f"])
        else:
            previous_words = max(1, len(message.split()))
        problem = direction

    return CompressionResult(
        doc_id=doc["doc_id"],
        model_key=model_key,
        method=method,
        level=level,
        window=window,
        message=message,
        final_bytes=final_bytes,
        ratio_source=window.ratio(final_bytes),
        compliant=window.compliant(final_bytes),
        direction=window.direction(final_bytes),
        attempts=attempts,
        prompt_version=prompt_version,
        n_items=n_items,
        json_valid=json_valid,
        elapsed_s=time.monotonic() - started,
    )
