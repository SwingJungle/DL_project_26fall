"""Unified provider adapter: one ``complete()`` for all four models (P1.2).

Verified gateway facts baked in (see DL_execution_plan.md §3.1):
* thinking is ON by default -> every request sends ``enable_thinking: false``;
* DeepSeek has no ``json_schema`` -> ``response_format`` is always
  ``{"type": "json_object"}`` and the schema is described in the prompt;
* the returned model id is checked against the pinned id and a mismatch is a
  hard error (version pinning discipline).
"""
from __future__ import annotations

import http.client
import json
import re
import socket
import threading
import time
from dataclasses import dataclass
from typing import Any

from ..common import canonical_json, now_iso, sha256_text
from .cache import CompletionCache
from .calls import CallLogger
from .config import budget_config, credentials, model_spec, models_config
from .cost import compute_cost_usd
from .retry import BudgetGuard, RateLimiter, retry_call

_API_KEY_RE = re.compile(r"sk-[A-Za-z0-9._\-]{8,}")


def redact(text: str) -> str:
    return _API_KEY_RE.sub("sk-***REDACTED***", text)


@dataclass
class Completion:
    text: str
    model_id: str
    usage: dict
    request_id: str | None
    ts: str
    latency_ms: int
    cost_usd: float
    cached: bool
    finish_reason: str | None = None
    attempts: int = 1
    prompt_version: str = ""


class ProviderError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        http_status: int | None = None,
        body: str | None = None,
        retryable: bool = False,
        retry_after: float | None = None,
    ):
        super().__init__(redact(message))
        self.http_status = http_status
        self.body = redact(body or "")
        self.retryable = retryable
        self.retry_after = retry_after


class ModelMismatch(RuntimeError):
    """The gateway returned a different model id than the pinned one."""


@dataclass
class RunContext:
    run_id: str
    cache: CompletionCache
    logger: CallLogger
    guard: BudgetGuard
    limiter: RateLimiter

    @classmethod
    def create(
        cls,
        run_id: str,
        *,
        cache: CompletionCache | None = None,
        max_calls: int | None = None,
        max_prompt_tokens: int | None = None,
        max_completion_tokens: int | None = None,
        max_cost_usd: float | None = None,
        min_interval_s: float | None = None,
    ) -> "RunContext":
        budget = budget_config()
        cap = budget.get("cap", {})
        rl = budget.get("rate_limit", {})
        return cls(
            run_id=run_id,
            cache=cache or CompletionCache(),
            logger=CallLogger(run_id=run_id),
            guard=BudgetGuard(
                max_calls=max_calls if max_calls is not None else cap.get("max_calls_per_run"),
                max_prompt_tokens=(
                    max_prompt_tokens
                    if max_prompt_tokens is not None
                    else cap.get("max_prompt_tokens_per_run")
                ),
                max_completion_tokens=(
                    max_completion_tokens
                    if max_completion_tokens is not None
                    else cap.get("max_completion_tokens_per_run")
                ),
                max_cost_usd=(
                    max_cost_usd if max_cost_usd is not None else cap.get("max_cost_usd_per_run")
                ),
            ),
            limiter=RateLimiter(
                min_interval_s if min_interval_s is not None else rl.get("min_interval_s", 0.0)
            ),
        )


_DEFAULT_CTX: RunContext | None = None


def default_context() -> RunContext:
    global _DEFAULT_CTX
    if _DEFAULT_CTX is None:
        _DEFAULT_CTX = RunContext.create("adhoc")
    return _DEFAULT_CTX


def request_payload(
    model_key: str,
    messages: list[dict],
    *,
    json_schema: dict | None,
    max_tokens: int,
    temperature: float,
) -> dict:
    spec = model_spec(model_key)
    payload: dict[str, Any] = {
        "model": spec["api_model"],
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "enable_thinking": False,
        "stream": False,
    }
    if json_schema is not None:
        payload["response_format"] = {"type": "json_object"}
    return payload


def cache_key_for(
    model_key: str,
    messages: list[dict],
    *,
    json_schema: dict | None,
    temperature: float,
    max_tokens: int,
    prompt_version: str,
) -> str:
    spec = model_spec(model_key)
    payload = {
        "cache_version": 1,
        "provider": spec["provider"],
        "api_model": spec["api_model"],
        "messages": messages,
        "json_schema": json_schema,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "enable_thinking": False,
        "prompt_version": prompt_version,
    }
    return sha256_text(canonical_json(payload))


def _http_post(
    url: str,
    api_key: str,
    payload: dict,
    *,
    timeout_s: float,
) -> tuple[dict, dict, int]:
    """POST with a hard wall-clock watchdog.

    ``urlopen``'s read timeout has been observed to sit in SSL poll for tens of
    minutes on a queued gateway connection. A watchdog timer closes the socket
    shortly after ``timeout_s`` so a stuck request fails and is retried instead
    of blocking a worker forever.
    """
    from urllib.parse import urlsplit

    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise ProviderError(f"unsupported URL scheme: {parts.scheme!r}")
    conn_cls = http.client.HTTPSConnection if parts.scheme == "https" else http.client.HTTPConnection
    conn = conn_cls(parts.hostname, parts.port, timeout=timeout_s)
    path = parts.path or "/"
    if parts.query:
        path += "?" + parts.query
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "dl-comms/1.0",
    }

    sock_holder: dict[str, socket.socket | None] = {"sock": None}
    aborted = threading.Event()

    def _abort() -> None:
        aborted.set()
        sock = sock_holder.get("sock")
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                sock.close()
            except OSError:
                pass

    watchdog = threading.Timer(max(timeout_s + 10.0, timeout_s * 1.2), _abort)
    watchdog.daemon = True
    watchdog.start()
    started = time.monotonic()
    try:
        conn.connect()
        sock_holder["sock"] = conn.sock
        if aborted.is_set():
            raise ProviderError(f"hard timeout after {timeout_s:.0f}s", retryable=True)
        conn.request("POST", path, body=body, headers=headers)
        response = conn.getresponse()
        status = response.status
        response_headers = {k.lower(): v for k, v in response.getheaders()}
        raw = response.read().decode("utf-8", "replace")
    except ProviderError:
        raise
    except (http.client.HTTPException, OSError, socket.timeout) as exc:
        if aborted.is_set():
            raise ProviderError(
                f"hard timeout after {timeout_s:.0f}s ({type(exc).__name__})",
                retryable=True,
            ) from None
        raise ProviderError(
            f"network error: {type(exc).__name__}: {exc}", retryable=True
        ) from None
    finally:
        watchdog.cancel()
        try:
            conn.close()
        except OSError:
            pass
    latency_ms = int((time.monotonic() - started) * 1000)

    if status >= 400:
        retry_after = response_headers.get("retry-after")
        parsed_retry = None
        if retry_after:
            try:
                parsed_retry = float(retry_after)
            except ValueError:
                parsed_retry = None
        raise ProviderError(
            f"HTTP {status}: {raw[:500]}",
            http_status=status,
            body=raw[:4000],
            retryable=status == 429 or 500 <= status < 600,
            retry_after=parsed_retry,
        )

    try:
        body_json = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ProviderError(
            f"invalid JSON response ({exc}): {raw[:300]}", retryable=True
        ) from None
    return body_json, response_headers, latency_ms


def _normalise_usage(usage: dict | None) -> dict:
    usage = usage or {}
    details = usage.get("completion_tokens_details") or {}
    prompt_details = usage.get("prompt_tokens_details") or {}
    return {
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "total_tokens": int(usage.get("total_tokens") or 0),
        "reasoning_tokens": int(details.get("reasoning_tokens") or 0),
        "cached_tokens": int(prompt_details.get("cached_tokens") or 0),
    }


def complete(
    model_key: str,
    messages: list[dict],
    *,
    json_schema: dict | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    tag: str = "",
    prompt_version: str = "",
    ctx: RunContext | None = None,
    timeout_s: float | None = None,
    cache_nonce: str = "",
) -> Completion:
    """Single entry point for all four models. Raises on exhausted retries.

    ``cache_nonce`` is folded into the cache key only (not into the prompt). It
    exists so a repetition study can re-issue the identical request without
    being served the cached completion, while the logged prompt_version stays
    interpretable.
    """
    cfg = models_config()
    defaults = cfg["defaults"]
    max_tokens = int(defaults["max_tokens"] if max_tokens is None else max_tokens)
    temperature = float(defaults["temperature"] if temperature is None else temperature)
    timeout_s = float(cfg["gateway"]["timeout_s"] if timeout_s is None else timeout_s)
    ctx = ctx or default_context()

    key = cache_key_for(
        model_key,
        messages,
        json_schema=json_schema,
        temperature=temperature,
        max_tokens=max_tokens,
        prompt_version=prompt_version + (f"+{cache_nonce}" if cache_nonce else ""),
    )

    cached = ctx.cache.get(key)
    if cached is not None:
        usage = json.loads(cached["usage_json"])
        completion = Completion(
            text=cached["text"],
            model_id=cached["model_returned"] or model_spec(model_key)["api_model"],
            usage=usage,
            request_id=cached["request_id"],
            ts=now_iso(),
            latency_ms=0,
            cost_usd=float(cached["cost_usd"] or 0.0),
            cached=True,
            finish_reason=cached["finish_reason"],
            attempts=0,
            prompt_version=prompt_version,
        )
        ctx.logger.log(
            {
                "run_id": ctx.run_id,
                "tag": tag,
                "model_key": model_key,
                "api_model": model_spec(model_key)["api_model"],
                "model_returned": completion.model_id,
                "prompt_version": prompt_version,
                "cache_key": key,
                "cached": True,
                "status": "ok",
                "attempts": 0,
                "http_status": None,
                "latency_ms": 0,
                "usage": usage,
                "cost_usd": completion.cost_usd,
                "request": {"messages": messages, "json_schema": json_schema,
                            "temperature": temperature, "max_tokens": max_tokens},
                "response_text": cached["text"],
                "cached_ts": cached["ts"],
            }
        )
        return completion

    base_url, api_key = credentials()
    url = base_url + cfg["gateway"]["chat_path"]
    payload = request_payload(
        model_key,
        messages,
        json_schema=json_schema,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    spec = model_spec(model_key)
    rl = budget_config().get("rate_limit", {})
    errors: list[str] = []

    def attempt() -> dict:
        ctx.guard.reserve_call()
        ctx.limiter.wait(spec["provider"])
        body, headers, latency_ms = _http_post(url, api_key, payload, timeout_s=timeout_s)
        choices = body.get("choices") or []
        if not choices:
            raise ProviderError(
                f"no choices in response: keys={sorted(body)}", retryable=True
            )
        choice = choices[0]
        message = choice.get("message") or {}
        text = message.get("content")
        if text is None:
            text = ""
        if not text.strip():
            raise ProviderError(
                f"empty content (finish_reason={choice.get('finish_reason')})",
                retryable=True,
            )
        returned = body.get("model")
        if returned != spec["api_model"]:
            raise ModelMismatch(
                f"pinned model mismatch: requested {spec['api_model']!r}, got {returned!r}"
            )
        usage = _normalise_usage(body.get("usage"))
        return {
            "text": text,
            "model_returned": returned,
            "request_id": body.get("id") or headers.get("x-request-id"),
            "finish_reason": choice.get("finish_reason"),
            "usage": usage,
            "latency_ms": latency_ms,
            "ts": now_iso(),
        }

    def on_retry(attempt_no: int, exc: BaseException, delay: float) -> None:
        errors.append(f"attempt {attempt_no}: {type(exc).__name__}: {exc}")
        print(
            f"[retry] {model_key} attempt {attempt_no} failed ({type(exc).__name__}: "
            f"{str(exc)[:120]}); sleeping {delay:.1f}s",
            flush=True,
        )

    try:
        result = retry_call(
            attempt,
            max_attempts=int(rl.get("max_attempts", 5)),
            base_delay_s=float(rl.get("backoff_base_s", 2.0)),
            max_delay_s=float(rl.get("backoff_max_s", 60.0)),
            retry_if=lambda exc: isinstance(exc, ProviderError) and exc.retryable,
            on_retry=on_retry,
        )
    except Exception as exc:  # noqa: BLE001 - logged then re-raised
        ctx.logger.log(
            {
                "run_id": ctx.run_id,
                "tag": tag,
                "model_key": model_key,
                "api_model": spec["api_model"],
                "prompt_version": prompt_version,
                "cache_key": key,
                "cached": False,
                "status": "error",
                "attempts": len(errors) + 1,
                "http_status": getattr(exc, "http_status", None),
                "error": f"{type(exc).__name__}: {exc}",
                "error_chain": errors,
                "request": {"messages": messages, "json_schema": json_schema,
                            "temperature": temperature, "max_tokens": max_tokens},
            }
        )
        raise

    cost_usd = compute_cost_usd(model_key, result["usage"])
    ctx.guard.record(result["usage"], cost_usd)
    ctx.cache.put(
        key,
        {
            "model_key": model_key,
            "provider": spec["provider"],
            "api_model": spec["api_model"],
            "model_returned": result["model_returned"],
            "request_json": canonical_json(payload),
            "response_text": result["text"],
            "usage_json": json.dumps(result["usage"], separators=(",", ":")),
            "request_id": result["request_id"],
            "finish_reason": result["finish_reason"],
            "ts": result["ts"],
            "latency_ms": result["latency_ms"],
            "cost_usd": cost_usd,
            "prompt_version": prompt_version,
            "tag": tag,
        },
    )
    completion = Completion(
        text=result["text"],
        model_id=result["model_returned"],
        usage=result["usage"],
        request_id=result["request_id"],
        ts=result["ts"],
        latency_ms=result["latency_ms"],
        cost_usd=cost_usd,
        cached=False,
        finish_reason=result["finish_reason"],
        attempts=len(errors) + 1,
        prompt_version=prompt_version,
    )
    ctx.logger.log(
        {
            "run_id": ctx.run_id,
            "tag": tag,
            "model_key": model_key,
            "api_model": spec["api_model"],
            "model_returned": result["model_returned"],
            "prompt_version": prompt_version,
            "cache_key": key,
            "cached": False,
            "status": "ok",
            "attempts": completion.attempts,
            "http_status": 200,
            "latency_ms": result["latency_ms"],
            "usage": result["usage"],
            "cost_usd": cost_usd,
            "request": {"messages": messages, "json_schema": json_schema,
                        "temperature": temperature, "max_tokens": max_tokens},
            "response_text": result["text"],
            "retry_errors": errors,
        }
    )
    return completion


def smoke_test(model_keys: list[str] | None = None, *, ctx: RunContext | None = None) -> list[dict]:
    """Four one-shot calls: thinking off, JSON object, pinned id, usage present."""
    keys = model_keys or list(models_config()["models"])
    ctx = ctx or RunContext.create("smoke")
    out = []
    for model_key in keys:
        started = time.monotonic()
        try:
            completion = complete(
                model_key,
                [
                    {"role": "system", "content": "Reply with JSON only."},
                    {"role": "user", "content": 'Return exactly {"ok": true, "tag": "smoke"}.'},
                ],
                json_schema={"type": "object", "required": ["ok"]},
                max_tokens=32,
                tag=f"smoke:{model_key}",
                prompt_version="smoke-v1",
                ctx=ctx,
            )
            parsed = json.loads(completion.text)
            out.append(
                {
                    "model_key": model_key,
                    "ok": True,
                    "json": parsed,
                    "model_returned": completion.model_id,
                    "usage": completion.usage,
                    "cached": completion.cached,
                    "wall_ms": int((time.monotonic() - started) * 1000),
                }
            )
        except Exception as exc:  # noqa: BLE001
            out.append(
                {
                    "model_key": model_key,
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "wall_ms": int((time.monotonic() - started) * 1000),
                }
            )
    return out
