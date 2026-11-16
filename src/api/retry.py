"""Retry, rate limiting and the hard budget stop (P1.4).

The gateway is a subscription token-plan endpoint (D-2): there is no reliable
USD price, so the hard stop is expressed in requests and tokens, and an
optional USD cap is honoured when a price table is configured.
"""
from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass, field
from typing import Callable


class BudgetExceeded(RuntimeError):
    """Raised before a request that would cross the run-level hard cap."""


@dataclass
class BudgetGuard:
    max_calls: int | None = None
    max_prompt_tokens: int | None = None
    max_completion_tokens: int | None = None
    max_cost_usd: float | None = None
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def reserve_call(self) -> None:
        with self._lock:
            if self.max_calls is not None and self.calls + 1 > self.max_calls:
                raise BudgetExceeded(
                    f"run call cap reached: {self.calls}/{self.max_calls}. "
                    "Resume with a higher --max-calls if this is intentional."
                )
            self.calls += 1

    def record(self, usage: dict | None, cost_usd: float = 0.0) -> None:
        usage = usage or {}
        with self._lock:
            self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
            self.completion_tokens += int(usage.get("completion_tokens") or 0)
            self.cost_usd += float(cost_usd or 0.0)
            if (
                self.max_prompt_tokens is not None
                and self.prompt_tokens > self.max_prompt_tokens
            ):
                raise BudgetExceeded(
                    f"prompt-token cap exceeded: {self.prompt_tokens}>{self.max_prompt_tokens}"
                )
            if (
                self.max_completion_tokens is not None
                and self.completion_tokens > self.max_completion_tokens
            ):
                raise BudgetExceeded(
                    f"completion-token cap exceeded: "
                    f"{self.completion_tokens}>{self.max_completion_tokens}"
                )
            if self.max_cost_usd is not None and self.cost_usd > self.max_cost_usd:
                raise BudgetExceeded(
                    f"cost cap exceeded: {self.cost_usd:.4f}>{self.max_cost_usd:.4f}"
                )

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "calls": self.calls,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "cost_usd": round(self.cost_usd, 6),
            }


class RateLimiter:
    """Minimum interval between request starts, per provider. Thread-safe."""

    def __init__(self, min_interval_s: float = 0.0):
        self.min_interval_s = float(min_interval_s)
        self._last: dict[str, float] = {}
        self._lock = threading.Lock()

    def wait(self, key: str) -> None:
        if self.min_interval_s <= 0:
            return
        with self._lock:
            now = time.monotonic()
            last = self._last.get(key)
            if last is not None:
                delay = self.min_interval_s - (now - last)
                if delay > 0:
                    time.sleep(delay)
                    now = time.monotonic()
            self._last[key] = now


def retry_call(
    fn: Callable[[], object],
    *,
    max_attempts: int = 5,
    base_delay_s: float = 2.0,
    max_delay_s: float = 60.0,
    retry_if: Callable[[BaseException], bool],
    on_retry: Callable[[int, BaseException, float], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    rand: Callable[[], float] = random.random,
):
    """Call ``fn`` with exponential backoff + jitter.

    ``retry_if(exc)`` decides retryability. Non-retryable exceptions propagate
    immediately. The last exception propagates after ``max_attempts``.
    """
    attempt = 0
    while True:
        attempt += 1
        try:
            return fn()
        except BaseException as exc:  # noqa: BLE001 - re-raised below
            if attempt >= max_attempts or not retry_if(exc):
                raise
            delay = min(max_delay_s, base_delay_s * (2 ** (attempt - 1)))
            delay = delay * (0.5 + rand() * 0.5)
            retry_after = getattr(exc, "retry_after", None)
            if retry_after:
                delay = max(delay, float(retry_after))
            if on_retry is not None:
                on_retry(attempt, exc, delay)
            sleep(delay)
