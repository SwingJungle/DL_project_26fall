"""Unit tests for cache, retry and hard-stop behaviour."""
import tempfile
import unittest
from pathlib import Path

from src.api.cache import CompletionCache
from src.api.retry import BudgetExceeded, BudgetGuard, RateLimiter, retry_call


class TestCache(unittest.TestCase):
    def test_put_get_and_stats(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = CompletionCache(Path(tmp) / "cache.sqlite3")
            self.assertIsNone(cache.get("missing"))
            cache.put(
                "k1",
                {
                    "model_key": "qwen3.7-plus",
                    "provider": "qwen",
                    "api_model": "qwen3.7-plus",
                    "model_returned": "qwen3.7-plus",
                    "request_json": "{}",
                    "response_text": "hello",
                    "usage_json": '{"prompt_tokens":3,"completion_tokens":2}',
                    "request_id": "r1",
                    "finish_reason": "stop",
                    "ts": "2026-11-15T00:00:00+08:00",
                    "latency_ms": 10,
                    "cost_usd": 0.0,
                    "prompt_version": "t",
                    "tag": "t",
                },
            )
            row = cache.get("k1")
            self.assertEqual(row["text"], "hello")
            self.assertEqual(cache.count(), 1)
            stats = cache.stats()
            self.assertEqual(stats["prompt_tokens"], 3)
            self.assertEqual(stats["completion_tokens"], 2)
            cache.close()


class TestRetry(unittest.TestCase):
    def test_retries_then_succeeds(self):
        calls = {"n": 0}

        def fn():
            calls["n"] += 1
            if calls["n"] < 3:
                raise TimeoutError("boom")
            return "ok"

        out = retry_call(
            fn,
            max_attempts=3,
            base_delay_s=0.0,
            retry_if=lambda e: isinstance(e, TimeoutError),
            sleep=lambda _: None,
        )
        self.assertEqual(out, "ok")
        self.assertEqual(calls["n"], 3)

    def test_non_retryable_propagates(self):
        calls = {"n": 0}

        def fn():
            calls["n"] += 1
            raise ValueError("bad request")

        with self.assertRaises(ValueError):
            retry_call(
                fn,
                max_attempts=5,
                retry_if=lambda e: isinstance(e, TimeoutError),
                sleep=lambda _: None,
            )
        self.assertEqual(calls["n"], 1)

    def test_budget_hard_stop(self):
        guard = BudgetGuard(max_calls=1)
        guard.reserve_call()
        with self.assertRaises(BudgetExceeded):
            guard.reserve_call()

    def test_token_cap(self):
        guard = BudgetGuard(max_prompt_tokens=5)
        with self.assertRaises(BudgetExceeded):
            guard.record({"prompt_tokens": 6})

    def test_rate_limiter_noop_when_zero(self):
        RateLimiter(0.0).wait("qwen")


if __name__ == "__main__":
    unittest.main()
