"""Adapter tests with a mocked transport: no network, no real credentials."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.api import adapters
from src.api.cache import CompletionCache
from src.api.calls import CallLogger
from src.api.retry import BudgetGuard, RateLimiter


def _fake_body(payload, model=None, content='{"ok":true}'):
    return {
        "id": "req-1",
        "model": model or payload["model"],
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "prompt_tokens_details": {"cached_tokens": 0},
            "completion_tokens_details": {"reasoning_tokens": 0},
        },
    }


class TestAdapter(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.ctx = adapters.RunContext(
            run_id="test",
            cache=CompletionCache(root / "cache.sqlite3"),
            logger=CallLogger(root / "calls.jsonl"),
            guard=BudgetGuard(max_calls=10),
            limiter=RateLimiter(0.0),
        )
        self.payloads = []

    def tearDown(self):
        self.ctx.cache.close()
        self.tmp.cleanup()

    def _patched(self, **kwargs):
        def fake_post(url, api_key, payload, *, timeout_s):
            self.payloads.append(payload)
            return _fake_body(payload, **kwargs), {"x-request-id": "h1"}, 12

        return mock.patch.object(adapters, "_http_post", side_effect=fake_post)

    def test_complete_sets_pinned_flags_and_caches(self):
        with self._patched():
            a = adapters.complete(
                "qwen3.7-plus",
                [{"role": "user", "content": "hi"}],
                json_schema={"type": "object"},
                tag="t",
                prompt_version="v1",
                ctx=self.ctx,
            )
            b = adapters.complete(
                "qwen3.7-plus",
                [{"role": "user", "content": "hi"}],
                json_schema={"type": "object"},
                tag="t",
                prompt_version="v1",
                ctx=self.ctx,
            )
        self.assertFalse(a.cached)
        self.assertTrue(b.cached)
        self.assertEqual(len(self.payloads), 1)
        payload = self.payloads[0]
        self.assertIs(payload["enable_thinking"], False)
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertEqual(payload["temperature"], 0.0)
        self.assertEqual(a.model_id, "qwen3.7-plus")
        self.assertEqual(a.usage["prompt_tokens"], 10)
        self.assertEqual(self.ctx.guard.calls, 1)

    def test_model_mismatch_is_fatal(self):
        with self._patched(model="qwen3.7-max"):
            with self.assertRaises(adapters.ModelMismatch):
                adapters.complete(
                    "qwen3.7-plus",
                    [{"role": "user", "content": "hi"}],
                    tag="t",
                    prompt_version="v1",
                    ctx=self.ctx,
                )

    def test_cache_nonce_forces_a_fresh_request(self):
        """Repetition studies must bypass the cache without changing the prompt."""
        with self._patched():
            adapters.complete(
                "qwen3.7-plus", [{"role": "user", "content": "hi"}],
                prompt_version="v1", cache_nonce="rep2", ctx=self.ctx,
            )
            repeated = adapters.complete(
                "qwen3.7-plus", [{"role": "user", "content": "hi"}],
                prompt_version="v1", cache_nonce="rep2", ctx=self.ctx,
            )
            other = adapters.complete(
                "qwen3.7-plus", [{"role": "user", "content": "hi"}],
                prompt_version="v1", cache_nonce="rep3", ctx=self.ctx,
            )
        self.assertTrue(repeated.cached)   # same nonce -> same cache entry
        self.assertFalse(other.cached)     # different nonce -> fresh request
        self.assertEqual(len(self.payloads), 2)

    def test_json_schema_participates_in_cache_key(self):
        with self._patched():
            adapters.complete(
                "qwen3.7-plus", [{"role": "user", "content": "hi"}],
                json_schema={"type": "object"}, ctx=self.ctx,
            )
            adapters.complete(
                "qwen3.7-plus", [{"role": "user", "content": "hi"}],
                json_schema=None, ctx=self.ctx,
            )
        self.assertEqual(len(self.payloads), 2)


if __name__ == "__main__":
    unittest.main()
