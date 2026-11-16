"""Prompt contract tests: versions, byte window, format separation."""
import unittest

from src.compress.budget import window_from_ratio
from src.compress.prompts import (
    PROMPT_VERSION_ABS,
    PROMPT_VERSION_COMPACT,
    PROMPT_VERSION_RECEIVER,
    PROMPT_VERSION_RECEIVER_FULL,
    PROMPT_VERSION_STRUCT,
    build_receiver_prompt,
    build_sender_prompt,
)


class TestPrompts(unittest.TestCase):
    def setUp(self):
        self.window = window_from_ratio(1000, 0.50, 0.02)
        self.doc = "The Kulgel Foundation was founded in 2045."

    def test_versions_pinned(self):
        self.assertEqual(PROMPT_VERSION_ABS, "p4.1-abs")
        self.assertEqual(PROMPT_VERSION_STRUCT, "p4.1-struct")
        self.assertEqual(PROMPT_VERSION_COMPACT, "p6.0-compact")
        self.assertEqual(PROMPT_VERSION_RECEIVER, "r1.0")
        self.assertEqual(PROMPT_VERSION_RECEIVER_FULL, "r1.0-full")

    def test_window_appears_in_prompt(self):
        prompt = build_sender_prompt("abstractive", self.doc, self.window)
        self.assertIn("480", prompt)
        self.assertIn("520", prompt)
        self.assertIn("500", prompt)
        self.assertIn(self.doc, prompt)

    def test_methods_differ_only_in_format(self):
        abs_prompt = build_sender_prompt("abstractive", self.doc, self.window)
        struct_prompt = build_sender_prompt("structured", self.doc, self.window)
        self.assertIn("plain prose", abs_prompt)
        self.assertNotIn('"v":1', abs_prompt)
        self.assertIn('"v":1', struct_prompt)
        self.assertIn("k then x", struct_prompt)
        self.assertIn("exactly 7 items", struct_prompt)
        self.assertNotIn("{stmt_bytes}", struct_prompt)
        self.assertIn("LENGTH PLAN", struct_prompt)
        # shared semantic rule block must be present in both
        for prompt in (abs_prompt, struct_prompt):
            self.assertIn("PRIORITY", prompt)
            self.assertIn("Hard limits", prompt)
            self.assertIn("Never truncate", prompt)
            self.assertIn("Do not change, round", prompt)

    def test_retry_prompt_over(self):
        prompt = build_sender_prompt(
            "abstractive",
            self.doc,
            self.window,
            previous_text="x" * 600,
            previous_bytes=600,
            problem="over",
        )
        self.assertIn("PREVIOUS ATTEMPT", prompt)
        self.assertIn("hard maximum", prompt)

    def test_retry_prompt_under(self):
        prompt = build_sender_prompt(
            "structured",
            self.doc,
            self.window,
            previous_text='{"v":1,"f":[]}',
            previous_bytes=13,
            problem="under",
        )
        self.assertIn("below the", prompt)

    def test_receiver_prompt(self):
        questions = [{"qid": "q1", "text": "Who founded it?"}]
        prompt = build_receiver_prompt("MESSAGE", questions)
        self.assertIn("q1", prompt)
        self.assertIn("MESSAGE", prompt)
        self.assertIn('"answers"', prompt)
        full = build_receiver_prompt("DOC", questions, source="full")
        self.assertIn("The document is complete.", full)


if __name__ == "__main__":
    unittest.main()
