"""Unit tests for the compact schema (P6.0 / G2 ablation)."""
import json
import unittest

from src.compress.compact import (
    CompactFormatError,
    canonicalize,
    parse_message,
    reserialize_full,
    serialize,
)
from src.compress.prompts import (
    PER_ITEM_OVERHEAD,
    PROMPT_VERSION_COMPACT,
    build_receiver_prompt,
    build_sender_prompt,
)
from src.compress.budget import window_from_ratio
from src.eval.retention import message_segments


class TestCompactMessage(unittest.TestCase):
    def test_canonicalizes_whitespace(self):
        raw = '[ ["E", "Kulgel was founded in 2045"], ["D"," 2045 " ] ]'
        text, parsed = canonicalize(raw)
        self.assertEqual(text, '[["E","Kulgel was founded in 2045"],["D","2045"]]')
        self.assertEqual(parsed["f"][1]["k"], "D")

    def test_preserves_unicode_without_escapes(self):
        raw = '[["E","Café Zündge"]]'
        text, _ = canonicalize(raw)
        self.assertIn("Café Zündge", text)

    def test_rejects_object_message(self):
        with self.assertRaises(CompactFormatError):
            parse_message('{"v":1,"f":[{"k":"E","x":"abc"}]}')

    def test_rejects_three_element_item(self):
        with self.assertRaises(CompactFormatError):
            parse_message('[["E","abc",1]]')

    def test_rejects_bad_kind(self):
        with self.assertRaises(CompactFormatError):
            parse_message('[["Q","abc"]]')

    def test_rejects_short_statement(self):
        with self.assertRaises(CompactFormatError):
            parse_message('[["E","a"]]')

    def test_rejects_empty_array(self):
        with self.assertRaises(CompactFormatError):
            parse_message("[]")

    def test_rejects_fence(self):
        with self.assertRaises(CompactFormatError):
            parse_message('```json\n[["E","abc"]]\n```')

    def test_round_trip(self):
        msg = {"v": 1, "f": [{"k": "E", "x": "a b"}, {"k": "D", "x": "2045"}]}
        text = serialize(msg)
        self.assertEqual(text, '[["E","a b"],["D","2045"]]')
        self.assertEqual(parse_message(text), msg)

    def test_overhead_is_lower_than_full_schema(self):
        """The whole point: compact syntax must cost fewer bytes per item."""
        compact = serialize({"v": 1, "f": [{"k": "E", "x": "x" * 40}] * 7})
        full = json.dumps(
            {"v": 1, "f": [{"k": "E", "x": "x" * 40}] * 7},
            separators=(",", ":"),
        )
        self.assertLess(len(compact) - 7 * 40, len(full) - 7 * 40)
        self.assertLessEqual(PER_ITEM_OVERHEAD["compact"], 10)
        self.assertGreaterEqual(PER_ITEM_OVERHEAD["structured"], 18)

    def test_reserialize_full_keeps_facts(self):
        msg = {"v": 1, "f": [{"k": "N", "x": "Leomvaen did not found it"}]}
        out = reserialize_full(msg)
        self.assertEqual(json.loads(out), [["N", "Leomvaen did not found it"]])


class TestCompactPrompt(unittest.TestCase):
    def setUp(self):
        self.window = window_from_ratio(1000, 0.50, 0.02)
        self.doc = "The Kulgel Foundation was founded in 2045."

    def test_version_pinned(self):
        self.assertEqual(PROMPT_VERSION_COMPACT, "p6.0-compact")

    def test_prompt_shape(self):
        prompt = build_sender_prompt("compact", self.doc, self.window)
        self.assertIn('["E","..."]', prompt)
        self.assertIn("two-element array", prompt)
        self.assertIn("exactly 7 elements", prompt)
        self.assertIn("about 9 bytes of JSON syntax", prompt)
        self.assertNotIn('"v":1', prompt)
        self.assertIn("Never truncate", prompt)
        self.assertIn(self.doc, prompt)

    def test_compact_gets_more_statement_bytes_than_full(self):
        compact = build_sender_prompt("compact", self.doc, self.window)
        full = build_sender_prompt("structured", self.doc, self.window)
        self.assertIn("about 9 bytes of JSON syntax", compact)
        self.assertIn("about 20 bytes of JSON syntax", full)

    def test_compact_retry_prompt(self):
        prompt = build_sender_prompt(
            "compact",
            self.doc,
            self.window,
            previous_text='[["E","abc"]]',
            previous_bytes=13,
            problem="invalid_json",
            error_detail="top level is dict, expected array",
        )
        self.assertIn("invalid compact JSON", prompt)
        self.assertIn("[kind, statement]", prompt)

    def test_receiver_note_only_for_compact(self):
        questions = [{"qid": "q1", "text": "Who founded it?"}]
        compact = build_receiver_prompt("[]", questions, message_format="compact")
        full = build_receiver_prompt("{}", questions)
        self.assertIn("[kind, statement]", compact)
        self.assertNotIn("[kind, statement]", full)
        with self.assertRaises(ValueError):
            build_receiver_prompt("x", questions, message_format="bogus")


class TestCompactRetention(unittest.TestCase):
    def test_segments_use_statements(self):
        msg = '[["E","Kulgel was founded"],["D","in 2045"]]'
        self.assertEqual(
            message_segments(msg, "compact"), ["Kulgel was founded", "in 2045"]
        )

    def test_malformed_falls_back_to_whole_message(self):
        self.assertEqual(message_segments("not json", "compact"), ["not json"])


if __name__ == "__main__":
    unittest.main()
