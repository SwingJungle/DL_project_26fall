"""Unit tests for structured message canonicalization (P3.2 / P3.3)."""
import unittest

from src.compress.structured import (
    StructuredFormatError,
    canonicalize,
    parse_message,
    serialize,
)


class TestStructured(unittest.TestCase):
    def test_canonicalizes_whitespace_and_key_order(self):
        raw = '{"f": [ {"x": "Kulgel was founded in 2045", "k": "D"} ], "v": 1}'
        text, parsed = canonicalize(raw)
        self.assertEqual(text, '{"v":1,"f":[{"k":"D","x":"Kulgel was founded in 2045"}]}')
        self.assertEqual(parsed["f"][0]["k"], "D")

    def test_preserves_unicode_without_escapes(self):
        raw = '{"v":1,"f":[{"k":"E","x":"Café Zündge"}]}'
        text, _ = canonicalize(raw)
        self.assertIn("Café Zündge", text)

    def test_rejects_extra_top_level_key(self):
        with self.assertRaises(StructuredFormatError):
            parse_message('{"v":1,"f":[{"k":"E","x":"abc"}],"note":"x"}')

    def test_rejects_extra_item_key(self):
        with self.assertRaises(StructuredFormatError):
            parse_message('{"v":1,"f":[{"k":"E","x":"abc","w":1}]}')

    def test_rejects_bad_version(self):
        with self.assertRaises(StructuredFormatError):
            parse_message('{"v":2,"f":[{"k":"E","x":"abc"}]}')

    def test_rejects_bad_kind(self):
        with self.assertRaises(StructuredFormatError):
            parse_message('{"v":1,"f":[{"k":"Q","x":"abc"}]}')

    def test_rejects_fence(self):
        with self.assertRaises(StructuredFormatError):
            parse_message('```json\n{"v":1,"f":[{"k":"E","x":"abc"}]}\n```')

    def test_rejects_empty_fact_list(self):
        with self.assertRaises(StructuredFormatError):
            parse_message('{"v":1,"f":[]}')

    def test_serialize_is_compact(self):
        msg = {"v": 1, "f": [{"k": "E", "x": "a b"}]}
        self.assertEqual(serialize(msg), '{"v":1,"f":[{"k":"E","x":"a b"}]}')


if __name__ == "__main__":
    unittest.main()
