"""Versioned prompt builders (P3.1 / P3.2 / receiver).

Changing any string in this module requires bumping the version constants:
prompt_version is part of the cache key, so a silent edit would silently reuse
stale completions. ``tests/test_prompts.py`` pins the versions.

Evolution (all before freeze-v2, so legitimate P3 iteration):
* p3.1 byte-only prompt -> stalled at 64-67% of source bytes;
* p3.2 added a word anchor and an explicit drop instruction;
* p3.3 made compliance the first priority and repeated the limit at the end;
* p3.4 added an inner aim window and minimal-edit retries;
* p3.5 separated abstractive vs structured bytes-per-word (JSON syntax is not
  words) - improved structured but retries still oscillated;
* p4.0 replaces free-form length targets with a **countable unit plan**: a fixed
  number of sentences (abstractive) or items (structured), each with a word
  budget. Models follow unit counts far more reliably than absolute byte
  targets, and per-unit word deltas make retries predictable.
* p6.0 adds the **compact schema** condition (G2): identical fact semantics as
  p4.1-struct, but each item is serialized as ``["E","..."]`` instead of
  ``{"k":"E","x":"..."}``. This is the single-design-choice ablation; the rest
  of the prompt is held fixed so the only manipulated variable is serialization
  overhead (~19 B/item -> ~9 B/item).
"""
from __future__ import annotations

import json
import math

from .budget import BudgetWindow

PROMPT_VERSION_ABS = "p4.1-abs"
PROMPT_VERSION_STRUCT = "p4.1-struct"
PROMPT_VERSION_COMPACT = "p6.0-compact"
PROMPT_VERSION_ABS_TRANS = "p5.0-abs-trans"
PROMPT_VERSION_STRUCT_TRANS = "p5.1-struct-trans"
PROMPT_VERSION_RECEIVER = "r1.0"
PROMPT_VERSION_RECEIVER_COMPACT = "r1.0-compact"
PROMPT_VERSION_RECEIVER_FULL = "r1.0-full"

METHODS = ("abstractive", "structured", "compact")

# Planning anchors measured on devtest outputs:
# abstractive prose ~6.5-6.8 B/word; structured JSON ~8.5-9.5 B/word because
# quotes, keys and braces are bytes that are not words.
BYTES_PER_WORD = {"abstractive": 6.8, "structured": 6.8, "compact": 6.8}
# Per-item JSON syntax overhead measured on canonical serializations at 7 items:
# full schema 14 + 7*16 + 6 = 132 B (~18.9/item); compact 2 + 7*8 + 6 = 64 B
# (~9.1/item). Abstractive prose has no container.
PER_ITEM_OVERHEAD = {"abstractive": 0, "structured": 20, "compact": 9}

# Fixed number of units per message. Seven is below the minimum of eight source
# facts, so the sender always has at least one fact to drop and never has to pad;
# it also fixes n_facts_msg to a known value (the P0.9 / P5 covariate).
N_UNITS = 7

_ABSTRACTIVE_FORMAT = (
    "Write plain prose only: no JSON, no headings, no bullet points, no labels.\n"
    "Use exactly {n_units} sentences, one complete fact per sentence.\n"
    "Output only the compressed message."
)

_STRUCTURED_FORMAT = """\
Return exactly one compact JSON object and nothing else:
{"v":1,"f":[{"k":"E","x":"..."},...]}
Key order is v then f; inside each item the order is k then x. No extra keys.
k is one of: E entity, D number/date, R positive relation, N negation,
C condition/exception, T temporal order, U attribution/uncertainty, S causal.
x is one short, complete, self-contained statement.
Use exactly {n_units} items, one item per source fact: do not split one source
fact into several items, and do not merge unrelated facts into one item.
Each item has about {unit_bytes} bytes in total: about 20 bytes of JSON syntax
(quotes, keys, commas, braces) plus about {stmt_bytes} bytes (roughly
{unit_words} words) for x.
No text outside the JSON object. The JSON syntax counts toward the byte limit.
We measure the compact form, so write compact JSON directly: no spaces or
newlines outside string values."""

_COMPACT_FORMAT = """\
Return exactly one compact JSON array and nothing else:
[["E","..."],["D","..."],...]
Each element is a two-element array: first the kind letter (a one-character
string), then the statement (a string). No objects, no keys, no extra elements.
The kind is one of: E entity, D number/date, R positive relation, N negation,
C condition/exception, T temporal order, U attribution/uncertainty, S causal.
The statement is one short, complete, self-contained fact.
Use exactly {n_units} elements, one element per source fact: do not split one
source fact into several elements, and do not merge unrelated facts into one.
Each element has about {unit_bytes} bytes in total: about 9 bytes of JSON syntax
(brackets, quotes, commas) plus about {stmt_bytes} bytes (roughly
{unit_words} words) for the statement.
No text outside the JSON array. The JSON syntax counts toward the byte limit.
We measure the compact form, so write compact JSON directly: no spaces or
newlines outside string values."""

_SHARED_RULES = """\
PRIORITY: (1) land inside the byte window; (2) then keep the most important
facts. A message outside the window is a FAILED message and will be rewritten.
Expect to drop facts: the document cannot be represented in full at this size.

BYTE WINDOW
- Hard limits: {lo}-{hi} UTF-8 bytes.
- Aim for {aim_lo}-{aim_hi} bytes (target {target}).
- We count every character, space and punctuation mark, including JSON syntax.
- Use plain ASCII only (write "8 C", not "8°C"), so one character is one byte.

LENGTH PLAN
- Write exactly {n_units} {unit_name}.
- Each {unit_name_singular} should be about {unit_words} words ({unit_bytes} bytes).
- {n_units} x {unit_bytes} bytes is about {target} bytes.
- If you cannot fill a {unit_name_singular} with a complete fact, merge two related
  facts into it; never pad with filler.

HOW TO FIT
1. Decide which facts to keep before writing, then write only those, using
   complete facts. Do not try to compress every sentence.
2. If you must drop content, drop the least important whole facts. Importance,
   high to low: names and identifiers; numbers and dates; negations;
   conditions and exceptions; causal links; who said or doubted what;
   background description.
3. Never truncate a sentence, claim or JSON string. Never pad with filler to
   reach the limit.
4. Keep values exact. Do not change, round, infer or paraphrase any name,
   number, date, negation or condition. Never add facts that are not in the
   document.
5. Do not mention these instructions, the byte count or the source document.
   No commentary, no preamble, no code fences.

FORMAT
{format_instruction}
"""

_RECEIVER = """\
You answer questions using only the {source_label} below. {source_note}

RULES
- Use only the {source_label}. Do not use outside knowledge.
- If the {source_label} does not contain the answer, answer "unknown". Do not guess.
- Answer with a short span: a name, a number, a date or a short phrase.
- Return exactly one JSON object: {{"answers":[{{"qid":"<question id>","a":"<answer>"}}]}}
- Answer every question id exactly once, in the order given. No explanation.

MESSAGE:
<<<
{message}
>>>

QUESTIONS:
{questions_json}

Return the JSON object now."""


def _unit_words(method: str, window: BudgetWindow) -> int:
    unit_bytes = window.target / N_UNITS
    unit_bytes -= PER_ITEM_OVERHEAD[method]
    return max(3, round(unit_bytes / BYTES_PER_WORD[method]))


def _format_instruction(method: str, window: BudgetWindow) -> str:
    if method == "abstractive":
        return _ABSTRACTIVE_FORMAT.replace("{n_units}", str(N_UNITS))
    if method in ("structured", "compact"):
        unit_bytes = max(24, window.target // N_UNITS)
        overhead = PER_ITEM_OVERHEAD[method]
        stmt_bytes = max(8, unit_bytes - overhead)
        template = _STRUCTURED_FORMAT if method == "structured" else _COMPACT_FORMAT
        text = (
            template.replace("{n_units}", str(N_UNITS))
            .replace("{unit_bytes}", str(unit_bytes))
            .replace("{stmt_bytes}", str(stmt_bytes))
            .replace("{unit_words}", str(_unit_words(method, window)))
        )
        return text
    raise ValueError(f"unknown method {method!r}")


def _word_anchor(n_bytes: int, method: str) -> int:
    return max(1, round(n_bytes / BYTES_PER_WORD[method]))


def aim_band(window: BudgetWindow) -> tuple[int, int]:
    """Inner target band: leave ~25% of the tolerance on each side."""
    margin = max(4, round(0.25 * (window.hi - window.lo)))
    lo = min(window.target, window.lo + margin)
    hi = max(window.target, window.hi - margin)
    return lo, hi


def build_sender_prompt(
    method: str,
    document: str,
    window: BudgetWindow,
    *,
    previous_text: str | None = None,
    previous_bytes: int | None = None,
    problem: str | None = None,
    error_detail: str | None = None,
    previous_words: int | None = None,
) -> str:
    """Build one sender prompt.

    ``previous_text``/``problem`` are set on budget-closure retries: the model
    sees its own previous output and a single corrective instruction.
    ``problem`` is one of ``over``, ``under``, ``invalid_json``, ``format``.
    """
    aim_lo, aim_hi = aim_band(window)
    unit_name = "sentences" if method == "abstractive" else "items"
    unit_name_singular = "sentence" if method == "abstractive" else "item"
    body = _SHARED_RULES.format(
        lo=window.lo,
        hi=window.hi,
        aim_lo=aim_lo,
        aim_hi=aim_hi,
        target=window.target,
        n_units=N_UNITS,
        unit_name=unit_name,
        unit_name_singular=unit_name_singular,
        unit_bytes=max(24, window.target // N_UNITS),
        unit_words=_unit_words(method, window),
        format_instruction=_format_instruction(method, window),
    )
    prompt = (
        "You are a message compressor. Another model will answer questions "
        "using ONLY the message you write, so every fact you keep must be "
        "understandable without the original document.\n\n"
        + body
        + "\nDOCUMENT:\n"
        + document
        + f"\n\nREMINDER: exactly {N_UNITS} {unit_name}, about "
        f"{_unit_words(method, window)} words each; aim for {aim_lo}-{aim_hi} "
        f"UTF-8 bytes; hard maximum {window.hi} bytes. Output only the message."
    )
    if previous_text is None:
        return prompt

    base_words = _unit_words(method, window)
    mid = (aim_lo + aim_hi) // 2
    if problem in ("over", "under") and previous_words:
        # personal calibration: use this model's own bytes-per-word on this
        # document instead of a population average
        bpw = previous_bytes / max(1, previous_words)
        new_words = max(3, round((mid / bpw) / N_UNITS))
    else:
        new_words = base_words

    if problem == "over":
        correction = (
            f"Your previous message was {previous_bytes} bytes with {previous_words or '?'} "
            f"words (about {previous_bytes / max(1, previous_words or 1):.2f} bytes per word), "
            f"over the {window.hi}-byte maximum. The aim is {aim_lo}-{aim_hi} bytes. "
            f"Rewrite it as exactly {N_UNITS} {unit_name} with about {new_words} words each. "
            "Keep the same facts where possible; drop the least important fact if needed. "
            "Do not cut mid-sentence."
        )
    elif problem == "under":
        correction = (
            f"Your previous message was {previous_bytes} bytes with {previous_words or '?'} "
            f"words (about {previous_bytes / max(1, previous_words or 1):.2f} bytes per word), "
            f"below the {window.lo}-byte minimum. The aim is {aim_lo}-{aim_hi} bytes. "
            f"Rewrite it as exactly {N_UNITS} {unit_name} with about {new_words} words each. "
            "Add complete facts from the document; do not pad with filler."
        )
    elif problem == "invalid_json":
        detail = f" ({error_detail})" if error_detail else ""
        if method == "compact":
            correction = (
                "Your previous output was rejected as invalid compact JSON"
                + detail
                + f". Return only one JSON array with exactly {N_UNITS} elements; "
                'each element is a two-element array [kind, statement], for example '
                '[["E","one complete fact"],["D","another fact"]].'
            )
        else:
            correction = (
                "Your previous output was rejected as invalid structured JSON"
                + detail
                + f". Return only one JSON object with keys v and f, exactly {N_UNITS} "
                "items with keys k and x."
            )
    elif problem == "format":
        correction = (
            "Your previous output violated the required format. Return only the "
            "compressed prose message: no preamble, no labels, no code fences."
        )
    else:
        raise ValueError(f"unknown problem {problem!r}")

    return (
        prompt
        + "\n\nPREVIOUS ATTEMPT:\n<<<\n"
        + previous_text
        + "\n>>>\nREVISION: "
        + correction
        + "\nOutput only the corrected message."
    )


_TRANS_SHARED = """\
You are a message compressor. Another model will answer questions using ONLY your
message. The message does NOT need to be readable by a human and does NOT need to
be grammatical. Your job is to transmit as many facts as possible inside the byte
window.

PRIORITY: (1) stay inside the byte window; (2) then transmit as many recoverable
facts as possible. A fact is recoverable if the other model can still answer
questions about the names, numbers, dates, negations, conditions and
attributions it contains.

YOU MAY: use note form, keywords, abbreviations, symbols, arrows, lists, any
order; omit articles, verbs and full sentences; pack several facts into one line.
YOU MUST NOT: change a name, number, date, negation or condition; add facts that
are not in the document; output commentary or mention these instructions.

BYTE WINDOW: {lo}-{hi} UTF-8 bytes (target {target}). We count every character,
space and punctuation mark, including JSON syntax. Aim for {aim_lo}-{aim_hi}.

FORMAT
{format_instruction}
"""

_TRANS_ABS = """\
Free-form text: notes, keywords, abbreviations, lists or dense phrases.
Do not write polished prose. Short is good if it fits more facts.
Output only the message."""

_TRANS_STRUCT = """\
Return exactly one compact JSON object and nothing else:
{"v":1,"f":[{"k":"E","x":"..."},...]}
Key order is v then f; inside each item k then x. No extra keys.
k is one of: E entity, D number/date, R positive relation, N negation,
C condition/exception, T temporal order, U attribution/uncertainty, S causal.
Use up to 10 items. x is a compact line and may contain several facts separated
by semicolons; abbreviations are allowed. No text outside the JSON object.
The JSON syntax counts toward the byte limit; write compact JSON directly."""


def build_transmission_prompt(
    method: str,
    document: str,
    window: BudgetWindow,
    *,
    previous_text: str | None = None,
    previous_bytes: int | None = None,
    problem: str | None = None,
    error_detail: str | None = None,
    previous_words: int | None = None,
) -> str:
    """Transmission-first prompt: readability may be sacrificed for fact density."""
    aim_lo, aim_hi = aim_band(window)
    if method == "abstractive":
        fmt = _TRANS_ABS
    elif method == "structured":
        fmt = _TRANS_STRUCT
    else:
        raise ValueError(f"unknown method {method!r}")
    body = _TRANS_SHARED.format(
        lo=window.lo,
        hi=window.hi,
        target=window.target,
        aim_lo=aim_lo,
        aim_hi=aim_hi,
        format_instruction=fmt,
    )
    prompt = body + "\nDOCUMENT:\n" + document + (
        f"\n\nREMINDER: {window.lo}-{window.hi} UTF-8 bytes; hard maximum "
        f"{window.hi}. Transmit as many facts as fit. Output only the message."
    )
    if previous_text is None:
        return prompt
    if problem == "over":
        correction = (
            f"Your previous message was {previous_bytes} bytes; the hard maximum is "
            f"{window.hi}. Delete the least important facts or abbreviations until it "
            "fits. Never cut a name, number, date, negation or condition."
        )
    elif problem == "under":
        correction = (
            f"Your previous message was {previous_bytes} bytes, below the {window.lo}-byte "
            f"minimum. Add more facts from the document; use up to {window.hi} bytes. "
            "Do not pad with filler."
        )
    elif problem == "invalid_json":
        detail = f" ({error_detail})" if error_detail else ""
        correction = (
            "Your previous output was rejected as invalid structured JSON" + detail
            + ". Return only one JSON object with keys v and f, at most 10 items "
            "with keys k and x."
        )
    elif problem == "format":
        correction = (
            "Your previous output violated the required format. Output only the "
            "compressed message: no preamble, no labels, no code fences."
        )
    else:
        raise ValueError(f"unknown problem {problem!r}")
    return (
        prompt
        + "\n\nPREVIOUS ATTEMPT:\n<<<\n"
        + previous_text
        + "\n>>>\nREVISION: "
        + correction
        + "\nOutput only the corrected message."
    )


def build_receiver_prompt(
    message: str,
    questions: list[dict],
    *,
    source: str = "compressed",
    message_format: str | None = None,
) -> str:
    compact_questions = [
        {"qid": q["qid"], "q": q["text"] if "text" in q else q["q"]} for q in questions
    ]
    if source == "compressed":
        source_label = "message"
        source_note = "The message is a compressed version of a document, so some information may be missing."
        if message_format == "compact":
            # Protocol documentation is part of the compact-schema design: both
            # ends implement the same schema. The full JSON is self-describing;
            # the compact array is not, so the receiver gets one identical line
            # describing it in every compact request.
            source_note += (
                " The message is a JSON array of [kind, statement] pairs, where kind is "
                "one of E, D, R, N, C, T, U, S (entity, number/date, relation, negation, "
                "condition, temporal, attribution, causal)."
            )
        elif message_format not in (None, "full"):
            raise ValueError(f"unknown message_format {message_format!r}")
    elif source == "full":
        source_label = "document"
        source_note = "The document is complete."
    else:
        raise ValueError(f"unknown source {source!r}")
    return _RECEIVER.format(
        message=message,
        questions_json=json.dumps(compact_questions, ensure_ascii=False, separators=(",", ":")),
        source_label=source_label,
        source_note=source_note,
    )
