"""Primary-set generator: facts first, text rendered from the fact table.

Design contract (see docs/primary_set_spec.md):
  * the fact table is the single source of truth; gold answers come from it
  * sentence text is a projection of the fact table, never the reverse
  * every document is deterministic given (seed, doc_index)
  * a document that fails validation is regenerated, never patched

Two namespaces are deliberately kept apart:
  * `renderer` — which sentence template produced a fact (12 slots)
  * `type`     — the analysis category for fact-level breakdown (8 categories)

CLI:
    python -m src.data.generate --n 200 --seed 20261111 --out data
    python -m src.data.generate --n 1 --dry-run
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------
# 1. Lexicon
# --------------------------------------------------------------------------

ONSETS = ["n", "t", "v", "m", "k", "r", "s", "l", "d", "f", "b", "p", "z", "g"]
VOWELS = ["a", "e", "i", "o", "u", "ae", "ia", "eo"]
CODAS = ["l", "r", "n", "s", "th", "sk", "rn", "nd", "", "m"]

REAL_NAME_BLACKLIST = {
    "alan", "mira", "john", "mary", "david", "sarah", "michael", "anna",
    "london", "paris", "tokyo", "beijing", "berlin", "new york",
    "google", "openai", "anthropic", "deepseek", "alibaba", "microsoft",
    "stanford", "oxford", "cambridge", "harvard",
}

ORG_SUFFIX = ["Institute", "Survey", "Consortium", "Foundation", "Bureau", "Guild"]

DOMAINS: dict[str, dict[str, list[str]]] = {
    "institute": {
        "task": ["atmospheric calibration", "signal verification", "archive indexing"],
        "artifact": ["sensor", "telemetry", "reference"],
        "project": ["2045 expansion", "northern annex", "coastal extension"],
        "council": ["funding council", "oversight board", "charter committee"],
    },
    "expedition": {
        "task": ["glacial mapping", "core sampling", "traverse logistics"],
        "artifact": ["drill", "seismic", "navigation"],
        "project": ["southern traverse", "winter resupply", "ridge survey"],
        "council": ["logistics board", "expedition committee", "transport office"],
    },
    "consortium": {
        "task": ["standards drafting", "interoperability testing", "audit scheduling"],
        "artifact": ["validation", "conformance", "reference"],
        "project": ["tier-two rollout", "second phase", "joint pilot"],
        "council": ["steering group", "technical board", "member assembly"],
    },
    "archive": {
        "task": ["manuscript conservation", "catalogue reconciliation", "provenance review"],
        "artifact": ["humidity", "light", "storage"],
        "project": ["east wing", "digitisation wave", "loan programme"],
        "council": ["trustees", "acquisitions panel", "conservation office"],
    },
}


def synth_word(rng: random.Random, syllables: int = 2) -> str:
    """Invented stem, e.g. 'neral', 'tesk', 'varn'."""
    return "".join(
        rng.choice(ONSETS) + rng.choice(VOWELS) + rng.choice(CODAS)
        for _ in range(syllables)
    ).capitalize()


# --------------------------------------------------------------------------
# 2. Fact schema
# --------------------------------------------------------------------------

#: analysis categories — the only set used for fact-level breakdown
FACT_TYPES = [
    "entity",
    "number_date",
    "relation_pos",
    "relation_neg",
    "condition_exception",
    "temporal",
    "causal",
    "attribution_uncertainty",
]


@dataclass
class Fact:
    fid: str
    renderer: str
    type: str
    sentence_idx: int
    text: str
    question: str
    gold: str
    answerable: bool = True

    @property
    def words(self) -> int:
        return len(self.text.split())


@dataclass
class Doc:
    doc_id: str
    split: str
    domain: str
    text: str
    facts: list[Fact]
    roles: dict[str, str] = field(default_factory=dict)
    questions: list[dict[str, Any]] = field(default_factory=list)

    @property
    def n_facts(self) -> int:
        return len(self.facts)

    @property
    def words(self) -> int:
        return len(self.text.split())

    @property
    def nbytes(self) -> int:
        return len(self.text.encode("utf-8"))

    @property
    def bytes_per_fact(self) -> float:
        return self.nbytes / self.n_facts


# --------------------------------------------------------------------------
# 3. Renderers — 12 slots over 8 analysis categories
# --------------------------------------------------------------------------

RENDERERS: dict[str, dict[str, Any]] = {
    "entity": {
        "type": "entity",
        "variants": [
            "The {org} is the body responsible for {task} in {city1}.",
            "In {city1}, the {org} handles {task} for the wider region.",
            "The {org}, based in {city1}, is responsible for {task}.",
            "Responsibility for {task} in {city1} rests with the {org}.",
        ],
        "question": "Which organisation is responsible for {task} in {city1}?",
        "gold": "{org}",
    },
    "founding": {
        "type": "number_date",
        "variants": [
            "The {org} was founded in {year1}.",
            "Its founding charter was signed in {year1}.",
            "The {org} came into existence in {year1}, after a short preparatory period.",
            "Founding records for the {org} are dated to {year1} and were never revised.",
        ],
        "question": "In what year was the {org} founded?",
        "gold": "{year1}",
    },
    "relation_a": {
        "type": "relation_pos",
        "variants": [
            "The {artifact1} array is maintained by {p1}.",
            "{p1} is responsible for the day-to-day upkeep of the {artifact1} array.",
            "Upkeep of the {artifact1} array falls to {p1}, who reports quarterly.",
            "The {artifact1} array has been under {p1} since the last reorganisation.",
        ],
        "question": "Who maintains the {artifact1} array?",
        "gold": "{p1}",
    },
    "relation_b": {
        "type": "relation_pos",
        "variants": [
            "A second installation, the {artifact2} probe, is run by {p3}.",
            "The {artifact2} probe answers to {p3} rather than to the {council}.",
            "{p3} took over the {artifact2} probe after the previous keeper retired.",
            "Operational control of the {artifact2} probe sits with {p3}.",
        ],
        "question": "Who runs the {artifact2} probe?",
        "gold": "{p3}",
    },
    "negation": {
        "type": "relation_neg",
        "variants": [
            "The {org} was founded by {p1}, not by {p2}, despite records that named {p2}.",
            "Contrary to earlier records, {p2} did not found the {org}; {p1} did.",
            "Although {p2} is often credited, the {org} was in fact founded by {p1}.",
            "It was {p1}, not {p2}, who founded the {org}.",
        ],
        "question": "Did {p2} found the {org}?",
        "gold": "No",
    },
    "condition_a": {
        "type": "condition_exception",
        "variants": [
            "The {artifact1} array operates only when the ambient temperature stays below {thresh} degrees Celsius.",
            "Operation of the {artifact1} array is permitted only below {thresh} degrees Celsius; above that it shuts down.",
            "The {artifact1} array runs only under {thresh} degrees Celsius, and stops itself above that threshold.",
            "Above {thresh} degrees Celsius the {artifact1} array does not operate; below it, it runs normally.",
        ],
        "question": "Under what condition does the {artifact1} array operate?",
        "gold": "only below {thresh} degrees Celsius",
    },
    "condition_b": {
        "type": "condition_exception",
        "variants": [
            "The {artifact2} probe is rated for continuous use below {thresh2} degrees Celsius only.",
            "Use of the {artifact2} probe above {thresh2} degrees Celsius is not permitted.",
            "The {artifact2} probe tolerates readings up to {thresh2} degrees Celsius and no further.",
            "Operating the {artifact2} probe requires temperatures under {thresh2} degrees Celsius.",
        ],
        "question": "What temperature limit applies to the {artifact2} probe?",
        "gold": "below {thresh2} degrees Celsius",
    },
    "temporal": {
        "type": "temporal",
        "variants": [
            "The {org} relocated from {city1} to {city2} in {year2}.",
            "In {year2} the {org} moved its operations from {city1} to {city2}.",
            "A relocation in {year2} took the {org} from {city1} to {city2}.",
            "The move from {city1} to {city2} happened in {year2} and was completed within a year.",
        ],
        "question": "In what year did the {org} relocate from {city1} to {city2}?",
        "gold": "{year2}",
    },
    "causal": {
        "type": "causal",
        "variants": [
            "The planned {project} was delayed because the {council} withheld the second tranche.",
            "Because the {council} did not release the second tranche, the {project} slipped.",
            "The {project} was held back after the {council} declined to release the second tranche.",
            "Funding for the {project} stalled when the {council} withheld the second tranche.",
        ],
        "question": "Why was the {project} delayed?",
        "gold": "the {council} withheld the second tranche",
    },
    "attribution": {
        "type": "attribution_uncertainty",
        "variants": [
            "According to {director}, the {project}, planned for {year3}, remains uncertain.",
            "{director} said the {project}, scheduled for {year3}, is still uncertain.",
            "The {project}, planned for {year3}, remains uncertain according to {director}.",
            "According to {director}, no decision has been taken on the {project}, planned for {year3}.",
        ],
        "question": "Who stated that the {project} was uncertain?",
        "gold": "{director}",
    },
    "quantity": {
        "type": "number_date",
        "variants": [
            "The {artifact1} array carries {count} sealed units.",
            "There are {count} sealed units in the {artifact1} array.",
            "The {artifact1} array was built with {count} sealed units in total.",
            "Each {artifact1} array holds {count} sealed units, none of them interchangeable.",
        ],
        "question": "How many sealed units does the {artifact1} array carry?",
        "gold": "{count}",
    },
    "location": {
        "type": "entity",
        "variants": [
            "A separate observatory in {city2} is operated by the same {council}.",
            "The {council} also runs an observatory in {city2}.",
            "An observatory in {city2}, run by the {council}, reports on a twelve-hour cycle.",
            "The {council} keeps a second site, an observatory, in {city2}.",
        ],
        "question": "Where is the observatory operated by the {council}?",
        "gold": "{city2}",
    },
}

#: long-form surface variants written natively (~20-27 words) so that the
#: word budget can be met without stacking hedges onto short sentences.
LONG_VARIANTS: dict[str, list[str]] = {
    "entity": [
        "The {org} is the body responsible for {task} in {city1}, and it has held that "
        "responsibility since the present arrangements were put in place.",
    ],
    "founding": [
        "The {org} was formally founded in {year1}, and its founding charter was registered "
        "in the same year by the office then responsible for such matters.",
    ],
    "relation_a": [
        "The {artifact1} array is maintained by {p1}, who took over its upkeep when the "
        "previous keeper left and has held the responsibility ever since.",
    ],
    "relation_b": [
        "A second installation, the {artifact2} probe, is run by {p3} rather than by the "
        "{council}, and it has been run that way for some years now.",
    ],
    "negation": [
        "The {org} was founded by {p1}, not by {p2}, even though several early accounts "
        "named {p2} as the founder.",
    ],
    "condition_a": [
        "The {artifact1} array operates only when the ambient temperature stays below "
        "{thresh} degrees Celsius, and it shuts itself down once that threshold is crossed.",
    ],
    "condition_b": [
        "The {artifact2} probe is rated for continuous use below {thresh2} degrees Celsius "
        "only, and readings taken above that point are treated as invalid.",
    ],
    "temporal": [
        "The {org} relocated from {city1} to {city2} in {year2}, taking its records with it "
        "and leaving the old site in the hands of a caretaker.",
    ],
    "causal": [
        "The planned {project} was delayed because the {council} withheld the second "
        "tranche, and no revised schedule has been issued since.",
    ],
    "attribution": [
        "According to {director}, the {project}, planned for {year3}, remains uncertain, "
        "and no timetable has been agreed for reaching a decision.",
    ],
    "quantity": [
        "The {artifact1} array carries {count} sealed units, each of which is keyed to the "
        "array it was built for.",
    ],
    "location": [
        "A separate observatory in {city2} is operated by the same {council}, and it "
        "reports on a twelve-hour cycle.",
    ],
}

#: renderers that must appear in every document
REQUIRED_RENDERERS = [
    "entity",
    "founding",
    "temporal",
    "attribution",
    "negation",
    "condition_a",
]

#: narrative order. Sentences are emitted in this order with a small random
#: jitter, so documents read coherently without being identical across seeds.
NARRATIVE_RANK = {
    "entity": 0,
    "founding": 1,
    "negation": 2,
    "relation_a": 3,
    "relation_b": 4,
    "quantity": 5,
    "condition_a": 6,
    "condition_b": 7,
    "location": 8,
    "temporal": 9,
    "causal": 10,
    "attribution": 11,
}

MULTIHOP_TEMPLATES = [
    {
        # difference question: needs the founding year and the relocation year
        "needs": ("founding", "temporal"),
        "question": "How many years after its founding did the {org} relocate to {city2}?",
        "answer": lambda r: str(int(r["year2"]) - int(r["year1"])),
    },
    {
        # ordinal question: needs the relocation year and the project year,
        # whose order varies by construction
        "needs": ("temporal", "attribution"),
        "question": "Which came first, the {org} relocation to {city2} or the {project} planned for {year3}?",
        "answer": lambda r: "the relocation"
        if int(r["year2"]) < int(r["year3"])
        else "the {project}".format(**r),
    },
]

#: attachable adverbial phrases. They add words without adding queryable
#: propositions — this is what lets the source stay "dense" at 22 words/fact
#: while the fact table stays at 8-10 items.
PAD_PHRASES = [
    "under the arrangements then in force",
    "as the register records it",
    "in line with earlier practice",
    "for the duration of the programme",
    "without further comment in the record",
    "a point that was not revisited",
    "alongside the office's other duties",
    "in keeping with the practice of the time",
    "for reasons that were never written down",
    "during the period covered by this account",
]

UNANSWERABLE = [
    "How many staff does the {org} employ at present?",
    "What was the {org} total operating budget in {year1}?",
    "Which official in {city2} approved the {project}?",
    "Was the {artifact1} array commissioned before or after {year1}?",
]


# --------------------------------------------------------------------------
# 4. Role binding
# --------------------------------------------------------------------------

def bind_roles(domain: str, rng: random.Random) -> dict[str, str]:
    pool = DOMAINS[domain]
    org = f"{synth_word(rng)} {rng.choice(ORG_SUFFIX)}"

    p1, p2, p3 = (synth_word(rng) for _ in range(3))
    while len({p1, p2, p3}) < 3:
        p1, p2, p3 = (synth_word(rng) for _ in range(3))

    city1, city2 = synth_word(rng), synth_word(rng)
    while city2 == city1:
        city2 = synth_word(rng)

    artifact1, artifact2 = rng.sample(pool["artifact"], 2)

    # relocation always post-dates the founding (an organisation cannot move
    # before it exists); the project year straddles the relocation so that the
    # temporal multi-hop question is not guessable from priors alone
    year1 = rng.randint(2034, 2052)
    year2 = year1 + rng.choice([2, 3, 4, 5])
    # centred on the relocation so the ordinal multi-hop stays near 50/50;
    # resample rather than clamp, otherwise the rejected tail skews the split
    for _ in range(30):
        year3 = year2 + rng.choice([-3, -2, -1, 1, 2, 3])
        if year3 > year1:
            break
    else:
        year3 = year2 + 1

    return {
        "org": org,
        "artifact1": artifact1,
        "artifact2": artifact2,
        "task": rng.choice(pool["task"]),
        "project": rng.choice(pool["project"]),
        "council": rng.choice(pool["council"]),
        "director": f"{synth_word(rng)} {synth_word(rng)}",
        "p1": p1,
        "p2": p2,
        "p3": p3,
        "city1": city1,
        "city2": city2,
        "year1": str(year1),
        "year2": str(year2),
        "year3": str(year3),
        "thresh": str(rng.randint(8, 16)),
        "thresh2": str(rng.randint(8, 16)),
        "count": str(rng.randint(12, 96)),
    }


# --------------------------------------------------------------------------
# 5. Document assembly
# --------------------------------------------------------------------------

def candidate_texts(
    renderer: str, roles: dict[str, str], used_pads: set[str]
) -> list[tuple[str, str | None]]:
    """Surface realisations of one fact as (text, pad_used) pairs.

    Pads already used elsewhere in the document are excluded so that no
    document repeats the same hedge twice.
    """
    out: list[tuple[str, str | None]] = []
    for variant in RENDERERS[renderer]["variants"]:
        base = variant.format(**roles)
        out.append((base, None))
        for pad in PAD_PHRASES:
            if pad in used_pads:
                continue
            out.append((base.rstrip(".") + ", " + pad + ".", pad))
    # long variants already carry their own subordinate clause: never pad them,
    # otherwise the sentence accumulates two trailing clauses and reads badly
    for variant in LONG_VARIANTS.get(renderer, []):
        out.append((variant.format(**roles), None))
    return out


def choose_renderers(rng: random.Random, n_facts: int) -> list[str]:
    rest = [r for r in RENDERERS if r not in REQUIRED_RENDERERS]
    need = n_facts - len(REQUIRED_RENDERERS)
    assert 0 <= need <= len(rest), f"{n_facts} facts not supported"
    chosen = REQUIRED_RENDERERS + rng.sample(rest, need)
    rng.shuffle(chosen)
    return chosen


def render_doc(domain: str, n_facts: int, rng: random.Random) -> Doc:
    roles = bind_roles(domain, rng)
    renderers = choose_renderers(rng, n_facts)

    # narrative order with jitter: coherent, but not identical across documents
    rest = [r for r in renderers if r != "entity"]
    rest.sort(key=lambda r: NARRATIVE_RANK[r] + rng.uniform(-0.45, 0.45))
    order = ["entity"] + rest

    # greedy quota: give every fact a share of the remaining word budget
    target_words = 22 * n_facts
    texts: list[str] = []
    used_pads: set[str] = set()
    remaining = target_words
    for i, rid in enumerate(order):
        quota = remaining / (len(order) - i)
        options = candidate_texts(rid, roles, used_pads)
        # prefer a bare variant when it is within ~2 words of the quota, so that
        # hedges stay occasional instead of appearing on every sentence
        scored = sorted(
            options,
            key=lambda o: abs(len(o[0].split()) - quota) + (0.0 if o[1] is None else 2.0),
        )
        text, pad = rng.choice(scored[:3])
        if pad:
            used_pads.add(pad)
        texts.append(text)
        remaining -= len(text.split())

    facts: list[Fact] = []
    for idx, (rid, text) in enumerate(zip(order, texts)):
        spec = RENDERERS[rid]
        facts.append(
            Fact(
                fid=f"f{idx + 1}",
                renderer=rid,
                type=spec["type"],
                sentence_idx=idx,
                text=text,
                question=spec["question"].format(**roles),
                gold=spec["gold"].format(**roles),
            )
        )

    return Doc(
        doc_id="",
        split="",
        domain=domain,
        text=" ".join(f.text for f in facts),
        facts=facts,
        roles=roles,
    )


def add_questions(doc: Doc, rng: random.Random) -> None:
    # one fact per document is left without a lookup question, so that the
    # 10% unanswerable item is a substitution rather than an extra question
    optional = [f for f in doc.facts if f.renderer not in REQUIRED_RENDERERS]
    drop = rng.choice(optional).fid if optional else doc.facts[-1].fid
    qs: list[dict[str, Any]] = [
        {
            "qid": "",
            "doc_id": doc.doc_id,
            "demand": "lookup",
            "text": f.question,
            "gold": f.gold,
            "answerable": True,
            "fact_ids": [f.fid],
            "fact_types": [f.type],
        }
        for f in doc.facts
        if f.fid != drop
    ]

    by_renderer = {f.renderer: f for f in doc.facts}
    for spec in MULTIHOP_TEMPLATES:
        a, b = spec["needs"]
        if a not in by_renderer or b not in by_renderer:
            continue
        fa, fb = by_renderer[a], by_renderer[b]
        if fa.sentence_idx == fb.sentence_idx:  # cross-sentence is mandatory
            continue
        qs.append(
            {
                "qid": "",
                "doc_id": doc.doc_id,
                "demand": "multi-hop",
                "text": spec["question"].format(**doc.roles),
                "gold": spec["answer"](doc.roles),
                "answerable": True,
                "fact_ids": [fa.fid, fb.fid],
                "fact_types": [fa.type, fb.type],
            }
        )

    qs.append(
        {
            "qid": "",
            "doc_id": doc.doc_id,
            "demand": "lookup",
            "text": rng.choice(UNANSWERABLE).format(**doc.roles),
            "gold": "unknown",
            "answerable": False,
            "fact_ids": [],
            "fact_types": [],
        }
    )

    rng.shuffle(qs)
    for i, q in enumerate(qs, start=1):
        q["qid"] = f"{doc.doc_id}-q{i}"
    doc.questions = qs


# --------------------------------------------------------------------------
# 6. Validation
# --------------------------------------------------------------------------

TARGET_BPF = 139.0     # bytes per fact at 50% budget, 22 words/fact
WORDS_PER_FACT = 22


def validate(doc: Doc) -> list[str]:
    p: list[str] = []
    n = doc.n_facts

    if not 8 <= n <= 10:
        p.append(f"n_facts={n}")

    target_words = WORDS_PER_FACT * n
    if not 0.9 * target_words <= doc.words <= 1.1 * target_words:
        p.append(f"words={doc.words}/{target_words}")

    bpf = doc.bytes_per_fact
    if not 0.9 * TARGET_BPF <= bpf <= 1.1 * TARGET_BPF:
        p.append(f"bytes_per_fact={bpf:.1f}")

    types = {f.type for f in doc.facts}
    if len(types) < 5:
        p.append(f"type_coverage={len(types)}")
    for req in ("relation_neg", "condition_exception", "attribution_uncertainty"):
        if req not in types:
            p.append(f"missing_type={req}")

    mh = [q for q in doc.questions if q["demand"] == "multi-hop"]
    if len(mh) != 2:
        p.append(f"multi_hop={len(mh)}")
    idx = {f.fid: f.sentence_idx for f in doc.facts}
    for q in mh:
        a, b = q["fact_ids"]
        if idx.get(a) == idx.get(b):
            p.append("multi_hop_same_sentence")

    if len(doc.questions) != doc.n_facts + 2:
        p.append(f"n_questions={len(doc.questions)}/expected {doc.n_facts + 2}")

    unans = sum(1 for q in doc.questions if not q["answerable"])
    if not 0.06 <= unans / len(doc.questions) <= 0.15:
        p.append(f"unanswerable={unans}/{len(doc.questions)}")

    lowered = doc.text.lower()
    for bad in REAL_NAME_BLACKLIST:
        if bad in lowered:
            p.append(f"real_name_leak={bad}")

    return p


# --------------------------------------------------------------------------
# 7. Driver
# --------------------------------------------------------------------------

SPLIT_SIZES = (("pilot", 25), ("dev", 30), ("eval", 145))


def assign_splits(n: int) -> list[str]:
    out: list[str] = []
    for name, size in SPLIT_SIZES:
        out.extend([name] * min(size, max(0, n - len(out))))
    while len(out) < n:
        out.append("eval")
    return out[:n]


def build(n: int, seed: int, max_retry: int = 60) -> tuple[list[Doc], dict[str, int]]:
    rng = random.Random(seed)
    splits = assign_splits(n)
    domains = list(DOMAINS)
    docs: list[Doc] = []
    seen: set[str] = set()
    stats = {"attempts": 0, "rejected": 0}

    for i, split in enumerate(splits):
        for _ in range(max_retry):
            stats["attempts"] += 1
            n_facts = rng.choice([8, 9, 10])
            doc = render_doc(rng.choice(domains), n_facts, rng)
            doc.doc_id = f"p-{i:04d}"
            doc.split = split
            add_questions(doc, rng)
            problems = validate(doc)
            digest = hashlib.sha256(doc.text.encode()).hexdigest()
            if problems or digest in seen:
                stats["rejected"] += 1
                continue
            seen.add(digest)
            docs.append(doc)
            break
        else:
            raise RuntimeError(f"doc {i}: no valid sample in {max_retry} attempts")
    return docs, stats


def to_jsonl(docs: list[Doc], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with (out / "docs.jsonl").open("w", encoding="utf-8") as fd, (
        out / "questions.jsonl"
    ).open("w", encoding="utf-8") as fq:
        for d in docs:
            fd.write(
                json.dumps(
                    {
                        "doc_id": d.doc_id,
                        "split": d.split,
                        "domain": d.domain,
                        "text": d.text,
                        "words": d.words,
                        "bytes": d.nbytes,
                        "n_facts": d.n_facts,
                        "bytes_per_fact": round(d.bytes_per_fact, 1),
                        "facts": [
                            {
                                "fid": f.fid,
                                "renderer": f.renderer,
                                "type": f.type,
                                "sentence_idx": f.sentence_idx,
                                "text": f.text,
                                "gold": f.gold,
                            }
                            for f in d.facts
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            for q in d.questions:
                fq.write(json.dumps(q, ensure_ascii=False) + "\n")


def summarize(docs: list[Doc], stats: dict[str, int]) -> None:
    n = len(docs)
    words = [d.words for d in docs]
    bpf = [d.bytes_per_fact for d in docs]
    qs = [len(d.questions) for d in docs]
    type_counts: dict[str, int] = {}
    for d in docs:
        for f in d.facts:
            type_counts[f.type] = type_counts.get(f.type, 0) + 1
    print(f"documents      : {n}")
    print(f"attempts       : {stats['attempts']}   rejected: {stats['rejected']}")
    print(f"words          : min {min(words)}  mean {sum(words)/n:.1f}  max {max(words)}")
    print(f"bytes_per_fact : min {min(bpf):.0f}  mean {sum(bpf)/n:.1f}  max {max(bpf):.0f}")
    print(f"questions/doc  : min {min(qs)}  max {max(qs)}")
    print("fact type mix  : " + ", ".join(f"{k}={v}" for k, v in sorted(type_counts.items())))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20261111)
    ap.add_argument("--out", type=str, default="data")
    ap.add_argument("--dry-run", action="store_true", help="print one sample document")
    args = ap.parse_args()

    docs, stats = build(args.n, args.seed)

    if args.dry_run:
        d = docs[0]
        print(f"--- {d.doc_id} [{d.split}] {d.domain}  words={d.words}  bpf={d.bytes_per_fact:.0f}\n")
        print(d.text + "\n")
        for q in d.questions:
            print(f"  [{q['demand']:9s}] {q['text']}\n              -> {q['gold']}")
        return

    to_jsonl(docs, Path(args.out))
    summarize(docs, stats)
    print(f"\nwrote {args.out}/docs.jsonl and {args.out}/questions.jsonl")


if __name__ == "__main__":
    main()
