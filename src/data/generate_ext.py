"""Generate the independent Institute and Expedition extension."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from statistics import mean, pstdev
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

def synth_word(rng: random.Random, syllables: int = 2) -> str:
    """Invented stem, e.g. 'neral', 'tesk', 'varn'."""
    return "".join(
        rng.choice(ONSETS) + rng.choice(VOWELS) + rng.choice(CODAS)
        for _ in range(syllables)
    ).capitalize()


# --------------------------------------------------------------------------
# 2. Fact schema
# --------------------------------------------------------------------------

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
    split: str | None
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
        "question": "Which organisation is responsible for {task} in {city1}?",
        "gold": "{org}",
    },
    "founding": {
        "type": "number_date",
        "question": "In what year was the {org} founded?",
        "gold": "{year1}",
    },
    "relation_a": {
        "type": "relation_pos",
        "question": "Who maintains the {artifact1} array?",
        "gold": "{p1}",
    },
    "relation_b": {
        "type": "relation_pos",
        "question": "Who runs the {artifact2} probe?",
        "gold": "{p3}",
    },
    "negation": {
        "type": "relation_neg",
        "question": "Did {p2} found the {org}?",
        "gold": "No",
    },
    "condition_a": {
        "type": "condition_exception",
        "question": "Under what condition does the {artifact1} array operate?",
        "gold": "only below {thresh} degrees Celsius",
    },
    "condition_b": {
        "type": "condition_exception",
        "question": "What temperature limit applies to the {artifact2} probe?",
        "gold": "below {thresh2} degrees Celsius",
    },
    "temporal": {
        "type": "temporal",
        "question": "In what year did the {org} relocate from {city1} to {city2}?",
        "gold": "{year2}",
    },
    "causal": {
        "type": "causal",
        "question": "Why was the {project} delayed?",
        "gold": "the {council} withheld the second tranche",
    },
    "attribution": {
        "type": "attribution_uncertainty",
        "question": "Who stated that the {project} was uncertain?",
        "gold": "{director}",
    },
    "quantity": {
        "type": "number_date",
        "question": "How many sealed units does the {artifact1} array carry?",
        "gold": "{count}",
    },
    "location": {
        "type": "entity",
        "question": "Where is the observatory operated by the {council}?",
        "gold": "{city2}",
    },
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

# Legacy phrases retained only to audit the two corpora.
LEGACY_PADS = [
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

def bind_roles(rng: random.Random) -> dict[str, str]:
    p1, p2, p3 = (synth_word(rng) for _ in range(3))
    while len({p1, p2, p3}) < 3:
        p1, p2, p3 = (synth_word(rng) for _ in range(3))

    city1, city2 = synth_word(rng), synth_word(rng)
    while city2 == city1:
        city2 = synth_word(rng)

    year1 = rng.randint(2034, 2052)
    year2 = year1 + rng.choice([2, 3, 4, 5])
    return {
        "org_stem": synth_word(rng),
        "director": f"{synth_word(rng)} {synth_word(rng)}",
        "p1": p1,
        "p2": p2,
        "p3": p3,
        "city1": city1,
        "city2": city2,
        "year1": str(year1),
        "year2": str(year2),
        "thresh": str(rng.randint(8, 16)),
        "thresh2": str(rng.randint(8, 16)),
        "count": str(rng.randint(12, 96)),
    }


# --------------------------------------------------------------------------
# 5. Document assembly
# --------------------------------------------------------------------------

def choose_renderers(rng: random.Random, n_facts: int) -> list[str]:
    rest = [r for r in RENDERERS if r not in REQUIRED_RENDERERS]
    need = n_facts - len(REQUIRED_RENDERERS)
    assert 0 <= need <= len(rest), f"{n_facts} facts not supported"
    chosen = REQUIRED_RENDERERS + rng.sample(rest, need)
    rng.shuffle(chosen)
    return chosen


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

def to_jsonl(docs: list[Doc], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with (out / "docs.jsonl").open("w", encoding="utf-8", newline="\n") as fd, (
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



SEED = 20261113
SURFACES = {
    "entity": [
        "In {city1}, responsibility for {task} belongs to the {org}.",
        "Of the work carried out in {city1}, the remit of the {org} is specifically {task}, the work of {task_detail}.",
        "The {org} handles {task} in {city1}, where its remit covers {task_detail}.",
        "For {task} in {city1}, including the work of {task_detail}, the responsible organisation is the {org}.",
        "The work of {task_detail} falls under {task} in {city1}, a responsibility assigned to the {org}.",
        "Within {city1}, it is the {org} that is responsible for {task}, including the work of {task_detail}."
    ],
    "founding": [
        "The {org} was established in {year1}.",
        "The founding of the {org} dates to {year1}.",
        "It was in {year1} that the {org} came into existence as an organisation.",
        "The year {year1} marks the point at which the {org} was established as an organisation in its own right.",
        "For the {org}, {year1} is the year of establishment, the point at which the organisation itself came into being.",
        "The founding date of the organisation known as the {org} falls in the year {year1}.",
        "It was during the year {year1} that the body known as the {org} was founded."
    ],
    "relation_a": [
        "{p1} maintains the {artifact1} array.",
        "Responsibility for keeping the {artifact1} array in working order rests with {p1}.",
        "The person charged with maintaining the {artifact1} array, and thus with keeping the installation in working condition, is {p1}.",
        "For the {artifact1} array, it is {p1} who is responsible for the upkeep of the equipment itself.",
        "Keeping the {artifact1} array serviceable is the responsibility of {p1}, the person assigned to maintain that equipment."
    ],
    "relation_b": [
        "{p3} is in charge of the {artifact2} probe.",
        "Operational control of the {artifact2} probe rests with {p3}.",
        "When it comes to running the {artifact2} probe, the person with responsibility for operating the equipment is {p3}.",
        "The {artifact2} probe is run by {p3}, whose role with respect to that equipment is one of operational control.",
        "The person who directs operation of the {artifact2} probe is {p3}, who is in charge of running that piece of equipment."
    ],
    "negation": [
        "{p1}, not {p2}, founded the {org}.",
        "The founder of the {org} was {p1}; {p2} was not a founder.",
        "Credit for establishing the {org} belongs to {p1}, and the founding role should not be attributed to {p2}.",
        "To identify who established the {org}, the relevant distinction is between {p1}, who founded it, and {p2}, who did not.",
        "The founding of the {org} is attributable to {p1}; {p2} had no part as a founder of the organisation."
    ],
    "condition_a": [
        "The {artifact1} array can operate only below {thresh} degrees Celsius.",
        "For the {artifact1} array, operation requires the surrounding air to be colder than {thresh} degrees Celsius.",
        "The {artifact1} array operates only when the temperature of the surrounding air is below {thresh} degrees Celsius, with the limit itself excluded.",
        "At {thresh} degrees Celsius or higher, the ambient temperature rules out operation of the {artifact1} array; the air must be colder for it to run.",
        "For air temperatures at or above {thresh} degrees Celsius, the operating rule bars use of the {artifact1} array; operation is restricted to temperatures below that boundary."
    ],
    "condition_b": [
        "The operating limit for the {artifact2} probe is strictly below {thresh2} degrees Celsius.",
        "Using the {artifact2} probe requires an ambient temperature lower than {thresh2} degrees Celsius.",
        "Air at {thresh2} degrees Celsius already falls outside the permitted operating range for the {artifact2} probe, which can be used only at lower temperatures.",
        "For the {artifact2} probe to be used, the temperature of the surrounding air must remain below {thresh2} degrees Celsius rather than reaching the limit.",
        "Only if the air is cooler than {thresh2} degrees Celsius can the {artifact2} probe be used, so an air temperature equal to that limit does not qualify."
    ],
    "temporal": [
        "The {org} moved from {city1} to {city2} in {year2}.",
        "In {year2}, the {org} transferred its base of operations from {city1} to {city2}.",
        "The move that took the {org} out of {city1} and established its base of operations in {city2} occurred in {year2}.",
        "For the relocation of the {org}, {city1} was the place of departure and {city2} the destination, with the move taking place in {year2}.",
        "The relocation of the {org} took place in {year2}, when the organisation transferred its operating base to {city2} from its former base in {city1}."
    ],
    "causal": [
        "The {council} withheld the second tranche of funding, delaying the {project}.",
        "The delay to the {project} resulted from the decision by the {council} to withhold the second tranche of funding.",
        "Because the {council} held back the second portion of the funding, the {project} fell behind the schedule on which it had been planned.",
        "What caused the {project} to slip was the withholding of its second instalment of funding by the {council}.",
        "The {project} fell behind its planned schedule because the {council} withheld the second instalment of funding for the work."
    ],
    "attribution": [
        "{director} said it remained uncertain whether the {project}, planned for {year3}, would go ahead.",
        "As {director} described it, {year3} was the planned year for the {project}, but whether the project would proceed was still uncertain.",
        "The {project} was planned for {year3}; according to {director}, there was still uncertainty about whether it would go ahead.",
        "Although {year3} was the year planned for the {project}, {director} described the prospect of the work going ahead as uncertain.",
        "The plans placed the {project} in {year3}, though {director} said there was still doubt about whether the work would proceed."
    ],
    "quantity": [
        "The {artifact1} array contains {count} sealed units.",
        "Counting the sealed units in the {artifact1} array gives a total of {count}.",
        "The number of sealed units housed within the {artifact1} array is {count}, taking the array as a whole for the count.",
        "There are {count} units of the sealed kind inside the {artifact1} array, a count that refers specifically to the sealed units in that installation.",
        "The total for the {artifact1} array is {count} when the count is restricted to the sealed units contained within that installation."
    ],
    "location": [
        "The observatory operated by the {council} is in {city2}.",
        "In {city2} is the observatory run by the {council}.",
        "The {council} operates an observatory whose site is in {city2}, the city where the facility itself is located.",
        "The site in {city2} houses the observatory for which the {council} holds operational responsibility.",
        "To locate the observatory for which the {council} is responsible, the relevant site is in {city2}, where the facility is situated."
    ]
}

TASKS = {
    "institute": [
        ("atmospheric calibration", "aligning instruments that measure the air"),
        ("acoustic calibration", "matching sound instruments to reference signals"),
        ("sediment classification", "sorting sediment samples by their physical characteristics"),
        ("optical calibration", "aligning light measurements with reference values"),
        ("mineral indexing", "organising specimen descriptions into a searchable index"),
        ("signal verification", "checking whether recorded signals match reference patterns"),
        ("soil classification", "grouping soil samples by shared physical properties"),
        ("thermal calibration", "matching temperature readings to known reference values"),
    ],
    "expedition": [
        ("glacial mapping", "plotting the extent and shape of glacier surfaces"),
        ("core sampling", "extracting cylindrical sections for examination"),
        ("traverse logistics", "organising supplies for travel between field stations"),
        ("ridge surveying", "recording the shape and elevation of ridgelines"),
        ("snow profiling", "describing the layers exposed in a snow section"),
        ("meltwater sampling", "collecting specimens from streams fed by melting ice"),
        ("crevasse mapping", "plotting the position and extent of breaks in the ice"),
        ("slope surveying", "measuring the angle and shape of exposed slopes"),
    ],
}
EXPANSIONS = {
    "institute": {
        "artifact": ["spectral", "thermal", "acoustic", "pressure", "optical", "magnetic", "sampling"],
        "project": ["northern annex", "coastal extension", "optics wing", "materials wing", "western annex", "sensor renewal", "reference laboratory", "signal workshop"],
        "council": ["funding council", "oversight board", "charter committee", "research panel", "grants committee", "facilities board"],
    },
    "expedition": {
        "artifact": ["radar", "ice sounding", "altimetry", "acoustic", "snow depth", "pressure", "tracking"],
        "project": ["southern traverse", "winter resupply", "ridge survey", "shelf crossing", "upper basin survey", "western ascent", "ice margin survey", "valley traverse"],
        "council": ["logistics board", "expedition committee", "transport office", "field operations board", "route planning panel", "supply committee"],
    },
}

MULTIHOP_TEMPLATES[1]["question"] = (
    "Which event has the earlier associated year: the relocation to {city2} or the planned {project}?"
)
UNANSWERABLE = [
    "How many staff members does the {org} employ?",
    "What was the operating budget of the {org} in {year1}?",
    "What is the street address of the {org} base in {city2}?",
    "How many vehicles did the {org} use for its move to {city2}?",
]


def make_facts(domain, n_facts, project_first, rng):
    roles = bind_roles(rng)
    roles["org"] = f'{roles.pop("org_stem")} {domain.title()}'
    roles["task"], roles["task_detail"] = rng.choice(TASKS[domain])
    roles["artifact1"], roles["artifact2"] = rng.sample(EXPANSIONS[domain]["artifact"], 2)
    for key in ("project", "council"):
        roles[key] = rng.choice(EXPANSIONS[domain][key])

    year1, year2 = int(roles["year1"]), int(roles["year2"])
    offsets = [-3, -2, -1] if project_first else [1, 2, 3]
    roles["year3"] = str(year2 + rng.choice([d for d in offsets if year2 + d > year1]))
    order = choose_renderers(rng, n_facts)
    facts = [
        Fact(
            fid=f"f{idx + 1}", renderer=rid, type=RENDERERS[rid]["type"],
            sentence_idx=idx, text="", question=RENDERERS[rid]["question"].format(**roles),
            gold=RENDERERS[rid]["gold"].format(**roles),
        )
        for idx, rid in enumerate(order)
    ]
    return roles, facts


def realise(facts, roles, rng):
    candidates = [[s.format(**roles) for s in SURFACES[f.renderer]] for f in facts]
    lengths = [[(len(s.split()), len(s.encode("utf-8"))) for s in pool] for pool in candidates]
    n = len(facts)
    best = None
    for _ in range(420):
        choices = [rng.randrange(len(pool)) for pool in candidates]
        measures = [options[i] for options, i in zip(lengths, choices)]
        words, sizes = zip(*measures)
        total_words = sum(words)
        density = (sum(sizes) + n - 1) / n
        if not 19.8 * n <= total_words <= 24.2 * n or not 125.1 <= density <= 152.9:
            continue
        spread = pstdev(words)
        score = abs(total_words / n - 22) + abs(density - 139) / 15 + max(0, 4.4 - spread)
        if best is None or score < best[0]:
            best = score, choices
    if best is None:
        return None
    return [pool[i] for pool, i in zip(candidates, best[1])]


def read_jsonl(path):
    with path.open(encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream]


def build_extension(seed, existing_docs, existing_questions):
    rng = random.Random(seed)
    existing_ids = {d["doc_id"] for d in existing_docs}
    existing_ids.update(q["doc_id"] for q in existing_questions)
    existing_qids = {q["qid"] for q in existing_questions}
    seen = {d["text"] for d in existing_docs}
    organisations = set()
    positions = defaultdict(Counter)
    rejected = Counter()
    docs = []
    for domain in ("institute", "expedition"):
        schedule = [(8 + i % 3, i % 2 == 0) for i in range(50)]
        rng.shuffle(schedule)
        for i, (n_facts, project_first) in enumerate(schedule, 1):
            doc_id = f"p-ext-{domain}-{i:03}"
            if doc_id in existing_ids or any(
                f"{doc_id}-q{j}" in existing_qids for j in range(1, n_facts + 3)
            ):
                raise ValueError(f"Existing ID collision: {doc_id}")
            for _ in range(160):
                roles, facts = make_facts(domain, n_facts, project_first, rng)
                if len({roles[key] for key in ("p1", "p2", "p3", "city1", "city2")}) != 5:
                    rejected["role_collision"] += 1
                    continue
                if roles["org"] in organisations:
                    rejected["organisation_collision"] += 1
                    continue
                if any(positions[domain, f.renderer][f.sentence_idx] >= 7 for f in facts):
                    rejected["position_cap"] += 1
                    continue
                sentences = realise(facts, roles, rng)
                if sentences is None:
                    rejected["density"] += 1
                    continue
                for fact, sentence in zip(facts, sentences):
                    fact.text = sentence
                doc = Doc(doc_id, None, domain, " ".join(sentences), facts, roles)
                add_questions(doc, rng)
                errors = validate(doc)
                if errors or doc.text in seen:
                    rejected.update([e.split("=")[0] for e in errors] or ["duplicate_text"])
                    continue
                docs.append(doc)
                seen.add(doc.text)
                organisations.add(roles["org"])
                for f in facts:
                    positions[domain, f.renderer][f.sentence_idx] += 1
                break
            else:
                raise RuntimeError(f"Could not render {doc_id}: {dict(rejected)}")
    return docs, rejected


def verify_export(out, docs, old_docs, old_questions):
    exported = read_jsonl(out / "docs.jsonl")
    questions = read_jsonl(out / "questions.jsonl")
    assert len(exported) == 100
    assert Counter(d["domain"] for d in exported) == {"institute": 50, "expedition": 50}
    assert len({d["doc_id"] for d in exported}) == 100
    assert len({d["text"] for d in exported}) == 100
    assert len({q["qid"] for q in questions}) == len(questions)
    assert not {d["doc_id"] for d in exported} & {d["doc_id"] for d in old_docs}
    assert not {q["qid"] for q in questions} & {q["qid"] for q in old_questions}
    assert not {d["text"] for d in exported} & {d["text"] for d in old_docs}
    by_id = defaultdict(list)
    for q in questions:
        assert q.keys() == old_questions[0].keys()
        by_id[q["doc_id"]].append(q)
    assert set(by_id) == {d["doc_id"] for d in exported}
    for row, doc in zip(exported, docs):
        assert row.keys() == old_docs[0].keys()
        assert row["doc_id"] == doc.doc_id and row["split"] is None
        facts = row["facts"]
        assert row["text"] == " ".join(f["text"] for f in facts) == doc.text
        assert len(facts) == row["n_facts"] == doc.n_facts
        assert len(row["text"].split()) == row["words"] == doc.words
        assert len(row["text"].encode("utf-8")) == row["bytes"] == doc.nbytes
        assert row["bytes_per_fact"] == round(doc.bytes_per_fact, 1)
        assert [f["sentence_idx"] for f in facts] == list(range(len(facts)))
        assert len({f["fid"] for f in facts}) == len(facts)
        assert {f["renderer"] for f in facts} >= set(REQUIRED_RENDERERS)
        assert not validate(doc)
        for f in facts:
            assert f.keys() == old_docs[0]["facts"][0].keys()
        linked = {f["fid"]: f for f in facts}
        qs = by_id[row["doc_id"]]
        assert qs == doc.questions
        assert len({q["text"] for q in qs}) == len(qs)
        assert sum(not q["answerable"] for q in qs) == 1
        lookups = set()
        for q in qs:
            evidence = [linked[fid] for fid in q["fact_ids"]]
            assert q["fact_types"] == [f["type"] for f in evidence]
            if not q["answerable"]:
                assert q["gold"] == "unknown" and not evidence
            elif q["demand"] == "lookup":
                assert len(evidence) == 1 and evidence[0]["gold"] == q["gold"]
                lookups.add(evidence[0]["renderer"])
            else:
                assert len(evidence) == 2 and evidence[0]["sentence_idx"] != evidence[1]["sentence_idx"]
                # Read the operands from the exported supporting sentences.
                years = [int(re.search(r"\b20\d{2}\b", f["text"])[0]) for f in evidence]
                if evidence[0]["renderer"] == "founding":
                    assert evidence[1]["renderer"] == "temporal"
                    assert q["gold"] == str(years[1] - years[0])
                else:
                    assert [f["renderer"] for f in evidence] == ["temporal", "attribution"]
                    assert years[0] != years[1] and str(years[1]) not in q["text"]
                    assert q["gold"] == ("the relocation" if years[0] < years[1] else f'the {doc.roles["project"]}')
        assert len(lookups) == len(facts) - 1 and lookups >= set(REQUIRED_RENDERERS)


def corpus_stats(rows):
    positions = defaultdict(Counter)
    gram_docs = Counter()
    words = Counter()
    shapes = set()
    spreads = []
    for row in rows:
        tokens = re.findall(r"[a-z]+|\d+", row["text"].lower())
        words.update(tokens)
        gram_docs.update(set(zip(*(tokens[i:] for i in range(6)))))
        spreads.append(pstdev(len(f["text"].split()) for f in row["facts"]))
        for f in row["facts"]:
            positions[f["renderer"]][f["sentence_idx"]] += 1
            shapes.add(re.sub(r"\b(?:[A-Z][a-z]+|\d+)\b", "#", f["text"]))
    return {
        "documents": len(rows),
        "facts": sum(row["n_facts"] for row in rows),
        "fact_counts": dict(sorted(Counter(row["n_facts"] for row in rows).items())),
        "words": [min(row["words"] for row in rows), max(row["words"] for row in rows)],
        "bytes_per_fact": [min(row["bytes_per_fact"] for row in rows), max(row["bytes_per_fact"] for row in rows)],
        "average_within_document_sentence_length_sd": round(mean(spreads), 3),
        "legacy_pad_matches": sum(any(pad in f["text"] for pad in LEGACY_PADS) for row in rows for f in row["facts"]),
        "vocabulary": len(words),
        "tokens": sum(words.values()),
        "type_token_ratio": round(len(words) / sum(words.values()), 4),
        "approximate_sentence_skeletons": len(shapes),
        "distinct_6gram_share_seen_in_multiple_documents": round(sum(v > 1 for v in gram_docs.values()) / len(gram_docs), 4),
        "positions": {
            rid: {
                "counts_zero_based": dict(sorted(hist.items())),
                "mode_share": round(max(hist.values()) / hist.total(), 4),
            }
            for rid, hist in sorted(positions.items())
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Build the independent 100-document extension.")
    parser.add_argument("--source", type=Path, required=True, help="Directory with the existing docs.jsonl and questions.jsonl")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if not __debug__:
        parser.error("Run with assertions enabled for export validation.")
    if args.out.resolve() == args.source.resolve():
        parser.error("Use a separate output directory.")
    old_docs = read_jsonl(args.source / "docs.jsonl")
    old_questions = read_jsonl(args.source / "questions.jsonl")
    docs, rejected = build_extension(args.seed, old_docs, old_questions)
    to_jsonl(docs, args.out)
    verify_export(args.out, docs, old_docs, old_questions)
    rows = read_jsonl(args.out / "docs.jsonl")
    report = corpus_stats(rows)
    report.update(
        seed=args.seed,
        old_documents_checked=len(old_docs),
        old_questions_checked=len(old_questions),
        attempts=len(docs) + rejected.total(),
        rejections=dict(rejected),
        questions=sum(len(d.questions) for d in docs),
        lookup=sum(d.n_facts - 1 for d in docs),
        multi_hop=2 * len(docs),
        unanswerable=len(docs),
        ordinal_gold=dict(Counter(
            "project" if int(d.roles["year3"]) < int(d.roles["year2"]) else "relocation" for d in docs
        )),
        fact_types=dict(Counter(f.type for d in docs for f in d.facts)),
        domains={domain: corpus_stats([r for r in rows if r["domain"] == domain]) for domain in ("institute", "expedition")},
        baseline_same_domains=corpus_stats([r for r in old_docs if r["domain"] in TASKS]),
        sha256={name: hashlib.sha256((args.out / name).read_bytes()).hexdigest() for name in ("docs.jsonl", "questions.jsonl")},
    )
    for stats in (report, *report["domains"].values()):
        assert all(len(p["counts_zero_based"]) >= 4 and p["mode_share"] < 0.4 for p in stats["positions"].values())
    (args.out / "statistics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Verified {len(docs)} documents and {report['questions']} questions.")


if __name__ == "__main__":
    main()
