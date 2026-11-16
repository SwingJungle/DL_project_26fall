"""Primary-set generator v1.1 - facts first, text rendered from the fact table.

What changed from v1 (see _previous/generate_v1_2026-11-12.py):

  1. Sentence frameworks moved out to `frameworks.py`. This file is logic only.
  2. Sentence length is no longer a fixed 22-word quota. A length vector is
     drawn per document (mean 22, spread >= 4) and the closest template is
     used. v1's fixed quota made every short template unreachable, which is
     why `founding` collapsed to three surface forms in 200 documents.
  3. Order is a full shuffle. v1 kept entity/founding/negation pinned to
     positions 0/1/2 in 100% of documents, which handed the prefix-truncation
     baseline about 56% of the facts for free (spec 13.2, P1).
  4. Padding phrases are gone. v1 attached a hedge to 66% of sentences and
     reused one phrase in 196/200 documents; ~19% of source bytes could be
     dropped with no fact loss (spec 13.2, P2). Length now comes from the
     templates themselves.
  5. Templates are chosen least-used-first within a length band, so the pool
     is used evenly instead of concentrating on whichever form happens to sit
     nearest the quota.
  6. Corpus-level gates: slot position concentration, skeleton count, within-
     document length spread, 6-gram document frequency.

CLI:
    python generate.py --n 200 --seed 20261111 --out data
    python generate.py --n 1 --dry-run
    python generate.py --lint
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import re
from pathlib import Path

try:
    import frameworks as F
except ImportError:  # pragma: no cover - package import
    from . import frameworks as F


def load_frameworks(path: str):
    """Load an alternative template module, so two authors can share the
    generator without sharing a sentence pool."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("frameworks_custom", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# --------------------------------------------------------------------------
# 1. Lexicon
# --------------------------------------------------------------------------

ONSETS = ["b", "d", "f", "g", "k", "l", "m", "n", "p", "r", "s", "t", "v", "z"]
VOWELS = ["a", "e", "i", "o", "u", "ae", "ia", "eo"]
CODAS = ["l", "r", "n", "m", "s", "th", "nd", ""]

REAL_NAME_BLACKLIST = {
    "alan", "mira", "john", "mary", "david", "sarah", "michael", "anna",
    "london", "paris", "tokyo", "beijing", "berlin", "york",
    "google", "openai", "anthropic", "deepseek", "alibaba", "microsoft",
    "stanford", "oxford", "cambridge", "harvard",
}

ORG_SUFFIX = ["Institute", "Survey", "Consortium", "Foundation", "Bureau", "Guild"]

DOMAINS: dict[str, dict[str, list[str]]] = {
    "institute": {
        "task": ["atmospheric calibration", "signal verification", "archive indexing"],
        "artifact": ["sensor", "telemetry", "reference"],
        "project": ["northern annex", "coastal extension", "second survey"],
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

#: org suffix that fits each domain, so the label and the name agree
DOMAIN_SUFFIX = {
    "institute": ["Institute", "Survey"],
    "expedition": ["Survey", "Foundation"],
    "consortium": ["Consortium", "Guild"],
    "archive": ["Bureau", "Foundation"],
}


class Names:
    """Invented stems, drawn without collision inside one build."""

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        self.used: set[str] = set()

    def word(self, syllables: int = 2) -> str:
        for _ in range(40):
            out = []
            for _ in range(syllables):
                out.append(
                    self.rng.choice(ONSETS)
                    + self.rng.choice(VOWELS)
                    + self.rng.choice(CODAS)
                )
            w = "".join(out).capitalize()
            if not 4 <= len(w) <= 9:
                continue
            if any(b in w.lower() for b in REAL_NAME_BLACKLIST):
                continue
            if w in self.used:
                continue
            self.used.add(w)
            return w
        w = "".join(
            self.rng.choice(ONSETS) + self.rng.choice(VOWELS) for _ in range(syllables)
        ).capitalize()
        self.used.add(w)
        return w


# --------------------------------------------------------------------------
# 2. Role binding
# --------------------------------------------------------------------------

def bind_roles(domain: str, rng: random.Random, names: Names) -> dict[str, str]:
    pool = DOMAINS[domain]
    org = f"{names.word()} {rng.choice(DOMAIN_SUFFIX[domain])}"

    p1, p2, p3 = names.word(), names.word(), names.word()
    city1, city2 = names.word(), names.word()
    artifact1, artifact2 = rng.sample(pool["artifact"], 2)

    year1 = rng.randint(2034, 2052)
    year2 = year1 + rng.choice([2, 3, 4, 5])
    # the project year straddles the relocation so the ordinal multi-hop is not
    # guessable; resample rather than clamp, otherwise the split skews
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
        "director": f"{names.word()} {names.word()}",
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
# 3. Sentence selection
# --------------------------------------------------------------------------

WORDS_PER_FACT = 22
#: the spec's design point is 139 bytes per fact (22 words at ~6.3 B/word).
#: The selector aims slightly below it: v1 actually produced 133.9, and 133
#: leaves the pool enough headroom to rotate templates instead of always
#: taking the longest one. Both values sit inside the 139 +/-10% band.
TARGET_BPF = 139.0
SELECT_BPF = 133.0
B_LO, B_HI = 45, 250
B_SD = 40.0
#: how many bytes of length error one full 100% of usage share is worth
SHARE_PENALTY = 200.0
USE_PENALTY = 1.6


def choose_slots(rng: random.Random, n_facts: int) -> list[str]:
    rest = list(F.OPTIONAL_SLOTS)
    need = n_facts - len(F.REQUIRED_SLOTS)
    if not 0 <= need <= len(rest):
        raise ValueError(f"{n_facts} facts not supported")
    return F.REQUIRED_SLOTS + rng.sample(rest, need)


def byte_targets(rng: random.Random, n: int) -> list[float]:
    """Per-sentence byte budget summing to 139 bytes per fact.

    The compression budget is defined in UTF-8 bytes, so bytes are what the
    selector has to hit. Word count follows from the template's density and is
    only checked as a broad sanity band afterwards.
    """
    total = SELECT_BPF * n
    ts = [min(B_HI, max(B_LO, rng.gauss(SELECT_BPF, B_SD))) for _ in range(n)]
    scale = total / sum(ts)
    return [min(B_HI, max(B_LO, t * scale)) for t in ts]


def pick_texts(
    slots: list[str],
    roles: dict[str, str],
    rng: random.Random,
    usage: dict[str, collections.Counter],
) -> list[int]:
    """Return, for each slot, the index of the template used."""
    n = len(slots)
    targets = byte_targets(rng, n)
    rendered = [[t.format(**roles) for t in F.TEMPLATES[s]] for s in slots]
    sizes = [[len(t.encode("utf-8")) for t in pool] for pool in rendered]

    chosen: list[int] = []
    for slot, pool, target in zip(slots, sizes, targets):
        # cost = distance from the length target, plus a charge for every
        # earlier use of the same template. Without the second term the
        # longest template in a slot is chosen almost every time, because it
        # is the only one that reaches the byte target.
        # penalty on the *share* of uses, Laplace-smoothed, so it works the
        # same at document 3 and at document 200. Penalising the raw count
        # leaves the first few documents free to concentrate on one template,
        # and that skew never washes out in a 100-document half.
        total_uses = sum(usage[slot].values())
        base = total_uses + len(pool)
        best = min(
            range(len(pool)),
            key=lambda i: (
                abs(pool[i] - target) + SHARE_PENALTY * (usage[slot][i] + 1) / base,
                rng.random(),
            ),
        )
        chosen.append(best)

    # greedy repair so the document lands on 139 bytes per fact
    total_all = SELECT_BPF * n
    for _ in range(40):
        total = sum(pool[i] for pool, i in zip(sizes, chosen))
        err = total - total_all
        if abs(err) <= 0.025 * total_all:
            break
        best = None
        for k, (slot, pool, cur) in enumerate(zip(slots, sizes, chosen)):
            for alt, alt_size in enumerate(pool):
                if alt == cur:
                    continue
                gain = abs(err) - abs(total - pool[cur] + alt_size - total_all)
                # the repair must not undo the rotation: a template that is
                # already over-used has to be much better on length to win
                gain -= USE_PENALTY * (usage[slot][alt] - usage[slot][cur])
                if gain > 0 and (best is None or gain > best[0]):
                    best = (gain, k, alt)
        if best is None:
            break
        chosen[best[1]] = best[2]

    for slot, i in zip(slots, chosen):
        usage[slot][i] += 1
    return chosen


# --------------------------------------------------------------------------
# 4. Document assembly
# --------------------------------------------------------------------------

def add_questions(doc: dict, rng: random.Random) -> None:
    facts = doc["facts"]
    optional = [f for f in facts if f["renderer"] not in F.REQUIRED_SLOTS]
    drop = rng.choice(optional)["fid"] if optional else facts[-1]["fid"]

    qs: list[dict] = []
    for f in facts:
        if f["fid"] == drop:
            continue
        qs.append(
            {
                "demand": "lookup",
                "text": F.QUESTION[f["renderer"]].format(**doc["roles"]),
                "gold": F.GOLD[f["renderer"]].format(**doc["roles"]),
                "answerable": True,
                "fact_ids": [f["fid"]],
                "fact_types": [f["type"]],
            }
        )

    by_slot = {f["renderer"]: f for f in facts}
    for spec in F.MULTIHOP:
        a, b = spec["needs"]
        if a not in by_slot or b not in by_slot:
            continue
        fa, fb = by_slot[a], by_slot[b]
        if fa["sentence_idx"] == fb["sentence_idx"]:
            continue
        roles = doc["roles"]
        if spec["compute"] == "year_diff":
            gold = str(int(roles["year2"]) - int(roles["year1"]))
        else:
            gold = (
                "the relocation"
                if int(roles["year2"]) < int(roles["year3"])
                else f"the {roles['project']}"
            )
        qs.append(
            {
                "demand": "multi-hop",
                "text": spec["question"].format(**roles),
                "gold": gold,
                "answerable": True,
                "fact_ids": [fa["fid"], fb["fid"]],
                "fact_types": [fa["type"], fb["type"]],
            }
        )

    qs.append(
        {
            "demand": "lookup",
            "text": rng.choice(F.UNANSWERABLE).format(**doc["roles"]),
            "gold": "unknown",
            "answerable": False,
            "fact_ids": [],
            "fact_types": [],
        }
    )

    rng.shuffle(qs)
    for i, q in enumerate(qs, start=1):
        q["qid"] = f"{doc['doc_id']}-q{i}"
        q["doc_id"] = doc["doc_id"]
    doc["questions"] = qs


def render_doc(
    domain: str, n_facts: int, rng: random.Random, names: Names,
    usage: dict[str, collections.Counter],
) -> dict:
    roles = bind_roles(domain, rng, names)
    slots = choose_slots(rng, n_facts)
    rng.shuffle(slots)
    if slots[0] not in ORG_SLOTS:
        swappable = [i for i, sl in enumerate(slots) if sl in ORG_SLOTS]
        if swappable:
            j = rng.choice(swappable)
            slots[0], slots[j] = slots[j], slots[0]
    idx = pick_texts(slots, roles, rng, usage)

    facts = []
    for pos, (slot, i) in enumerate(zip(slots, idx)):
        facts.append(
            {
                "fid": f"f{pos + 1}",
                "renderer": slot,
                "type": F.SLOT_TYPE[slot],
                "sentence_idx": pos,
                "text": F.TEMPLATES[slot][i].format(**roles),
                "gold": F.GOLD[slot].format(**roles),
            }
        )

    text = " ".join(f["text"] for f in facts)
    return {
        "doc_id": "",
        "split": "",
        "domain": domain,
        "text": text,
        "words": len(text.split()),
        "bytes": len(text.encode("utf-8")),
        "n_facts": len(facts),
        "bytes_per_fact": round(len(text.encode("utf-8")) / len(facts), 1),
        "facts": facts,
        "roles": roles,
    }


# --------------------------------------------------------------------------
# 5. Validation
# --------------------------------------------------------------------------

PRONOUN_OPEN = ("its ", "it ", "this ", "these ", "they ")

#: slots whose every template names the organisation. One of these opens the
#: document, otherwise a fully shuffled doc starts mid-thought ("Only below 13
#: degrees Celsius does the humidity unit operate ...") before the reader knows
#: what the document is about. Four slots rotate, so position is still varied.
ORG_SLOTS = ("entity", "founding", "temporal", "negation")


def validate(doc: dict) -> list[str]:
    p: list[str] = []
    n = doc["n_facts"]

    if not 8 <= n <= 10:
        p.append(f"n_facts={n}")

    target_words = WORDS_PER_FACT * n
    if not 0.9 * target_words <= doc["words"] <= 1.1 * target_words:
        p.append(f"words={doc['words']}/{target_words}")

    if not 0.9 * TARGET_BPF <= doc["bytes_per_fact"] <= 1.1 * TARGET_BPF:
        p.append(f"bytes_per_fact={doc['bytes_per_fact']:.1f}")

    types = {f["type"] for f in doc["facts"]}
    if len(types) < 5:
        p.append(f"type_coverage={len(types)}")
    for req in ("relation_neg", "condition_exception", "attribution_uncertainty"):
        if req not in types:
            p.append(f"missing_type={req}")
    slots = {f["renderer"] for f in doc["facts"]}
    for req in F.REQUIRED_SLOTS:
        if req not in slots:
            p.append(f"missing_slot={req}")
    if len(slots) != n:
        p.append("duplicate_slot")

    mh = [q for q in doc["questions"] if q["demand"] == "multi-hop"]
    if len(mh) != 2:
        p.append(f"multi_hop={len(mh)}")
    pos = {f["fid"]: f["sentence_idx"] for f in doc["facts"]}
    for q in mh:
        a, b = q["fact_ids"]
        if pos.get(a) == pos.get(b):
            p.append("multi_hop_same_sentence")

    if len(doc["questions"]) != n + 2:
        p.append(f"n_questions={len(doc['questions'])}")

    unans = sum(1 for q in doc["questions"] if not q["answerable"])
    if not 0.06 <= unans / len(doc["questions"]) <= 0.15:
        p.append(f"unanswerable={unans}")

    lowered = doc["text"].lower()
    for bad in REAL_NAME_BLACKLIST:
        if re.search(rf"\b{re.escape(bad)}\b", lowered):
            p.append(f"real_name_leak={bad}")

    for f in doc["facts"]:
        if f["text"].lower().startswith(PRONOUN_OPEN):
            p.append(f"pronoun_open={f['renderer']}")
        if f["type"] in ("entity", "number_date", "relation_pos", "temporal",
                         "attribution_uncertainty"):
            if f["gold"].lower() not in f["text"].lower():
                p.append(f"gold_not_in_sentence={f['renderer']}")

    return p


# --------------------------------------------------------------------------
# 6. Driver
# --------------------------------------------------------------------------

SPLIT_SIZES = (("pilot", 25), ("dev", 30), ("eval", 145))


def assign_splits(n: int) -> list[str]:
    out: list[str] = []
    for name, size in SPLIT_SIZES:
        out.extend([name] * min(size, max(0, n - len(out))))
    while len(out) < n:
        out.append("eval")
    return out[:n]


USAGE: dict = {}


def build(n: int, seed: int, max_retry: int = 80,
          domains_wanted: list[str] | None = None,
          id_start: int = 0,
          split_plan: dict[str, int] | None = None) -> tuple[list[dict], dict]:
    rng = random.Random(seed)
    names = Names(rng)
    usage: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)

    domains_wanted = domains_wanted or list(DOMAINS)
    split_plan = split_plan or dict(SPLIT_SIZES)
    if sum(split_plan.values()) != n:
        raise SystemExit(f"split plan sums to {sum(split_plan.values())}, expected {n}")
    if n % len(domains_wanted):
        raise SystemExit(f"{n} documents do not divide evenly across {len(domains_wanted)} domains")

    # cross split x domain so every cell is filled, then shuffle inside
    pairs: list[tuple[str, str]] = []
    offset = 0
    for split_name in ("pilot", "dev", "eval"):
        count = split_plan.get(split_name, 0)
        if not count:
            continue
        base, extra = divmod(count, len(domains_wanted))
        for k in range(len(domains_wanted)):
            # rotate who receives the remainder, so totals stay even
            dom = domains_wanted[(k + offset) % len(domains_wanted)]
            pairs.extend([(split_name, dom)] * (base + (1 if k < extra else 0)))
        offset += extra
    rng.shuffle(pairs)
    splits = [p[0] for p in pairs]
    domains = [p[1] for p in pairs]
    # balanced 8/9/10 rather than drawn independently
    nf_pool = ([8] * (n // 3 + (n % 3 > 0)) + [9] * (n // 3 + (n % 3 > 1))
               + [10] * (n // 3))
    rng.shuffle(nf_pool)

    docs: list[dict] = []
    seen: set[str] = set()
    stats = {"attempts": 0, "rejected": 0, "reasons": collections.Counter()}

    for i, (split, domain) in enumerate(zip(splits, domains)):
        for _ in range(max_retry):
            stats["attempts"] += 1
            n_facts = nf_pool[i]
            doc = render_doc(domain, n_facts, rng, names, usage)
            doc["doc_id"] = f"p-{id_start + i:04d}"
            doc["split"] = split
            add_questions(doc, rng)
            problems = validate(doc)
            digest = hashlib.sha256(doc["text"].encode()).hexdigest()
            if problems or digest in seen:
                stats["rejected"] += 1
                for msg in problems:
                    stats["reasons"][msg.split("=")[0]] += 1
                continue
            seen.add(digest)
            docs.append(doc)
            break
        else:
            raise RuntimeError(f"doc {i}: no valid sample in {max_retry} attempts")
    USAGE.clear()
    USAGE.update(usage)
    return docs, stats


def to_jsonl(docs: list[dict], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with (out / "docs.jsonl").open("w", encoding="utf-8") as fd, (
        out / "questions.jsonl"
    ).open("w", encoding="utf-8") as fq:
        for d in docs:
            payload = {k: v for k, v in d.items() if k not in ("questions", "roles")}
            fd.write(json.dumps(payload, ensure_ascii=False) + "\n")
            for q in d["questions"]:
                fq.write(json.dumps(q, ensure_ascii=False) + "\n")


def skeleton(sentence: str) -> str:
    toks = sentence.split()
    out = []
    for i, t in enumerate(toks):
        if i > 0 and re.fullmatch(r"[A-Z][A-Za-z]+", t):
            out.append("X")
        else:
            out.append(re.sub(r"\d+", "#", t))
    return " ".join(out)


def corpus_report(docs: list[dict], usage: dict | None = None) -> bool:
    """Spec 13.4 gates. Returns True when every gate passes."""
    sents = [f["text"] for d in docs for f in d["facts"]]
    pos: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    sds: list[float] = []
    for d in docs:
        for f in d["facts"]:
            pos[f["renderer"]][f["sentence_idx"]] += 1
        lens = [len(f["text"].split()) for f in d["facts"]]
        if len(lens) > 1:
            mean = sum(lens) / len(lens)
            sds.append((sum((x - mean) ** 2 for x in lens) / len(lens)) ** 0.5)

    sk = collections.Counter(skeleton(s) for s in sents)
    top5 = sum(c for _, c in sk.most_common(5)) / max(1, len(sents))

    df: collections.Counter = collections.Counter()
    for d in docs:
        toks = re.findall(r"[a-z']+", d["text"].lower())
        df.update({" ".join(toks[i:i + 6]) for i in range(max(0, len(toks) - 5))})
    shared = sum(1 for v in df.values() if v >= 2) / max(1, len(df))

    ok = True
    print(f"sentences            : {len(sents)}")
    print(f"distinct skeletons   : {len(sk)}")
    print(f"top-5 skeleton share : {top5:6.1%}   (gate <= 20%)")
    ok &= top5 <= 0.20
    mean_sd = sum(sds) / len(sds)
    print(f"within-doc length sd : {mean_sd:6.2f}   (gate >= 3.5)")
    ok &= mean_sd >= 3.5
    # advisory: this tracks how much template text is still verbatim across
    # documents, i.e. whether the pool is large enough. It does not invalidate
    # the data on its own, v1 sat at 14.6% under the same measure.
    print(f"6-grams shared by >=2 docs: {shared:6.1%}   (advisory, v1 14.6%)")

    if usage:
        worst = max(
            (max(c.values()) / sum(c.values()), slot) for slot, c in usage.items() if sum(c.values())
        )
        print(f"most used single template : {worst[0]:6.1%} ({worst[1]})   (gate <= 42%)")
        ok &= worst[0] <= 0.42

    print("slot position concentration (gate: dominant share <= 45%):")
    for slot, c in sorted(pos.items(), key=lambda kv: -sum(kv[1].values())):
        tot = sum(c.values())
        idx, hit = c.most_common(1)[0]
        flag = "ok " if hit / tot <= 0.45 else "!! "
        ok &= hit / tot <= 0.45
        print(f"  {flag}{slot:<12} n={tot:4d}  dominant idx={idx:2d} {hit / tot:5.1%}  span={min(c)}..{max(c)}")
    return ok


def summarize(docs: list[dict], stats: dict) -> None:
    n = len(docs)
    words = [d["words"] for d in docs]
    bpf = [d["bytes_per_fact"] for d in docs]
    print(f"documents    : {n}")
    print(f"attempts     : {stats['attempts']}   rejected: {stats['rejected']}"
          f" ({stats['rejected'] / max(1, stats['attempts']):.1%})")
    if stats["reasons"]:
        print("  reasons    : " + ", ".join(f"{k}={v}" for k, v in stats["reasons"].most_common(6)))
    print(f"words        : min {min(words)}  mean {sum(words) / n:.1f}  max {max(words)}")
    print(f"bytes/fact   : min {min(bpf):.0f}  mean {sum(bpf) / n:.1f}  max {max(bpf):.0f}")
    tc = collections.Counter(f["type"] for d in docs for f in d["facts"])
    print("fact types   : " + ", ".join(f"{k}={v}" for k, v in sorted(tc.items())))
    print(f"domain mix   : {dict(collections.Counter(d['domain'] for d in docs))}")
    print(f"n_facts mix  : {dict(sorted(collections.Counter(d['n_facts'] for d in docs).items()))}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20261111)
    ap.add_argument("--out", type=str, default="data")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--lint", action="store_true", help="check frameworks.py only")
    ap.add_argument("--domains", type=str, default=None,
                    help="comma list, e.g. consortium,archive")
    ap.add_argument("--id-start", type=int, default=0,
                    help="first doc number, so two authors do not collide")
    ap.add_argument("--splits", type=str, default=None,
                    help="pilot=12,dev=15,eval=73")
    ap.add_argument("--frameworks", type=str, default=None,
                    help="path to an alternative template module")
    args = ap.parse_args()

    global F
    if args.frameworks:
        F = load_frameworks(args.frameworks)

    if args.lint:
        return 1 if F.lint() else 0

    bad = F.lint(verbose=False)
    if bad:
        print(f"frameworks.py has {len(bad)} problem(s):")
        for b in bad:
            print(f"  !! {b}")
        return 2

    domains_wanted = [d.strip() for d in args.domains.split(",")] if args.domains else None
    if domains_wanted:
        unknown = [d for d in domains_wanted if d not in DOMAINS]
        if unknown:
            raise SystemExit(f"unknown domain(s): {unknown}; choose from {list(DOMAINS)}")
    split_plan = None
    if args.splits:
        split_plan = {}
        for part in args.splits.split(","):
            k, v = part.split("=")
            split_plan[k.strip()] = int(v)

    docs, stats = build(args.n, args.seed, domains_wanted=domains_wanted,
                        id_start=args.id_start, split_plan=split_plan)

    if args.dry_run:
        d = docs[0]
        print(f"--- {d['doc_id']} [{d['split']}] {d['domain']}  "
              f"words={d['words']}  bpf={d['bytes_per_fact']:.0f}\n")
        print(d["text"] + "\n")
        for q in d["questions"]:
            print(f"  [{q['demand']:9s}] {q['text']}\n              -> {q['gold']}")
        return 0

    to_jsonl(docs, Path(args.out))
    summarize(docs, stats)
    print()
    gates = corpus_report(docs, USAGE)
    print(f"\nwrote {args.out}/docs.jsonl and {args.out}/questions.jsonl")
    print("CORPUS GATES:", "PASS" if gates else "FAIL")
    return 0 if gates else 1


if __name__ == "__main__":
    raise SystemExit(main())
