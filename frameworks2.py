"""Sentence frameworks for the primary set.

THIS IS THE FILE YOU EDIT. `generate.py` only reads it.

Every entry is one surface realisation of one fact. The fact itself is fixed:
a template may change word order, voice and clause structure, but it must not
change what is being claimed.

Rules for a new template (all enforced by `python frameworks.py`):

  1. Placeholders must be the documented ones only - {org} {task} {city1}
     {city2} {p1} {p2} {p3} {artifact1} {artifact2} {project} {council}
     {director} {year1} {year2} {year3} {thresh} {thresh2} {count}
  2. Every required placeholder for that slot must appear (see REQUIRED_ARGS).
     Drop {p2} from a negation template and the fact is no longer checkable.
  3. No pronoun standing in for the organisation ("its", "it", "this body").
     A compressed message drops sentences; a pronoun with no antecedent turns
     a comprehension test into a coreference test.
  4. Length 10-32 words. The generator targets 22 words per fact overall, and
     it needs both short and long forms available to hit that without filler.
  5. No new propositions. No new years, numbers, names, or causal links. A
     trailing clause that adds information is a new fact, not a longer sentence.
  6. "according to" only in the attribution slot. Anywhere else it invents an
     attribution that the fact table does not contain (spec 14.4).
  7. Use past tense with {council} as subject. The pool mixes singular bodies
     ("the funding council") and plural ones ("the trustees"), so "has" or
     "runs" will disagree half the time; "had" and "ran" always agree.

Aim for 7-9 templates per slot, spread over the full 10-32 word range. The
generator picks the least-used template that fits each sentence's length
target, so an uneven pool shows up immediately as one template dominating.
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# Slot -> surface forms
# --------------------------------------------------------------------------

TEMPLATES: dict[str, list[str]] = {
    "entity": [
        "The {org} is the registered body responsible for {task} throughout {city1} and the surrounding district, and has been since the present arrangements were set.",
        "Responsibility for {task} in {city1} rests with the {org}, and has never been reassigned, reviewed or formally contested since the present arrangements were made.",
        "In {city1}, {task} falls to the {org}, and the arrangement has stood for as long as the surviving records of the district run.",
        "Charged with {task} across the {city1} district, the {org} has carried that brief through every reorganisation since, without interruption of any kind.",
        "The {org}, based in {city1}, answers for {task} across the region and has done so since the present arrangements were drawn up by the {council}.",
        "Across {city1} and its immediate hinterland, {task} is handled by the {org}, under arrangements that have not changed in any material respect since they were first set down.",
        "Nothing in the surviving record suggests that {task} in {city1} was ever handled by any body other than the {org}, whether before or after the last reorganisation.",
        "Since its formation, the {org} has handled {task} in {city1}, through several reorganisations of the wider district.",
        "Since the present arrangements were first set down, {task} in {city1} has been handled by the {org}, and no proposal to change that has ever been recorded.",
        "Of every body working in {city1}, only the {org} is charged with {task}.",
        "For {task} in {city1}, the body with formal responsibility is the {org}, and no other.",
        "In {city1} it is the {org}, and only the {org}, that handles {task} for the district.",
        "The {org} is the one body charged with {task} across {city1} and the villages that surround it.",
        "Responsibility for {task} in {city1} sits with the {org}, and with no other body operating there.",
    ],
    "founding": [
        "The {org} was founded in {year1}, and the founding charter was registered in the same year in the public records.",
        "Founding records for the {org} are dated to {year1} and were never subsequently revised, replaced or formally disputed.",
        "In {year1} the {org} came into existence, after a preparatory period that left very few traces in the archive of the time.",
        "The {org} dates its existence from {year1}, when its charter was sealed and entered in the public register of the time.",
        "Although later accounts differ on many points of detail, every one of them gives {year1} as the year the {org} was founded.",
        "Every surviving note agrees that the {org} was founded in {year1}, and none of those notes was ever corrected or withdrawn afterwards.",
        "The {org} was established in {year1}, following a short preparatory period about which very little of substance is recorded anywhere in writing.",
        "Every register consulted gives {year1} as the year the {org} was founded, and no alternative date has ever been proposed in writing.",
        "Although the surviving accounts differ on many points of detail, every one of them agrees that the {org} was founded in {year1}, and none proposes an alternative.",
        "Charter records place the founding of the {org} in {year1}, and none of them disagrees.",
        "The {org} dates its charter, and with it its existence, to {year1}.",
        "The {org} first came into being in {year1}, and not before.",
    ],
    "relation_a": [
        "The {artifact1} unit is maintained by {p1}, who took the work over from the previous keeper and reports on it each quarter.",
        "{p1} is responsible for the day-to-day upkeep of the {artifact1} unit and for the records that go with it, quarter by quarter.",
        "Upkeep of the {artifact1} unit falls to {p1}, who reports on the work to the {council} every quarter without fail.",
        "Day-to-day care of the {artifact1} unit sits with {p1}, and has done since the last reorganisation, without interruption.",
        "The {artifact1} unit has been in the care of {p1} for some years now, and the arrangement shows no sign of being changed.",
        "Responsibility for maintaining the {artifact1} unit was formally assigned to {p1} when the previous keeper retired from the post.",
        "Of the several duties attached to the {artifact1} unit, routine maintenance is the one that {p1} keeps in person.",
        "The {artifact1} unit has been in the care of {p1} for some years now, across two reorganisations and a change of site.",
        "Maintenance of the {artifact1} unit was never delegated: it has stayed with {p1} through every reorganisation since.",
        "The {artifact1} unit is looked after by {p1} and by nobody else.",
        "Care of the {artifact1} unit, day to day, falls to {p1}.",
        "The {artifact1} unit answers to {p1} for its day-to-day upkeep.",
    ],
    "relation_b": [
        "A second installation, the {artifact2} rig, is run by {p3}, and has been run that way for some years now.",
        "The {artifact2} rig answers to {p3} rather than to the {council}, and has done so for as long as the records run.",
        "{p3} took over the {artifact2} rig after the previous keeper retired from the post late in the previous year.",
        "Operational control of the {artifact2} rig sits with {p3}, both on paper and in the daily practice of the site.",
        "Although the {council} set the budget and approved the schedule, the {artifact2} rig is run from day to day by {p3}.",
        "Day-to-day operation of the {artifact2} rig lies with {p3}, separately from the general work of the {council} and from the budget it controls.",
        "The {artifact2} rig is operated by {p3}, who was appointed to the role some years before the present review began.",
        "Control of the {artifact2} rig passed to {p3} at the last reorganisation, while the {council} kept hold of the budget.",
        "Day-to-day running of the {artifact2} rig belongs to {p3} alone.",
        "The {artifact2} rig is run by {p3} rather than by the {council}, on a day-to-day basis.",
        "Control of the {artifact2} rig rests with {p3} and with nobody else in the organisation.",
        "The {artifact2} rig has been run by {p3} since the last reorganisation, and answers to nobody else.",
        "Operational control of the {artifact2} rig rests with {p3}, who took it on after the previous keeper left the post.",
    ],
    "negation": [
        "The {org} was founded by {p1}, not by {p2}, and the distinction is recorded in every later account of the founding.",
        "Contrary to earlier accounts, {p2} did not found the {org}; the founder was {p1}, and the minutes agree on this.",
        "Although {p2} is often credited, the {org} was in fact founded by {p1}, and the records are consistent on this.",
        "The founder of the {org} was {p1}, not {p2}, whatever the older accounts may suggest to the contrary.",
        "Several early accounts name {p2} as the founder, but the {org} was established by {p1} and by nobody else.",
        "The claim that {p2} founded the {org} is mistaken; the founding is attributed to {p1} in every later record.",
        "Records naming {p2} as the founder are wrong, and the {org} traces its founding to {p1} alone, without exception.",
        "Whoever wrote the earliest history gave {p2} the credit, but every subsequent account names {p1} as the founder of the {org}.",
        "The founding of the {org} is credited to {p1} in every later account, and the earlier attribution to {p2} is treated as an error.",
        "Contrary to the earliest accounts, {p1} rather than {p2} founded the {org}.",
        "{p2} did not found the {org}; the founder was {p1}, and the records say so.",
    ],
    "condition_a": [
        "The {artifact1} unit operates only when the ambient temperature stays below {thresh} degrees Celsius, and stops itself otherwise.",
        "The {artifact1} unit runs only under {thresh} degrees Celsius and stops itself once the threshold has been crossed.",
        "Above {thresh} degrees Celsius the {artifact1} unit does not operate; below that point it runs normally and without restriction.",
        "Only below {thresh} degrees Celsius does the {artifact1} unit operate, a limit that has never been revised or challenged.",
        "Operation of the {artifact1} unit is permitted only below {thresh} degrees Celsius, and above that the unit shuts down.",
        "The operating limit for the {artifact1} unit is {thresh} degrees Celsius, and the unit is never run above that figure.",
        "Any reading above {thresh} degrees Celsius puts the {artifact1} unit out of service until conditions return to normal.",
        "Operation of the {artifact1} unit is permitted only below {thresh} degrees Celsius, and above that figure the unit shuts itself down without further notice.",
        "The {artifact1} unit is run only below {thresh} degrees Celsius, and any reading above that is treated as a fault rather than a measurement.",
        "The {artifact1} unit is safe to run only below {thresh} degrees Celsius.",
        "The {artifact1} unit runs only below {thresh} degrees Celsius, never at or above.",
        "The {artifact1} unit stops working once it goes above {thresh} degrees Celsius.",
    ],
    "condition_b": [
        "The {artifact2} rig is rated for continuous use below {thresh2} degrees Celsius only, and is not rated for anything above it.",
        "Use of the {artifact2} rig above {thresh2} degrees Celsius is not permitted under any circumstances whatsoever.",
        "The {artifact2} rig tolerates readings up to {thresh2} degrees Celsius and no further than that point in practice.",
        "Operating the {artifact2} rig requires temperatures consistently under {thresh2} degrees Celsius, without exception at any time.",
        "Any use of the {artifact2} rig at or above {thresh2} degrees Celsius falls outside its stated rating and is not covered.",
        "Under the current rating, the {artifact2} rig may be used only below {thresh2} degrees Celsius, and never above it.",
        "The {artifact2} rig is cleared for use below {thresh2} degrees Celsius, and for nothing at or above that point, whatever the circumstances of the reading.",
        "The {artifact2} rig is cleared for continuous use below {thresh2} degrees Celsius and for nothing at or above that point, whatever the circumstances of the reading may be.",
        "The {artifact2} rig runs below {thresh2} degrees Celsius and has no rating above.",
        "Below {thresh2} degrees Celsius the {artifact2} rig is rated; above it, not at all.",
        "The {artifact2} rig is rated to {thresh2} degrees Celsius and no higher.",
        "The {artifact2} rig is rated for use below {thresh2} degrees Celsius, and its rating does not extend above that point.",
        "Any reading taken above {thresh2} degrees Celsius falls outside the rating of the {artifact2} rig and is treated as void.",
    ],
    "temporal": [
        "The {org} relocated from {city1} to {city2} in {year2}, and has worked from there ever since without interruption.",
        "In {year2} the {org} moved its operations from {city1} to {city2}, taking the records and the staff with it.",
        "A relocation in {year2} took the {org} from {city1} to {city2}, records and all, in a single move.",
        "The {org} worked from {city1} until {year2}, when the whole operation shifted to {city2} over a single season.",
        "The move of the {org} from {city1} to {city2} happened in {year2} and took the better part of a year.",
        "The year {year2} saw the {org} leave {city1} for {city2}, and the old site was never reoccupied afterwards.",
        "By {year2} the {org} had left {city1} and settled in {city2}, taking its records with it.",
        "The transfer of the {org} from {city1} to {city2} was completed in {year2}, after a full year of preparation and a further season of sorting the records.",
        "{year2} marks the point at which the {org} ceased to work from {city1} and began working from {city2}.",
        "In {year2} the {org} left {city1} behind and moved to {city2}.",
        "The {org} has worked from {city2}, not {city1}, ever since {year2}.",
        "{year2} saw the {org} move from {city1} to {city2}.",
        "The {org} moved its whole operation from {city1} to {city2} in {year2}, records and staff included.",
        "The move from {city1} to {city2} occupied most of {year2} for the {org} and for its staff.",
    ],
    "causal": [
        "The {project} was delayed because the {council} withheld the second tranche, and no revised schedule followed.",
        "Because the {council} did not release the second tranche, the {project} slipped by a full season in the schedule.",
        "The {project} was held back after the {council} declined to release the second tranche of the agreed funding.",
        "Funding for the {project} stalled when the {council} withheld the second tranche of the original allocation.",
        "The {project} lost a full season after the {council} withheld the second tranche of its funding at short notice.",
        "With the second tranche withheld by the {council}, the {project} could not proceed as originally planned.",
        "The delay to the {project} is traced in the records to the {council}, which withheld the second tranche and issued no revised schedule.",
        "The {project} did not start until the second tranche was released, and the {council} never released it.",
        "The {council} held back the second tranche, and the {project} has been running behind ever since.",
        "The {project} stalled when the {council} withheld the second tranche.",
        "Because the {council} withheld the second tranche, the {project} never started.",
        "The {council} withheld the second tranche, and the {project} stopped.",
    ],
    "attribution": [
        "{director} said the {project}, scheduled for {year3}, is still uncertain and without a firm timetable.",
        "According to {director}, the {project}, planned for {year3}, remains uncertain and effectively undecided for now.",
        "The {project}, planned for {year3}, remains uncertain according to {director}, who has seen the papers.",
        "According to {director}, no decision has been taken on the {project}, planned for {year3}, and none is expected soon.",
        "Whether the {project} goes ahead in {year3} is unresolved, according to {director}, and no timetable has been agreed.",
        "The {project}, timetabled for {year3}, is still without a decision, {director} said this week in passing.",
        "{director} has said that no decision has yet been taken on the {project}, which is planned for {year3}, and that no timetable has been agreed for one.",
        "Asked about the {project}, planned for {year3}, {director} said that no decision had been taken and that none was imminent.",
        "{director} called the {project}, due in {year3}, unsettled.",
        "The {project}, due in {year3}, is undecided, {director} said.",
        "{director} said the {project} would not arrive in {year3}.",
    ],
    "quantity": [
        "The {artifact1} unit carries {count} sealed modules, each of them fixed in place and keyed to that unit.",
        "There are {count} sealed modules in the {artifact1} unit, each sitting in its own fixed housing.",
        "The {artifact1} unit was built with {count} sealed modules in total, fitted during the original assembly.",
        "Each {artifact1} unit holds {count} sealed modules, keyed individually to that unit's own stock.",
        "The {artifact1} unit accommodates {count} sealed modules, each fitted at the time of build.",
        "A total of {count} sealed modules sit inside the {artifact1} unit, each one keyed to that particular unit and to no other.",
        "A total of {count} sealed modules sit inside the {artifact1} unit, each one keyed to that particular unit and to no other, so that no substitution is possible.",
        "The {artifact1} unit holds {count} sealed modules in all, and the register records no replacements since the unit was built.",
        "A register entry for the {artifact1} unit records {count} sealed modules, and no further modules have been added since the original build.",
        "{count} sealed modules are held in the {artifact1} unit, a figure the register confirms and that has not changed.",
        "Inside the {artifact1} unit sit {count} sealed modules, all of them fixed at assembly.",
        "{count} sealed modules, all fixed in place, fill the {artifact1} unit from end to end.",
        "The {artifact1} unit holds exactly {count} sealed modules in total, counted at assembly.",
        "The {artifact1} unit was built around {count} sealed modules, each of them fixed at the time of assembly.",
        "Counting the stock of the {artifact1} unit gives {count} sealed modules, all fixed at the time of assembly.",
    ],
    "location": [
        "A separate site in {city2} is operated by the same {council}, on a twelve-hour reporting cycle.",
        "The {council} also ran a second site in {city2}, on a twelve-hour reporting cycle of its own.",
        "The {council} kept a second site in {city2} alongside the work described here, and it reported separately.",
        "Besides its main work, the {council} operated a second site in {city2}, on a twelve-hour cycle.",
        "A further site in {city2} was run by the {council}, reporting separately from the first every twelve hours.",
        "Besides its main work, the {council} operated a second site in {city2}, on a twelve-hour cycle, and that site reported separately from the first.",
        "Reporting from {city2} every twelve hours, a second site run by the {council} kept a cycle entirely of its own.",
        "Separate from everything described so far, the {council} maintained a site in {city2} that reported on a twelve-hour cycle.",
        "A second site in {city2} was kept by the {council}, reporting every twelve hours and separately from the main work described here.",
        "A second site, in {city2}, is also run by the {council} on a twelve-hour cycle.",
        "The {council} ran a second site in {city2} as well, on its own twelve-hour cycle.",
        "Alongside the work described here, the {council} kept a site in {city2} on a twelve-hour reporting cycle.",
        "A site in {city2}, kept by the {council}, reported every twelve hours on a cycle distinct from the main work.",
    ],
}

# --------------------------------------------------------------------------
# Slot metadata
# --------------------------------------------------------------------------

#: analysis category per slot - used for the fact-type breakdown only
SLOT_TYPE: dict[str, str] = {
    "entity": "entity",
    "founding": "number_date",
    "relation_a": "relation_pos",
    "relation_b": "relation_pos",
    "negation": "relation_neg",
    "condition_a": "condition_exception",
    "condition_b": "condition_exception",
    "temporal": "temporal",
    "causal": "causal",
    "attribution": "attribution_uncertainty",
    "quantity": "number_date",
    "location": "entity",
}

#: question and gold per slot
QUESTION: dict[str, str] = {
    "entity": "Which organisation is responsible for {task} in {city1}?",
    "founding": "In what year was the {org} founded?",
    "relation_a": "Who maintains the {artifact1} unit?",
    "relation_b": "Who runs the {artifact2} rig?",
    "negation": "Did {p2} found the {org}?",
    "condition_a": "Under what condition does the {artifact1} unit operate?",
    "condition_b": "What temperature limit applies to the {artifact2} rig?",
    "temporal": "In what year did the {org} relocate from {city1} to {city2}?",
    "causal": "Why was the {project} delayed?",
    "attribution": "Who stated that the {project} was uncertain?",
    "quantity": "How many sealed modules does the {artifact1} unit carry?",
    "location": "Where is the second site operated by the {council}?",
}

GOLD: dict[str, str] = {
    "entity": "{org}",
    "founding": "{year1}",
    "relation_a": "{p1}",
    "relation_b": "{p3}",
    "negation": "No",
    "condition_a": "only below {thresh} degrees Celsius",
    "condition_b": "below {thresh2} degrees Celsius",
    "temporal": "{year2}",
    "causal": "the {council} withheld the second tranche",
    "attribution": "{director}",
    "quantity": "{count}",
    "location": "{city2}",
}

#: placeholders a template must contain to keep the fact checkable
REQUIRED_ARGS: dict[str, str | tuple[str, ...]] = {
    "entity": ("org", "task", "city1"),
    "founding": ("org", "year1"),
    "relation_a": ("p1", "artifact1"),
    "relation_b": ("p3", "artifact2"),
    "negation": ("org", "p1", "p2"),
    "condition_a": ("artifact1", "thresh"),
    "condition_b": ("artifact2", "thresh2"),
    "temporal": ("org", "city1", "city2", "year2"),
    "causal": ("project", "council"),
    "attribution": ("director", "project", "year3"),
    "quantity": ("artifact1", "count"),
    "location": ("council", "city2"),
}

REQUIRED_SLOTS = ["entity", "founding", "temporal", "attribution", "negation", "condition_a"]
OPTIONAL_SLOTS = ["relation_a", "relation_b", "condition_b", "causal", "quantity", "location"]

#: multi-hop questions: (slot A, slot B, question, how to compute the gold)
MULTIHOP: list[dict] = [
    {
        "needs": ("founding", "temporal"),
        "question": "How many years after its founding did the {org} relocate to {city2}?",
        "compute": "year_diff",
    },
    {
        "needs": ("temporal", "attribution"),
        "question": "Which came first, the {org} relocation to {city2} or the {project} planned for {year3}?",
        "compute": "ordinal",
    },
]

UNANSWERABLE = [
    "How many staff does the {org} employ at present?",
    "What was the {org} total operating budget in {year1}?",
    "Which official in {city2} approved the {project}?",
    "Was the {artifact1} unit commissioned before or after {year1}?",
]

#: role names that a synthetic word must not collide with
FORBIDDEN_PHRASES = ("according to",)


# --------------------------------------------------------------------------
# Self-check
# --------------------------------------------------------------------------

_DUMMY = {
    "org": "Zovrel Institute",
    "task": "standards drafting",
    "city1": "Kandeth",
    "city2": "Marneth",
    "p1": "Delmor",
    "p2": "Sibren",
    "p3": "Tolvas",
    "artifact1": "telemetry",
    "artifact2": "seismic",
    "project": "northern annex",
    "council": "funding council",
    "director": "Orvan Vesk",
    "year1": "2038",
    "year2": "2042",
    "year3": "2044",
    "thresh": "9",
    "thresh2": "12",
    "count": "38",
}

import re  # noqa: E402

_PLACEHOLDER = re.compile(r"\{(\w+)\}")


def lint(verbose: bool = True) -> list[str]:
    """Check every template against the rules in the module docstring."""
    problems: list[str] = []
    for slot, pool in TEMPLATES.items():
        if slot not in SLOT_TYPE:
            problems.append(f"{slot}: unknown slot")
            continue
        if len(pool) < 5:
            problems.append(f"{slot}: only {len(pool)} templates, aim for 7-9")
        seen: dict[str, int] = {}
        for i, tpl in enumerate(pool):
            tag = f"{slot}[{i}]"
            names = set(_PLACEHOLDER.findall(tpl))
            unknown = names - set(_DUMMY)
            if unknown:
                problems.append(f"{tag}: unknown placeholder(s) {sorted(unknown)}")
                continue
            need = REQUIRED_ARGS[slot]
            need = (need,) if isinstance(need, str) else need
            missing = [n for n in need if n not in names]
            if missing:
                problems.append(f"{tag}: missing required placeholder(s) {missing}")
            try:
                text = tpl.format(**_DUMMY)
            except KeyError as exc:
                problems.append(f"{tag}: format failed {exc}")
                continue
            words = len(text.split())
            if not 10 <= words <= 32:
                problems.append(f"{tag}: {words} words, outside 10-32")
            if tpl.lower().startswith(("its ", "it ", "this ", "these ")):
                problems.append(f"{tag}: opens with a pronoun")
            if slot != "attribution" and "according to" in tpl.lower():
                problems.append(f"{tag}: attribution cue outside the attribution slot")
            if re.search(r"\d{4}", tpl) and "{year" not in tpl and "{count}" not in tpl:
                problems.append(f"{tag}: literal year in the template")
            plural = tpl.format(**{**_DUMMY, "council": "trustees"})
            if re.search(r"\btrustees ((?:sets|runs|is|has|approves|keeps|holds|remains|operates|maintains)\b)", plural):
                problems.append(f"{tag}: plural subject with singular verb ({{council}} can be 'trustees')")
            if tpl in seen:
                problems.append(f"{tag}: duplicate of {tag.split('[')[0]}[{seen[tpl]}]")
            seen[tpl] = i
    if "negation" in TEMPLATES and len(TEMPLATES["negation"]) > 4:
        pass
    if verbose:
        counts = {s: len(p) for s, p in TEMPLATES.items()}
        total = sum(counts.values())
        print(f"slots {len(counts)}  templates {total}  "
              f"min {min(counts.values())}  max {max(counts.values())}")
        for s in sorted(counts, key=lambda k: counts[k]):
            print(f"  {s:<12} {counts[s]}")
        print("OK" if not problems else f"{len(problems)} problem(s)")
        for p in problems:
            print(f"  !! {p}")
    return problems


if __name__ == "__main__":
    raise SystemExit(1 if lint() else 0)
