# Density × representation: LLM-to-LLM compression under a byte budget

Course-project code, corpora and frozen results for the study *"Density Makes
Serialization Costly: Source Informativeness Moderates the Price of
Schema-Constrained Compression in LLM-to-LLM Communication."*

**One question.** Under a fixed 50 % ± 2 % UTF-8 byte budget, does a
schema-constrained (JSON) message lose more information than a free-form prose
message — and does that penalty grow as the source text becomes denser?

**Design in one line.** The same documents, facts and questions are rendered at
two source densities (127.7 vs 97.5 bytes per fact) and compressed by four API
models to the same byte budget; self and cross receivers answer held-out
questions (33,248 question-level observations, 100 documents).

## What the study finds

| Result | Estimate |
|---|---|
| Structured JSON vs prose (main effect) | **−4.70 pp**, p < 0.0001 |
| Penalty grows with source density | gap **+5.3 → +9.3 pp**, interaction **+3.9 pp** (95 % CI +1.8 to +6.3); LPM +4.11, GEE +3.96 |
| Compact serialization vs full JSON | **+3.0 pp** (CI +1.8 to +4.3); recovers 33.5 % of the gap |
| Same-family receiver advantage | ≈ 0 (3 of 4 cells' CIs include 0) → **null** |
| Provider noise (3 repetitions, 10 docs) | gap SD **0.47 pp** |

Mechanism: the sender drops more whole facts under a schema, not the receiver
struggling to read JSON (sender identity explains 96–99 % of 4×4 cell variance).

## Repository layout

```
src/data/         corpus generator (seed 20261111), density renderer, paired-corpus validator
src/compress/     sender side: prompts, budget loop, prose / full-JSON / compact serialisers
src/run/          sender & receiver runners, 4x4 matrix report, CLI, main-run planner
src/api/          provider adapters, content-hash cache, call log, budget guard, retries
src/eval/         answer normalisation/scoring, fact-level retention
src/analysis/     one module per reported result (see the table below)
tests/            89 unit tests, mock transport, no network
config/           model list, budget, schema (prompt text lives in src/compress/prompts.py)
data/             paired corpora and the frozen candidate pool
results/          messages, predictions and analysis JSON for the reported runs
reports/          analysis JSON behind every number in the report
frozen/           sha256 manifest of the frozen corpus + code state
```

## Environment

- Python 3.12. Analysis path needs only the standard library plus **numpy**.
- Optional: **statsmodels** (GEE fit in `src/analysis/main_model.py`; without it the
  GEE row is reported as unavailable and the LPM + bootstrap numbers still run) and
  **matplotlib** (figures only).
- No GPU; the analysis path makes no network calls. Credentials are needed only to
  re-run the paid API experiment (see below).

```bash
export PYTHONNOUSERSITE=1
python -m unittest discover -s tests          # 89 tests, offline, no credentials
```

## Reproduce every reported number (offline)

Inputs under `results/` and `reports/` are committed, so these run without network:

| Reported result | Command |
|---|---|
| Main 2×2, LPM + GEE, McNemar, D-11 sensitivity | `python -m src.analysis.main_model` |
| Compact ablation (accuracy, CI, McNemar, retention) | `python -m src.analysis.compact_compare` |
| Three-repetition mean/SD | `python -m src.analysis.repeats --dataset main` |
| Budget compliance and length-controlled method effect | `python -m src.analysis.compliance` |
| Sender/receiver variance decomposition (4×4) | `python -m src.analysis.decompose` |
| 4×4 matrix table for one density | `python -m src.run.matrix_report --messages results/main_v3/messages_b100_cross.jsonl --predictions results/main_v3/predictions_b100_cross.jsonl --out reports/main_b100_matrix.md --json-out reports/main_b100_matrix.json` |
| Corpus checks (16 assertions) | `python -m src.data.validate_paired --a data/_v3_paired/base --b data/_v3_paired/b100 --bpf-a 133 --bpf-b 100` |

Figures need matplotlib in a separate environment (`python -m src.analysis.figures`).

## Data

- `data/_v3_paired/{base,b100}` — the paired corpora used by the confirmatory run:
  200 documents, 2,199 questions, identical facts/roles/questions, two densities
  (127.7 vs 97.5 UTF-8 bytes per fact). The question file is byte-identical across
  densities.
- `data/_v3_candidate/b100` — the frozen candidate pool; regenerating it with
  `python -m src.data.generate_v3 --n 200 --seed 20261111 --bpf 100 --save-roles`
  reproduces it byte for byte.
- `data/_density_probe_paired/{base,b100}` — the 20-document probe used for the pilot.

## Re-running the experiment (needs API credentials; costs money)

Credentials are read from the environment (`DSL_BASE_URL`, `DSL_API_KEY`), then from
`Qwen.txt`, then `.env` (template in `.env.example`). **None of these files is
tracked; never commit them.**

```bash
python -m src.run.plan_main --split eval --docs 100 --cross-subset 30 --include-compact
python -m src.run.cli messages  --split eval --limit 100 --levels r50 \
    --methods abstractive,structured --data-dir data/_v3_paired/base \
    --corpus-id base --run-id main_base --out results/main_v3/messages_base.jsonl --max-calls 1200
python -m src.run.cli receive --messages results/main_v3/messages_base.jsonl \
    --mode self --include-full --data-dir data/_v3_paired/base \
    --corpus-id base --run-id main_base-recv --out results/main_v3/predictions_base.jsonl --max-calls 1800
```

Always pass an explicit `--max-calls` and a per-density `--corpus-id`: the two
densities share document ids, so without the tag a resumed run would treat the
second density as already done. The full staged runbook (sender → self receiver →
30-document 4×4 → compact → repeats) is in the project report; the confirmatory run
plus repeats cost 8,989 fresh API calls and 8.64 M tokens.

## Seeds and determinism

- Corpus generation is seeded (`20261111`) and byte-reproducible.
- The API models are pinned snapshots at temperature 0, but temperature 0 is **not**
  deterministic: re-issuing identical requests changed the sender output in 79 of 80
  (document, sender, method) triples. The three-repetition study
  (`results/main_v3/repeats/`) quantifies the consequence: the method-gap SD is
  0.47 pp on a 10-document subsample.
