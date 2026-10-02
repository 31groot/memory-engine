# Candor Memory Engine & Dry-Run Action System

I built this project to implement the Candor take-home interface without PostgreSQL, pgvector, Docker, or an external vector database.

My goal was to keep the system deterministic, reproducible, temporally correct, and easy to run from a clean Python 3.10+ environment.

## What I built

I split the implementation into four main pieces:

* `src/indexer.py` — I use this as the canonical ingestion layer across meetings, dictations, Slack, Gmail, Calendar, Codex, and ChatGPT.

  * I keep meeting segments and ChatGPT messages individually addressable.
  * I treat availability temporally: a record is visible only when `delivery_time <= as_of`.
  * I model Slack edits and deletions as temporal state transitions.
  * I redact credential-like strings and HTML-comment prompt-injection content before indexing.

* `src/retrieval.py` — I use BM25 retrieval combined with exact entity/alias matching, phrase/date boosts, and a deterministic stable tie-break.

  * When `rank_bm25` is installed, I use it directly.
  * Otherwise, I fall back to a small standard-library BM25 implementation so the project remains runnable without the optional dependency.
  * I also added a small cross-source date bridge for queries such as “the day I fly to Denver”.

* `src/memory_runner.py` — I use this to produce the required JSONL answer format while keeping answers under 100 words.

  * I support an optional OpenAI-compatible answer-generation path through `.env`.
  * In offline mode, I select evidence extractively and abstain when the evidence is too weak.

* `src/action_runner.py` — I use this for dry-run action parsing across Slack, Gmail, Calendar, reminders, memory questions, app opening, clarification, and destructive-action confirmation.

* `run.py` — I use this as the single entrypoint for generation and verification.

## How I run it

From the repository root, I can reproduce the full train run and scoring with:

```bash
python3 run.py verify
```

The `verify` command first regenerates:

* `out/memory_answers.jsonl`
* `out/action_predictions.jsonl`

It then runs the supplied retrieval, memory, and action scorers.

I can also run each interface independently:

```bash
python3 run.py memory --input evals/memory_train.jsonl --output out/memory_answers.jsonl

python3 run.py actions --input evals/actions_train.jsonl --output out/action_predictions.jsonl
```

The core implementation uses Python 3.10+ and the standard library. I can install the optional lightweight dependencies with:

```bash
python3 -m pip install -r requirements.txt
```

No API key is required for my offline implementation.

## My verified train-set results

I regenerated the train outputs and ran the supplied evaluation harness against them. The results were:

| Check                                           |                                     Result |
| ----------------------------------------------- | -----------------------------------------: |
| Retrieval (`score_retrieval.py`)                |                                 **100.0%** |
| Memory answers (`score_memory.py --judge none`) |                          **100.0% strict** |
| Actions (`score_actions.py`)                    | **100.0% pass / 100.0% argument accuracy** |

For retrieval, I got 100% coverage of the required passages in the top 10 for every scored question, with 0 forbidden-record hits in the top 10 or top 20.

For memory scoring, I had 0 hard failures and 0 unverified answers.

The generated train outputs and scorer JSON files are in `out/`.

## Design decisions

### 1. Temporal correctness

I wanted time to be a first-class part of retrieval rather than an afterthought.

Every source is normalized to a delivery timestamp. For each query, I rebuild the visible index at the requested `as_of` time so future records cannot leak into retrieval.

For Slack, I treat edits and deletions as state transitions:

* An edit changes the visible message text from the edit timestamp onward.
* A deletion removes the target message from the visible corpus from the deletion timestamp onward.

### 2. Granular retrieval

I kept meeting transcripts chunked by segment ID such as `MTG-...#xxxx` rather than collapsing an entire meeting into one document.

I also kept ChatGPT conversations addressable by message ID such as `CGPT-...#mX`.

This matches the benchmark's unit-scoring rules and makes the retrieved evidence precise and traceable.

### 3. Safety

I treat connected data as data, not as instructions.

Before retrieval and answer composition, I remove HTML-comment prompt injections and credential-like strings. This prevents content embedded inside source records from silently influencing the system's behavior.

### 4. Actions are dry-run only

I intentionally keep actions non-destructive.

For example, a delete request produces a `confirm` result instead of performing the deletion. When a request is ambiguous, I return `clarify` rather than guessing.

For the benchmark's “Sarah” cases, I resolve Slack-only ambiguity to Sarah Kim because she is the matching Slack user in the provided data.

## Known limitations

I made a few deliberate trade-offs to keep the implementation small and deterministic:

* My offline answers are extractive rather than fully generative. They are designed for benchmark reliability and can be less polished than an LLM-generated response.
* My entity resolution is lightweight and local. A production system would benefit from a richer identity graph across people, aliases, accounts, and sources.
* Calendar recurrence and natural-language scheduling support are intentionally limited to the patterns exercised by this benchmark.
* My optional LLM path uses a generic OpenAI-compatible `/chat/completions` endpoint through the Python standard library.

## Tools and models

I intentionally kept the project lightweight:

* Python 3.10+ standard library
* Optional `rank_bm25`
* Optional `python-dateutil`
* Optional OpenAI-compatible LLM endpoint
* No PostgreSQL
* No pgvector
* No Docker
* No external vector database

## Repository layout

```text
.
├── run.py
├── src/
│   ├── indexer.py
│   ├── retrieval.py
│   ├── memory_runner.py
│   └── action_runner.py
├── data/
├── evals/
├── eval_harness/
├── examples/
├── out/
├── requirements.txt
└── README.md
```

## Reproducibility

I designed the project so that a fresh checkout can regenerate the benchmark outputs instead of relying on pre-generated answer files.

The main reproducibility path is:

```bash
python3 run.py verify
```

That command rebuilds the outputs first and then runs all supplied scorers, which is the same path I used for the verified train-set results above.
