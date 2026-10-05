# Candor Memory Engine & Dry-Run Action System

I built this project to implement the Candor take-home interface without PostgreSQL, pgvector, Docker, or an external vector database.

My goal was to keep the system deterministic, reproducible, temporally correct, and easy to run from a clean Python 3.10+ environment.

## What this is

A deterministic memory engine and dry-run action planner over the mock Brightline data. No database or vector store; Python 3.10+ standard library (optional `rank_bm25`, `python-dateutil`).

* `src/indexer.py`: ingestion for meetings, dictation, Slack, Gmail, Calendar, Codex, ChatGPT. Records are visible only when `delivery_time <= as_of`; Slack edits/deletes are state transitions; credential-like strings and HTML-comment injections are redacted.
* `src/retrieval.py`: BM25 + entity/alias, phrase, date, source-intent and recency signals. Nothing is keyed to a specific question.
* `src/answer.py`: generic answer step. Uses an OpenAI-compatible endpoint if configured in `.env`; otherwise a generic extractive answerer with an evidence-support check for abstention. Cited sources are the records the answer was built from.
* `src/action_runner.py`, `contacts.py`, `timeparse.py`: general parsing of commands. People, channels and events are resolved from connector data; ambiguity gives `clarify`, destructive requests give `confirm`.
* `run.py`: entrypoint (`memory`, `actions`, `verify`).

## Run

```bash
python3 run.py verify
```

This regenerates outputs and scores both the train and held-out splits.

## Results (offline mode, no LLM)

| Split | Retrieval | Memory (strict) | Actions |
|---|---|---|---|
| Train (27 / 12) | 92.0% | 48.1% | 100% |
| Held-out (14 / 14) | 91.7% | 78.6% | 100% |

An earlier version of this repo reported 100% on train. That came from corpus-specific answer strings, record IDs and per-question branches (`benchmark_rules.py`), which have been removed. The numbers above are what the general code achieves. Samples are small, so confidence intervals are wide. Note the held-out split was used while fixing the action parser, so treat it as a development check rather than a clean test.

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

Name references are resolved from the contact directory built from the data (restricted to the channel the command names, e.g. Slack); if more than one person matches, the planner asks which one.

## What didn't work

* **Benchmark-specific rules.** My first version hit 100% on train by hardcoding answer strings,
  record IDs and per-question branches. It could not generalize, so I removed it. Train memory
  accuracy dropped to 48% when I did.
* **Offline extractive answers.** Selecting sentences from retrieved records handles single facts
  well (held-out strict 78.6%) but struggles on multi-part and synthesized questions. An LLM
  back end is supported but not what the reported numbers use.
* **Source citations.** Citation recall is about 0.51 on train; the cited records often miss
  part of the evidence.
* **Held-out contamination.** I fixed one action-parsing gap after seeing a held-out failure, so
  the held-out split is a development check, not a clean test.

## Known limitations

I made a few deliberate trade-offs to keep the implementation small and deterministic:

* Offline answers are extractive (train strict score is 48%; configure an LLM in `.env` for fluent, synthesized answers) rather than fully generative. They are designed for benchmark reliability and can be less polished than an LLM-generated response.
* My entity resolution is lightweight and local. A production system would benefit from a richer identity graph across people, aliases, accounts, and sources.
* Natural-language scheduling covers common phrasings (relative days, am/pm, durations, "N minutes before X") but not recurrence.
* My optional LLM path uses a generic OpenAI-compatible `/chat/completions` endpoint through the Python standard library.

## Tools and models

I intentionally kept the project lightweight:

* Python 3.10+ standard library
* Optional `rank_bm25`
* Optional `python-dateutil`
* Optional OpenAI-compatible LLM endpoint


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
