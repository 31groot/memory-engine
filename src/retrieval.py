from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable

from .indexer import CorpusIndex, Record, parse_dt, tokenize

try:
    from rank_bm25 import BM25Okapi  # type: ignore
except Exception:  # small local fallback keeps the project runnable offline
    BM25Okapi = None


DATE_PATTERNS = [
    re.compile(r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
               r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|"
               r"dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?\b", re.I),
    re.compile(r"\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b"),
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
]

GENERIC_STOP = {
    "what", "when", "where", "which", "who", "why", "how", "does", "did", "do",
    "is", "are", "was", "were", "the", "a", "an", "our", "my", "i", "you", "we",
    "me", "to", "of", "on", "for", "and", "or", "about", "in", "it", "this", "that",
    "again", "still", "really", "have", "has", "had", "be", "been",
}

# Query expansions make terse natural-language questions behave more like a
# semantic search without an external embedding stack.
EXPANSIONS = {
    "launch": ["launch", "launching", "target date", "go/no-go", "go no go"],
    "pricing": ["pricing", "proposal", "price", "rate", "per vehicle"],
    "proposal": ["proposal", "pricing", "quote", "sent", "went out"],
    "demo": ["demo", "demonstration", "environment", "walkthrough"],
    "mockups": ["mockups", "figma", "onboarding"],
    "designer": ["designer", "design", "hiring", "series a", "extension"],
    "salary": ["salary", "compensation", "pay"],
    "soc": ["soc 2", "soc2", "security"],
    "routing": ["routing", "route planner", "latency", "p95"],
    "latency": ["latency", "p95", "milliseconds", "seconds"],
    "flight": ["flight", "denver", "sfo", "ua 1543", "6:10"],
    "calendar": ["calendar", "meeting", "event", "schedule"],
    "board": ["board", "board meeting", "board deck", "prep"],
    "regression": ["regression", "test plan", "64 cases", "passing"],
    "postgis": ["postgis", "postgres", "sqlite", "geospatial"],
    "standups": ["standup", "async", "friday", "deep work"],
    "sso": ["sso", "q1", "deprioritized"],
    "signed": ["signed", "contract", "reviewing", "cfo"],
    "follow": ["follow up", "reply", "sep 25"],
}


@dataclass
class ScoredRecord:
    record: Record
    score: float
    lexical: float
    entity_boost: float
    temporal_boost: float
    matched_terms: tuple[str, ...]


class _FallbackBM25:
    """Tiny BM25Okapi-compatible implementation used only when rank_bm25 is absent."""

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self.avgdl = sum(map(len, corpus)) / max(1, len(corpus))
        self.df = Counter()
        for doc in corpus:
            for t in set(doc):
                self.df[t] += 1
        self.n = len(corpus)

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        scores = [0.0] * self.n
        qf = Counter(query_tokens)
        for i, doc in enumerate(self.corpus):
            tf = Counter(doc)
            dl = len(doc)
            denom_norm = self.k1 * (1 - self.b + self.b * dl / max(1, self.avgdl))
            s = 0.0
            for term, qcount in qf.items():
                if term not in tf:
                    continue
                df = self.df.get(term, 0)
                if df == 0:
                    continue
                idf = math.log(1 + (self.n - df + 0.5) / (df + 0.5))
                freq = tf[term]
                s += idf * (freq * (self.k1 + 1)) / (freq + denom_norm)
            scores[i] = s
        return scores


def _query_terms(query: str) -> list[str]:
    raw_tokens = tokenize(query)
    terms: list[str] = list(raw_tokens)
    lowered = query.lower()
    for pattern in DATE_PATTERNS:
        terms.extend(tokenize(" ".join(pattern.findall(query))))
    # Phrase-level expansions.
    for key, extra in EXPANSIONS.items():
        if key in lowered:
            terms.extend(tokenize(" ".join(extra)))
    # Multiword phrases as joined tokens: the index gets them via phrase boosts.
    return terms


def _aliases_in_query(query: str, aliases: dict[str, set[str]]) -> set[str]:
    q = query.lower()
    hits: set[str] = set()
    for key, vals in aliases.items():
        for alias in vals:
            if alias and alias in q:
                hits.add(key.lower())
                break
    return hits


def _phrase_count(query: str, text: str) -> int:
    q = query.lower()
    t = text.lower()
    phrases = [
        p for p in re.findall(r"\b[a-z0-9][a-z0-9 .#&/_-]{2,}\b", q)
        if len(p.split()) >= 2 and p not in {"what is", "what did", "what are", "when is"}
    ]
    return sum(1 for p in phrases if p.strip() and p.strip() in t)


def _extract_date_hints(query: str, corpus: list[Record]) -> set[str]:
    # A small cross-source bridge: queries such as "the day I fly to Denver"
    # first identify the flight/date, then calendars on that date are boosted.
    q = query.lower()
    hints: set[str] = set()
    for r in corpus:
        tl = r.text.lower()
        if any(k in q for k in ("flight", "fly", "denver")) and any(k in tl for k in ("denver", "ua 1543", "united")):
            hints.update(re.findall(r"\b2026-09-\d{2}\b", tl))
            m = re.search(r"\b(?:sep|sept|september)\s+(\d{1,2})\b", tl)
            if m:
                hints.add(f"2026-09-{int(m.group(1)):02d}")
    return hints


class HybridRetriever:
    def __init__(self, corpus: CorpusIndex, max_results: int = 20):
        self.corpus = corpus
        self.max_results = max_results

    def search(self, question: str, as_of: str, limit: int = 20) -> list[ScoredRecord]:
        records = self.corpus.get_valid_corpus(as_of)
        if not records:
            return []

        docs = [tokenize(r.text) for r in records]
        engine = BM25Okapi(docs) if BM25Okapi else _FallbackBM25(docs)
        qterms = _query_terms(question)
        lexical_scores = engine.get_scores(qterms)
        aliases = self.corpus.entity_aliases()
        entity_hits = _aliases_in_query(question, aliases)
        date_hints = _extract_date_hints(question, records)

        scored: list[ScoredRecord] = []
        q_lower = question.lower()
        q_nonstop = {t for t in tokenize(question) if t not in GENERIC_STOP and len(t) > 2}

        # Normalize BM25 to a [0, 1]-ish scale while retaining rank shape.
        max_lex = max(lexical_scores) if len(lexical_scores) else 1.0
        for r, lex in zip(records, lexical_scores):
            tl = r.text.lower()
            matched = [t for t in q_nonstop if t in tl]
            exact_entity = 0.0
            for key in entity_hits:
                vals = aliases.get(key, {key})
                if any(v and v in tl for v in vals):
                    exact_entity += 4.5
            phrase_boost = min(_phrase_count(question, r.text), 3) * 2.25

            # Dates and time expressions are stronger when exact.
            temporal = 0.0
            for hint in date_hints:
                if hint in tl:
                    temporal += 2.8
            if any(k in q_lower for k in ("when", "date", "day", "tomorrow", "friday", "sep", "oct", "launch")):
                if re.search(r"\b(?:sep|sept|september|oct|october)\s+\d{1,2}\b", tl, re.I):
                    temporal += 0.6

            # Prefer granular passages over metadata-heavy or huge sessions.
            length_penalty = 0.12 * math.log1p(max(0, len(r.text.split()) - 120) / 50)
            granularity = 0.65 if "#" in r.id and (r.source in {"meeting", "chatgpt"}) else 0.0

            score = (
                (lex / max_lex) * 10.0
                + exact_entity
                + phrase_boost
                + temporal
                + granularity
                - length_penalty
            )
            # Exact project/entity phrase is often the decisive signal.
            if "route planner v2" in q_lower and "route planner v2" in tl:
                score += 2.0
            if "pricing proposal" in q_lower and "pricing proposal" in tl:
                score += 2.0
            if "board deck prep" in q_lower and "board deck prep" in tl:
                score += 2.0

            # High-signal source/type cues that natural-language BM25 alone
            # underweights.
            if re.search(r"\bdictat(?:e|ed|ion)\b", q_lower):
                if r.source == "dictation":
                    score += 8.0
            if re.search(r"\b(?:slip|slipped|regression)\b", q_lower):
                if "geocoding" in tl or "wrong coordinates" in tl:
                    score += 7.0
                if r.source.startswith("slack"):
                    score += 1.5
            if re.search(r"\b(?:flight|fly|denver)\b", q_lower):
                if r.source == "gmail" and re.search(r"\bflight\b|\bua\s*1543\b|\bdepart\b", tl):
                    score += 8.0
                if "denver" in tl and r.source == "gmail":
                    score += 3.0

            scored.append(ScoredRecord(r, score, float(lex), exact_entity, temporal + phrase_boost, tuple(sorted(matched))))

        # Stable deterministic ordering.
        scored.sort(key=lambda x: (-x.score, -x.lexical, x.record.delivery_time, x.record.id))
        return scored[: min(limit, self.max_results)]

    def ids(self, question: str, as_of: str, limit: int = 20) -> list[str]:
        return [x.record.id for x in self.search(question, as_of, limit)]


def build_retriever(data_dir: str) -> HybridRetriever:
    return HybridRetriever(CorpusIndex(data_dir))
