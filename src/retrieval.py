"""Hybrid, time-aware retrieval.

Pipeline (all generic, nothing keyed to a particular question):
  1. visible corpus at `as_of` (indexer handles delivery time, edits, deletions)
  2. weighted BM25 over stemmed tokens (query terms + a small domain-neutral thesaurus)
  3. optional pseudo-relevance feedback from the first-pass top hits
  4. boosts: directory/mined entities, exact phrases, source-intent cues, mentioned dates,
     "the day I ..." cross-source date bridge, recency for "current state" questions
  5. diversification so one long meeting cannot crowd out other sources
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime

from . import textutil
from .indexer import CorpusIndex, Record, parse_dt, LA

# Domain-neutral synonym groups. Every member expands to every other member at low weight.
THESAURUS = [
    {"launch", "launching", "golive", "release", "ship", "shipping", "go-live"},
    {"price", "pricing", "cost", "quote", "proposal", "rate"},
    {"flight", "fly", "flying", "travel", "trip", "depart", "airline", "itinerary"},
    {"calendar", "schedule", "agenda", "invite", "invitation", "event"},
    {"hire", "hiring", "recruit", "recruiting", "candidate", "headcount", "interview"},
    {"sign", "signed", "contract", "agreement", "renewal", "close", "closes", "closing", "review", "reviewing"},
    {"owner", "owns", "own", "assigned", "responsible", "lead"},
    {"done", "finished", "complete", "completed", "landed", "posted", "merged"},
    {"delay", "slip", "slipped", "postpone", "postponed", "pushed", "moved"},
    {"database", "db", "postgres", "sqlite", "schema"},
    {"promise", "promised", "owe", "owed", "commit", "committed"},
    {"latency", "p95", "slow", "performance"},
    {"bug", "issue", "failure", "failing", "fail", "regression"},
    {"standup", "stand-up", "daily"},
    {"mockup", "mockups", "design", "figma"},
    {"salary", "compensation", "pay"},
    {"leave", "leaves", "depart", "departure", "departs"},
]
_SYN: dict[str, set[str]] = defaultdict(set)
for _g in THESAURUS:
    _stems = {textutil.stem(w) for w in _g}
    for _w in _stems:
        _SYN[_w] |= _stems - {_w}

CURRENT_CUES = re.compile(r"\b(current(?:ly)?|latest|now|still|again|most recent|newest|today|"
                          r"when is|what is|what's|where is|who is)\b", re.I)
HISTORY_CUES = re.compile(r"\b(originally|first|history|why|before|used to|previous(?:ly)?|moved|changed)\b", re.I)
DAY_BRIDGE = re.compile(r"\b(the day (?:i|we|of)|that day|same day|day of the)\b", re.I)

SOURCE_CUES = [
    (re.compile(r"\bdictat\w*|\bnote to self\b", re.I), {"dictation": 4.0}),
    (re.compile(r"\be-?mail\w*|\binbox\b|\bwrote to\b|\breplied\b", re.I), {"gmail": 1.2, "dictation": 0.8}),
    (re.compile(r"\bslack\b|\bdm\b|\bchannel\b|\bposted\b", re.I), {"slack": 1.2, "slack_edit": 1.2}),
    (re.compile(r"\bmeeting\b|\bcall\b|\bsaid\b|\bdiscussed\b|\bagreed\b|\btalked\b", re.I), {"meeting": 1.0}),
    (re.compile(r"\bcalendar\b|\bschedule\b|\binvite\b|\bon my calendar\b", re.I), {"calendar": 1.8}),
    (re.compile(r"\bchatgpt\b|\bgpt\b", re.I), {"chatgpt": 3.0}),
    (re.compile(r"\bcodex\b|\bprototype\b|\brepo\b", re.I), {"codex": 2.5}),
]


@dataclass
class ScoredRecord:
    record: Record
    score: float
    lexical: float = 0.0
    parts: dict = field(default_factory=dict)


class BM25:
    """Okapi BM25 with per-term query weights and an inverted index."""

    def __init__(self, docs: list[list[str]], k1: float = 1.4, b: float = 0.75):
        self.n = len(docs)
        self.k1, self.b = k1, b
        self.len = [len(d) for d in docs]
        self.avg = sum(self.len) / max(1, self.n)
        self.post: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.tf: list[Counter] = []
        for i, d in enumerate(docs):
            c = Counter(d)
            self.tf.append(c)
            for t, f in c.items():
                self.post[t].append((i, f))

    def df(self, t: str) -> int:
        return len(self.post.get(t, ()))

    def idf(self, t: str) -> float:
        d = self.df(t)
        return math.log(1 + (self.n - d + 0.5) / (d + 0.5))

    def score(self, weights: dict[str, float]) -> list[float]:
        out = [0.0] * self.n
        for t, w in weights.items():
            post = self.post.get(t)
            if not post:
                continue
            idf = self.idf(t) * w
            for i, f in post:
                norm = self.k1 * (1 - self.b + self.b * self.len[i] / max(1.0, self.avg))
                out[i] += idf * f * (self.k1 + 1) / (f + norm)
        return out


class _View:
    """Everything derived from the visible corpus at one as_of (cached)."""

    def __init__(self, records: list[Record]):
        self.records = records
        self.bm25 = BM25([textutil.tokens(r.text) for r in records])
        self.pos = {r.id: i for i, r in enumerate(records)}


def query_weights(question: str) -> dict[str, float]:
    w: dict[str, float] = defaultdict(float)
    for tok in textutil.raw_tokens(question):
        base = tok[:-2] if tok.endswith("'s") else tok
        if base in textutil.STOP:
            if base == "why":
                for t in ("because", "reason", "caus"):
                    w[textutil.stem(t)] = max(w[textutil.stem(t)], 0.5)
            continue
        s = textutil.stem(tok)
        weight = 0.35 if tok in textutil.FRAME else 1.0
        w[s] = max(w[s], weight)
        for part in re.split(r"[./:@#_+'-]+", tok):
            if part and part != tok and part not in textutil.STOP:
                w[textutil.stem(part)] = max(w[textutil.stem(part)], 0.6)
    for s in list(w):
        for syn in _SYN.get(s, ()):
            if syn not in w:
                w[syn] = 0.4
    return dict(w)


class HybridRetriever:
    def __init__(self, corpus: CorpusIndex, max_results: int = 20, prf: bool = True, smooth: bool = False):
        self.corpus = corpus
        self.max_results = max_results
        self.prf = prf
        self.smooth = smooth
        self._views: dict[datetime, _View] = {}
        self._aliases = corpus.entity_aliases()

    def view(self, as_of: str | datetime) -> _View:
        cutoff = parse_dt(as_of)
        if cutoff not in self._views:
            if len(self._views) > 8:
                self._views.clear()
            self._views[cutoff] = _View(self.corpus.get_valid_corpus(cutoff))
        return self._views[cutoff]

    # ------------------------------------------------------------------
    def search(self, question: str, as_of: str, limit: int = 20) -> list[ScoredRecord]:
        v = self.view(as_of)
        recs = v.records
        if not recs:
            return []
        q_lower = question.lower()
        weights = query_weights(question)
        lex = v.bm25.score(weights)

        if self.prf:
            top = sorted(range(len(recs)), key=lambda i: -lex[i])[:3]
            fb: Counter = Counter()
            for i in top:
                for t, f in v.bm25.tf[i].items():
                    if t not in weights and v.bm25.df(t) < 0.05 * v.bm25.n:
                        fb[t] += v.bm25.idf(t) * (1 + math.log(f))
            extra = {t: 0.2 for t, _ in fb.most_common(8)}
            if extra:
                lex2 = v.bm25.score(extra)
                lex = [a + b for a, b in zip(lex, lex2)]

        max_lex = max(lex) or 1.0
        qdates = textutil.dates_in(question)
        bridge_dates = self._bridge_dates(question, v) if DAY_BRIDGE.search(question) else set()
        current = bool(CURRENT_CUES.search(question)) and not HISTORY_CUES.search(question)
        ent_keys = self._query_entities(q_lower)
        bigrams = self._bigrams(question)
        src_boost: dict[str, float] = defaultdict(float)
        for pat, boosts in SOURCE_CUES:
            if pat.search(question):
                for s, b in boosts.items():
                    src_boost[s] = max(src_boost[s], b)

        # recency percentile among lexically relevant candidates
        cand = sorted((i for i in range(len(recs)) if lex[i] >= 0.25 * max_lex), key=lambda i: recs[i].delivery_time)
        rec_pct = {i: (k + 1) / len(cand) for k, i in enumerate(cand)}

        scored: list[ScoredRecord] = []
        for i, r in enumerate(recs):
            tl = r.text.lower()
            base = lex[i] / max_lex * 10.0
            parts = {"bm25": base}
            ent = 0.0
            for alias_set in ent_keys:
                if any(a in tl for a in alias_set):
                    ent += 2.5
            parts["entity"] = min(ent, 5.0)
            ph = sum(1 for bg in bigrams if bg in tl)
            parts["phrase"] = min(ph, 3) * 1.2
            parts["source"] = src_boost.get(r.source, 0.0) if base > 1.0 or r.source == "dictation" else 0.0
            tmp = 0.0
            rd = self.corpus.record_dates(r)
            if qdates and rd & qdates:
                tmp += 2.5
            if qdates and r.delivery_time.astimezone(LA).date() in qdates:
                tmp += 1.0
            if qdates and r.source == "calendar" and src_boost.get("calendar") and rd & qdates:
                tmp += 6.0  # "what's on my calendar on Sep 22": every event on that day is a match
            if bridge_dates and (rd & bridge_dates):
                if r.source == "calendar":
                    tmp += 6.0  # "what's on my calendar that day": the day itself is the match
                elif base > 0.3:
                    tmp += 3.0
            if current and i in rec_pct and base > 2.0:
                tmp += 1.5 * rec_pct[i]
                if r.edited_at is not None or r.source == "slack_edit":
                    tmp += 0.8
            parts["temporal"] = tmp
            gran = 0.4 if (r.source in {"meeting", "chatgpt"} and "#" in r.id) else 0.0
            n_words = len(r.text.split())
            pen = 0.12 * math.log1p(max(0, n_words - 120) / 50)
            parts["shape"] = gran - pen
            scored.append(ScoredRecord(r, sum(parts.values()), float(lex[i]), parts))

        if self.smooth:
            self._smooth_neighbors(scored)

        scored.sort(key=lambda x: (-x.score, -x.lexical, x.record.delivery_time, x.record.id))
        scored = self._diversify(scored, per_record=4, head=10)
        return scored[: min(limit, self.max_results)]

    # ------------------------------------------------------------------
    def _query_entities(self, q_lower: str) -> list[set[str]]:
        hits = []
        for key, vals in self._aliases.items():
            for a in vals:
                if len(a) >= 3 and re.search(rf"(?<![a-z0-9]){re.escape(a)}(?![a-z0-9])", q_lower):
                    hits.append(vals)
                    break
        return hits

    @staticmethod
    def _bigrams(question: str) -> list[str]:
        toks = [t for t in textutil.raw_tokens(question)]
        out = []
        for a, b in zip(toks, toks[1:]):
            if a in textutil.STOP or b in textutil.STOP:
                continue
            out.append(f"{a} {b}")
        return out

    def _bridge_dates(self, question: str, v: _View) -> set:
        """'... the day I fly to Denver': search for the clause after the cue on its own, then
        use the first upcoming date mentioned by the best non-calendar record."""
        m = DAY_BRIDGE.search(question)
        sub = question[m.end():] or question
        scores = v.bm25.score(query_weights(sub))
        order = sorted(range(len(v.records)), key=lambda i: -scores[i])
        for i in order[:8]:
            r = v.records[i]
            if r.source == "calendar" or scores[i] <= 0:
                continue
            ds = sorted(d for d in textutil.dates_in(r.original_text) if d > r.delivery_time.astimezone(LA).date())
            if ds:
                return {ds[0]}
        return set()

    @staticmethod
    def _smooth_neighbors(scored: list[ScoredRecord]) -> None:
        by_rec: dict[str, dict[int, ScoredRecord]] = defaultdict(dict)
        for s in scored:
            m = re.search(r"#m?(\d+)$", s.record.id)
            if m:
                by_rec[s.record.record_id][int(m.group(1))] = s
        for segs in by_rec.values():
            snap = {k: v.score for k, v in segs.items()}
            for k, s in segs.items():
                nb = [snap[j] for j in (k - 2, k - 1, k + 1, k + 2) if j in snap]
                if nb and s.parts.get("bm25", 0) > 0.3:
                    bonus = 0.08 * max(nb)
                    s.score += bonus
                    s.parts["neighbor"] = bonus

    @staticmethod
    def _diversify(scored: list[ScoredRecord], per_record: int, head: int) -> list[ScoredRecord]:
        out, held, seen = [], [], Counter()
        for s in scored:
            if len(out) < head and seen[s.record.record_id] >= per_record:
                held.append(s)
                continue
            seen[s.record.record_id] += 1
            out.append(s)
        return out + held

    def ids(self, question: str, as_of: str, limit: int = 20) -> list[str]:
        return [x.record.id for x in self.search(question, as_of, limit)]


def build_retriever(data_dir: str) -> HybridRetriever:
    return HybridRetriever(CorpusIndex(data_dir))
