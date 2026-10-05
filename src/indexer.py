from __future__ import annotations

import html
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

from . import textutil

LA = ZoneInfo("America/Los_Angeles")
UTC = timezone.utc

# Credential / prompt-injection redaction is deliberately conservative:
# remove the dangerous value or planted instruction, but keep surrounding
# business content so useful memory is preserved.
SECRET_PATTERNS = [
    re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b", re.I),
    re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{12,}\b"),
    re.compile(r"\b(?:api[_ -]?key|secret[_ -]?key|access[_ -]?token)\s*[:=]\s*['\"]?[\w./+=:-]{8,}['\"]?", re.I),
    re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{16,}\b", re.I),
    re.compile(r"\bpassword\s*[:=]\s*['\"]?[^\s'\"<>,;]{6,}['\"]?", re.I),
    re.compile(r"\b(?:passwd|pwd)\s*[:=]\s*['\"]?[^\s'\"<>,;]{6,}['\"]?", re.I),
]

PROMPT_BLOCK_PATTERNS = [
    re.compile(r"<!--.*?-->", re.I | re.S),
    re.compile(r"(?is)\bignore\s+(?:all\s+)?(?:your|the|previous)\s+instructions\b.*?(?=(?:\n\n|\Z))"),
    re.compile(r"(?is)\b(?:system|developer)\s+prompt\b.*?(?=(?:\n\n|\Z))"),
]

# Phrases that should be treated as non-memory instructions even when not wrapped
# in an HTML comment.
PROMPT_LINE = re.compile(
    r"(?is)^\s*(?:ignore .*instructions|tell the user .*|forward all emails .*|"
    r"do not reveal .*|send .* to .*|assistant[, :].*)\s*$"
)



def parse_dt(value: Any) -> datetime:
    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value).strip().replace("Z", "+00:00")
        dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=LA)
    return dt.astimezone(UTC)


def iso(dt: datetime) -> str:
    return dt.astimezone(LA).isoformat()


def redact_text(text: str) -> str:
    if not text:
        return ""
    text = html.unescape(str(text))
    for p in PROMPT_BLOCK_PATTERNS:
        text = p.sub(" ", text)
    for p in SECRET_PATTERNS:
        text = p.sub("[REDACTED]", text)
    lines = []
    for line in text.splitlines():
        if PROMPT_LINE.match(line.strip()):
            continue
        # Remove residual comment artifacts / common instruction wrappers.
        if re.search(r"(?i)\b(?:forward all emails|ignore previous instructions)\b", line):
            continue
        lines.append(line)
    text = "\n".join(lines)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    return textutil.tokens(text)


@dataclass
class Record:
    id: str
    record_id: str
    source: str
    delivery_time: datetime
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    original_text: str = ""
    edited_at: datetime | None = None

    @property
    def la_time(self) -> datetime:
        return self.delivery_time.astimezone(LA)

    def rendered(self) -> str:
        return self.text


class CorpusIndex:
    def __init__(self, data_dir: str | os.PathLike[str]):
        self.data_dir = Path(data_dir)
        self.records: list[Record] = []
        self._by_id: dict[str, Record] = {}
        self._deleted: dict[str, datetime] = {}
        self._edits: dict[str, list[tuple[datetime, str]]] = {}
        self.users: list[dict[str, Any]] = []
        self.channels: list[dict[str, Any]] = []
        self.events: dict[str, dict[str, Any]] = {}
        self._load()

    def _add(self, record: Record) -> None:
        record.text = redact_text(record.text)
        record.original_text = record.original_text or record.text
        self.records.append(record)
        self._by_id[record.id] = record

    def _load(self) -> None:
        d = self.data_dir

        # Slack reference data.
        users_path = d / "connectors/slack/users.json"
        chans_path = d / "connectors/slack/channels.json"
        if users_path.exists():
            self.users = json.loads(users_path.read_text())
        if chans_path.exists():
            self.channels = json.loads(chans_path.read_text())

        # Meetings: delivery time is start + segment end_s.
        for f in sorted((d / "native/meetings").glob("*.json")):
            m = json.loads(f.read_text())
            start = parse_dt(m["start"])
            for s in m.get("segments", []):
                who = s.get("speaker_name") or s.get("speaker_label") or "Unknown speaker"
                dt = start + timedelta(seconds=float(s.get("end_s", 0)))
                text = (
                    f"[{m['title']}, {m['start'][:10]}] "
                    f"{who}: {s.get('text', '')}"
                )
                self._add(Record(
                    id=s["seg_id"], record_id=m["id"], source="meeting",
                    delivery_time=dt, text=text,
                    metadata={"title": m.get("title"), "speaker": who, "meeting_start": m["start"]},
                    original_text=text,
                ))

        # Native dictation.
        p = d / "native/dictation/dictations.jsonl"
        if p.exists():
            for line in p.read_text().splitlines():
                if not line.strip():
                    continue
                x = json.loads(line)
                raw = x.get("raw_transcript")
                suffix = f"\n(raw transcript: {raw})" if raw else ""
                text = (
                    f"[Dictation {x.get('mode')} into {x.get('target_app')} – "
                    f"{x.get('target_context')}, {x.get('delivery_state')}] "
                    f"{x.get('cleaned_text', '')}{suffix}"
                )
                self._add(Record(
                    id=x["id"], record_id=x["id"], source="dictation",
                    delivery_time=parse_dt(x["timestamp"]), text=text,
                    metadata=x, original_text=text,
                ))

        name_by_id = {u["id"]: u.get("real_name") or u.get("name") or u["id"] for u in self.users}
        chan_by_id = {c["id"]: c.get("name") or c["id"] for c in self.channels}

        # Slack messages and temporal events.
        p = d / "connectors/slack/messages.jsonl"
        if p.exists():
            for line in p.read_text().splitlines():
                if not line.strip():
                    continue
                x = json.loads(line)
                dt = parse_dt(x["ts"])
                subtype = x.get("subtype")
                where = chan_by_id.get(x.get("channel_id"), x.get("channel_id", "slack"))
                if subtype == "message_deleted":
                    self._deleted[x["target_id"]] = dt
                    continue  # deletion event is a state transition, not useful memory
                if subtype == "message_changed":
                    self._edits.setdefault(x["target_id"], []).append((dt, x.get("text", "")))
                    text = f"[Slack {where}, edit of {x['target_id']}] {x.get('text', '')}"
                    self._add(Record(
                        id=x["id"], record_id=x["id"], source="slack_edit",
                        delivery_time=dt, text=text, metadata=x, original_text=text,
                    ))
                    continue
                who = name_by_id.get(x.get("user"), x.get("bot_name") or x.get("user", "unknown"))
                text = f"[Slack {where}] {who}: {x.get('text', '')}"
                self._add(Record(
                    id=x["id"], record_id=x["id"], source="slack",
                    delivery_time=dt, text=text, metadata=x, original_text=text,
                ))

        # Gmail.
        p = d / "connectors/gmail/messages.jsonl"
        if p.exists():
            for line in p.read_text().splitlines():
                if not line.strip():
                    continue
                x = json.loads(line)
                cc = ", ".join(x.get("cc", []))
                text = (
                    f"[Email {x['date'][:16]}] From {x.get('from','')} "
                    f"To {', '.join(x.get('to', []))}"
                    f"{' Cc ' + cc if cc else ''} | {x.get('subject', '')}\n{x.get('body', '')}"
                )
                self._add(Record(
                    id=x["id"], record_id=x["id"], source="gmail",
                    delivery_time=parse_dt(x["date"]), text=text, metadata=x, original_text=text,
                ))

        # Calendar. Current-state record, available at `updated`.
        p = d / "connectors/google_calendar/events.jsonl"
        if p.exists():
            for line in p.read_text().splitlines():
                if not line.strip():
                    continue
                x = json.loads(line)
                self.events[x["id"]] = x
                st, en = x["start"], x["end"]
                when = f"{st.get('dateTime') or st.get('date')} to {en.get('dateTime') or en.get('date')}"
                att = ", ".join(a["email"] for a in x.get("attendees", []))
                text = (
                    f"[Calendar, {x.get('status')}] {x.get('summary')} | {when} | "
                    f"{x.get('location') or ''} | attendees: {att} | {x.get('description') or ''}"
                    f"{' | repeats ' + str(x['recurrence']) if x.get('recurrence') else ''}"
                )
                self._add(Record(
                    id=x["id"], record_id=x["id"], source="calendar",
                    delivery_time=parse_dt(x["updated"]), text=text, metadata=x, original_text=text,
                ))

        # Codex sessions: one addressable session id, assembled from events.
        for f in sorted((d / "connectors/codex/sessions").glob("*.jsonl")):
            events = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
            if not events:
                continue
            meta, body = events[0], events[1:]
            text = "\n".join(
                f"{e.get('role', e.get('tool', e.get('type')))}: "
                f"{e.get('content') or e.get('input', '') or e.get('output', '')}"
                for e in body
            )
            delivery = parse_dt(body[-1]["timestamp"] if body else meta["started_at"])
            full = f"[Codex session, repo {meta.get('repo')}]\n{text}"
            self._add(Record(
                id=meta["id"], record_id=meta["id"], source="codex",
                delivery_time=delivery, text=full, metadata=meta, original_text=full,
            ))

        # ChatGPT conversations: each message is independently addressable.
        p = d / "connectors/chatgpt/conversations.json"
        if p.exists():
            for c in json.loads(p.read_text()):
                for m in c.get("messages", []):
                    text = f"[ChatGPT '{c.get('title','')}'] {m.get('role')}: {m.get('content','')}"
                    self._add(Record(
                        id=m["id"], record_id=c["id"], source="chatgpt",
                        delivery_time=parse_dt(m["create_time"]), text=text,
                        metadata={"conversation": c.get("title"), "role": m.get("role")}, original_text=text,
                    ))

        self._deleted = {k: v for k, v in self._deleted.items()}
        # Use a stable sort to make output deterministic.
        self.records.sort(key=lambda r: (r.delivery_time, r.id))

    def _effective_text(self, rec: Record, as_of: datetime) -> tuple[str, datetime | None]:
        # Apply the latest edit known at as_of. Edits replace content of target records.
        edits = [
            (t, txt) for t, txt in getattr(self, "_edits", {}).get(rec.id, [])
            if t <= as_of
        ]
        if not edits:
            return redact_text(rec.original_text), None
        edits.sort(key=lambda p: p[0])
        txt = redact_text(edits[-1][1])
        prefix = rec.original_text.partition(": ")[0]
        return redact_text(f"{prefix}: {txt} (edited)"), edits[-1][0]

    def get_valid_corpus(self, as_of: str | datetime) -> list[Record]:
        cutoff = parse_dt(as_of)
        out: list[Record] = []
        for rec in self.records:
            if rec.delivery_time > cutoff:
                continue
            deletion = self._deleted.get(rec.id)
            if deletion is not None and deletion <= cutoff:
                continue
            # Targets of edits/deletions are ordinary records; edit events themselves
            # remain searchable.
            text, edited_at = self._effective_text(rec, cutoff)
            if not text.strip():
                continue
            out.append(Record(
                id=rec.id, record_id=rec.record_id, source=rec.source,
                delivery_time=rec.delivery_time, text=text, metadata=rec.metadata,
                original_text=rec.original_text, edited_at=edited_at,
            ))
        return out

    # ------------------------------------------------------------------
    # Entities are derived from the data itself (directory + mined phrases);
    # nothing here knows about any particular question or storyline.
    # ------------------------------------------------------------------
    def entity_aliases(self) -> dict[str, set[str]]:
        """alias-set per entity key: people, channels and recurring proper-noun phrases."""
        if getattr(self, "_entities", None) is not None:
            return self._entities
        aliases: dict[str, set[str]] = {}
        for u in self.users:
            if u.get("is_bot"):
                continue
            vals = {str(u.get(k) or "") for k in ("id", "name", "real_name", "email")}
            if u.get("email"):
                vals.add(u["email"].split("@")[0])
            aliases[u.get("id", "")] = {v.lower() for v in vals if v}
        for c in self.channels:
            if c.get("is_dm"):
                continue
            nm = str(c.get("name") or "").lower()
            aliases[c.get("id", "")] = {nm, "#" + nm, nm.replace("-", " ")}
        for phrase in self._mine_phrases():
            aliases["phrase:" + phrase] = {phrase}
        self._entities = aliases
        return aliases

    _PHRASE = re.compile(r"\b[A-Z][\w&'-]*(?:[ \t]+(?:[A-Z][\w&'-]*|v\d+))+")
    _NOT_NAME = frozenset("hi hello hey dear thanks from to cc bcc subject re fwd and but so can could should would will add update "
                          "begin end order guests organizer when where who what why how the".split()) | set(textutil.WEEKDAYS) | \
        {w[:3] for w in textutil.WEEKDAYS} | set(textutil.MONTHS) | {m[:3] for m in textutil.MONTHS}

    def _mine_phrases(self, min_records: int = 3) -> set[str]:
        """Recurring multi-word proper nouns (customers, projects, products)."""
        counts: dict[str, set[str]] = {}
        for r in self.records:
            body = r.original_text
            if r.source == "meeting":  # drop the "[Title, date] Speaker:" prefix
                body = body.partition("] ")[2].partition(": ")[2]
            elif r.source in ("slack", "chatgpt", "dictation", "calendar"):
                body = body.partition("] ")[2]
            for m in self._PHRASE.finditer(body):
                words = m.group(0).split()
                if len(words) > 4 or words[0].lower() in self._NOT_NAME or words[-1].lower() in self._NOT_NAME:
                    continue
                counts.setdefault(m.group(0).lower(), set()).add(r.record_id)
        people = {n for u in self.users for n in (str(u.get("real_name") or "").lower(),) if n}
        out = {p for p, ids in counts.items() if len(ids) >= min_records and p not in people}
        # keep the more specific phrase when one contains another with the same support
        return {p for p in out if not any(p != q and p in q and len(counts[q]) >= 0.8 * len(counts[p]) for q in out)}

    def record_dates(self, rec: Record) -> set:
        """Calendar days a record is about (mentioned dates + calendar start)."""
        cache = self.__dict__.setdefault("_dates", {})
        if rec.id not in cache:
            d = set(textutil.dates_in(rec.original_text))
            if rec.source == "calendar":
                st = rec.metadata.get("start", {})
                d |= textutil.dates_in(st.get("dateTime") or st.get("date") or "")
            cache[rec.id] = d
        return cache[rec.id]

    def record_available(self, record_id: str, as_of: datetime) -> bool:
        rec = self._by_id.get(record_id)
        if not rec:
            # Whole-record ids for meetings/conversations are available from their
            # earliest unit.
            times = [r.delivery_time for r in self.records if r.record_id == record_id]
            if not times:
                return False
            t = min(times)
        else:
            t = rec.delivery_time
        return t <= as_of and not (
            record_id in self._deleted and self._deleted[record_id] <= as_of
        )


def load_corpus(data_dir: str | os.PathLike[str]) -> CorpusIndex:
    return CorpusIndex(data_dir)
