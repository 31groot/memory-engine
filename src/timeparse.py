"""Natural-language dates, times and durations relative to an `as_of` moment (Alex's local time)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta

from . import textutil
from .indexer import LA

_WD = {d: i for i, d in enumerate(textutil.WEEKDAYS)}
_WD.update({d[:3]: i for i, d in enumerate(textutil.WEEKDAYS)})
_NUM = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "ten": 10, "fifteen": 15,
        "twenty": 20, "thirty": 30, "forty-five": 45, "sixty": 60}


@dataclass
class When:
    day: date | None = None
    at: time | None = None
    duration: timedelta | None = None
    spans: list[tuple[int, int]] = field(default_factory=list)


def _n(tok: str) -> int | None:
    tok = tok.lower()
    return int(tok) if tok.isdigit() else _NUM.get(tok)


def bare_hour(h: int) -> int:
    """'at 2' in a work context means 2pm, 'at 9' means 9am, 'at 12' is noon."""
    return h + 12 if 1 <= h <= 7 else h


def parse_when(text: str, as_of: datetime) -> When:
    base = as_of.astimezone(LA)
    today = base.date()
    w = When()
    t = text

    def take(m: re.Match) -> None:
        w.spans.append(m.span())

    # --- duration ---------------------------------------------------------------------------
    m = re.search(r"\b(half an hour|an hour and a half|(\d+|an?|one|two|three|four|five|ten|fifteen|twenty|thirty|forty-five|sixty)[\s-]*(hours?|hrs?|minutes?|mins?))\b(?!\s+(?:before|after))", t, re.I)
    if m:
        s = m.group(1).lower()
        if s.startswith("half"):
            w.duration = timedelta(minutes=30)
        elif s.startswith("an hour and"):
            w.duration = timedelta(minutes=90)
        else:
            n = _n(m.group(2)) or 1
            w.duration = timedelta(hours=n) if m.group(3).lower().startswith(("h")) else timedelta(minutes=n)
        take(m)

    # --- day --------------------------------------------------------------------------------
    m = re.search(r"\b(tomorrow|today|day after tomorrow)\b", t, re.I)
    if m:
        w.day = today + timedelta(days={"tomorrow": 1, "today": 0}.get(m.group(1).lower(), 2))
        take(m)
    if w.day is None:
        m = re.search(r"\b(?:(next|this|on)\s+)?(monday|tuesday|wednesday|thursday|friday|saturday|sunday|mon|tue|tues|wed|thu|thur|thurs|fri|sat|sun)\b", t, re.I)
        if m:
            target = _WD[m.group(2).lower()[:3] if m.group(2).lower()[:3] in _WD else m.group(2).lower()]
            delta = (target - today.weekday()) % 7 or 7  # always the next such weekday after today
            w.day = today + timedelta(days=delta)
            take(m)
    if w.day is None:
        ds = textutil.DATE_MD.search(t)
        if ds:
            try:
                d = date(int(ds.group(3) or today.year), textutil.MONTH_NUM[ds.group(1).lower().rstrip(".")], int(ds.group(2)))
                w.day = d if (ds.group(3) or d >= today) else d.replace(year=d.year + 1)
                take(ds)
            except ValueError:
                pass
    if w.day is None:
        m = re.search(r"\b(?:on\s+)?the\s+(\d{1,2})(?:st|nd|rd|th)\b|\bon\s+(\d{1,2})(?:st|nd|rd|th)\b", t, re.I)
        if m:
            dom = int(m.group(1) or m.group(2))
            y, mo = today.year, today.month
            if dom < today.day:
                mo, y = (1, y + 1) if mo == 12 else (mo + 1, y)
            try:
                w.day = date(y, mo, dom)
                take(m)
            except ValueError:
                pass

    # --- time of day ------------------------------------------------------------------------
    m = re.search(r"\b(?:at\s+)?(noon|midnight)\b", t, re.I)
    if m:
        w.at = time(12 if m.group(1).lower() == "noon" else 0, 0)
        take(m)
    else:
        m = re.search(r"\b(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)", t, re.I)
        if not m:
            m = re.search(r"\bat\s+(\d{1,2})(?::(\d{2}))?(?!\d)(?!\s*(?:minutes?|mins?|hours?))()", t, re.I)
        if m:
            h, mi = int(m.group(1)), int(m.group(2) or 0)
            ap = (m.group(3) or "").lower().replace(".", "")
            if ap == "pm" and h < 12:
                h += 12
            elif ap == "am" and h == 12:
                h = 0
            elif not ap:
                h = bare_hour(h)
            if 0 <= h < 24:
                w.at = time(h, mi)
                take(m)
    return w


def strip_spans(text: str, spans: list[tuple[int, int]]) -> str:
    out, last = [], 0
    for a, b in sorted(spans):
        out.append(text[last:a])
        last = max(last, b)
    out.append(text[last:])
    return re.sub(r"\s{2,}", " ", "".join(out)).strip(" ,.;")


def combine(day: date, at: time) -> datetime:
    return datetime(day.year, day.month, day.day, at.hour, at.minute, tzinfo=LA)


OFFSET_BEFORE = re.compile(
    r"\b(?P<n>\d+|an?|one|two|three|four|five|ten|fifteen|twenty|thirty|forty-five|sixty|half an?)\s*"
    r"(?P<unit>hours?|hrs?|minutes?|mins?|days?)\s+(?P<dir>before|after|ahead of|prior to)\s+(?:the\s+)?"
    r"(?P<event>.+?)(?=\s+to\s+|\s+so\s+|\s+and\s+|$)", re.I)


def parse_offset(text: str) -> tuple[timedelta, str, re.Match] | None:
    m = OFFSET_BEFORE.search(text)
    if not m:
        return None
    raw = m.group("n").lower()
    n = 0.5 if raw.startswith("half") else (_n(raw) or 1)
    unit = m.group("unit").lower()
    delta = timedelta(days=n) if unit.startswith("d") else timedelta(hours=n) if unit.startswith("h") else timedelta(minutes=n)
    if m.group("dir").lower() in ("after",):
        pass
    return (delta if m.group("dir").lower() == "after" else -delta), m.group("event").strip(), m
