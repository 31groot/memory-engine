"""Small, dependency-free text helpers shared by retrieval, answering and actions."""
from __future__ import annotations

import re
from datetime import date

MONTHS = ["january", "february", "march", "april", "may", "june", "july",
          "august", "september", "october", "november", "december"]
MONTH_NUM = {m: i + 1 for i, m in enumerate(MONTHS)}
MONTH_NUM.update({m[:3]: i + 1 for i, m in enumerate(MONTHS)})
MONTH_NUM["sept"] = 9
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

STOP = frozenset("""
a an and are as at be been being but by can could did do does for from had has have he her his how i if in into is it its
me my of on or our she so than that the their them then there these they this to us was we were what when where which who
whom whose why will with would you your about again still really just also any some all more most very
""".split())

# Words that frame a question but do not say what it is about.
FRAME = frozenset("""
tell say said says told mention mentioned ask asked remind know remember recall find show give
much many long often now currently today get got go going
""".split())

_TOKEN = re.compile(r"[a-z0-9]+(?:[.'’/:@#_+-][a-z0-9]+)*")


def stem(tok: str) -> str:
    t = tok.replace("’", "'")
    if t.endswith("'s"):
        t = t[:-2]
    if len(t) <= 3 or any(ch.isdigit() for ch in t):
        return t
    for suf, rep in (("ies", "y"), ("ing", ""), ("ed", ""), ("es", ""), ("ly", ""), ("s", "")):
        if t.endswith(suf) and len(t) - len(suf) >= 3:
            if suf == "s" and t.endswith("ss"):
                return t
            t = t[: len(t) - len(suf)] + rep
            break
    if len(t) > 3 and t.endswith("e"):
        t = t[:-1]
    if len(t) > 3 and t[-1] == t[-2] and t[-1] not in "aeiouls":
        t = t[:-1]
    return t


def raw_tokens(text: str) -> list[str]:
    return _TOKEN.findall((text or "").lower().replace("’", "'"))


def tokens(text: str, keep_stop: bool = False) -> list[str]:
    """Stemmed tokens; compound tokens ("go/no-go", "p95") also emit their parts."""
    out: list[str] = []
    for tok in raw_tokens(text):
        parts = [tok]
        if re.search(r"[./:@#_+-]", tok):
            parts += [p for p in re.split(r"[./:@#_+'-]+", tok) if p]
        for p in parts:
            base = p[:-2] if p.endswith("'s") else p
            if not keep_stop and base in STOP:
                continue
            out.append(stem(p))
    return out


def split_sentences(text: str) -> list[str]:
    parts: list[str] = []
    for line in re.split(r"\n+", (text or "").strip()):
        line = line.strip(" -•|")
        if not line:
            continue
        for p in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\[\"'(])", line):
            p = p.strip()
            if p:
                parts.append(p)
    return parts


_MD = r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
DATE_MD = re.compile(rf"\b({_MD})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?\b(?:,?\s+(\d{{4}}))?", re.I)
DATE_ISO = re.compile(r"(?<!\d)(\d{4})-(\d{2})-(\d{2})(?!\d)")
DATE_SLASH = re.compile(r"\b(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?\b")


def dates_in(text: str, default_year: int = 2026) -> set[date]:
    out: set[date] = set()
    t = text or ""
    for m in DATE_ISO.finditer(t):
        try:
            out.add(date(int(m.group(1)), int(m.group(2)), int(m.group(3))))
        except ValueError:
            pass
    for m in DATE_MD.finditer(t):
        try:
            out.add(date(int(m.group(3) or default_year), MONTH_NUM[m.group(1).lower().rstrip(".")], int(m.group(2))))
        except (ValueError, KeyError):
            pass
    for m in DATE_SLASH.finditer(t):
        mo, d = int(m.group(1)), int(m.group(2))
        if 1 <= mo <= 12 and 1 <= d <= 31:
            try:
                y = int(m.group(3)) if m.group(3) else default_year
                out.add(date(y + 2000 if y < 100 else y, mo, d))
            except ValueError:
                pass
    return out


VALUE_RE = re.compile(
    rf"\$\s?\d[\d,]*(?:\.\d+)?[kKmM]?|\b\d+(?:\.\d+)?\s?(?:%|percent|ms|milliseconds?|seconds?|days?|weeks?|hours?|minutes?)\b"
    rf"|\b{_MD}\.?\s+\d{{1,2}}\b|\b\d{{1,2}}/\d{{1,2}}\b|\b\d{{1,2}}(?::\d{{2}})?\s?(?:am|pm)\b|\b\d+\s*(?:of|/)\s*\d+\b",
    re.I,
)


def values_in(text: str) -> set[str]:
    return {re.sub(r"\s+", " ", m.group(0).lower()) for m in VALUE_RE.finditer(text or "")}
