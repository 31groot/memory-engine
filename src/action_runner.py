"""Dry-run action planning.

A command is turned into a list of structured actions; nothing is executed. The planner is generic:
people, channels and events are resolved from the connector data, times are parsed relative to `as_of`,
and message content comes from the command itself (or from memory when the command refers to a fact,
e.g. "the corrected NRR"). Destructive requests become `confirm`; unresolved or ambiguous references
become `clarify` instead of a guess.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from . import textutil
from .contacts import Directory, Person
from .indexer import CorpusIndex, LA, parse_dt
from .timeparse import combine, parse_offset, parse_when, strip_spans

ACTION_VERBS = r"(?:thank|tell|message|email|e-mail|slack|dm|ping|remind|book|schedule|set up|ask|send|open|delete|remove|move|reschedule|push|cancel|let)"
POLITE = re.compile(r"^(?:please|hey|ok(?:ay)?|can you|could you|would you|will you|i need you to|i want you to|go ahead and)[,\s]+", re.I)
QUESTION_START = re.compile(r"^(what|when|where|who|whom|whose|why|how|which|did|do|does|is|are|was|were|has|have|had|am|can i|could i)\b", re.I)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _iso(dt: datetime) -> str:
    return dt.astimezone(LA).isoformat()


def clarify(q: str) -> list[dict]:
    return [{"type": "clarify", "args": {"question": q}}]


# ----------------------------------------------------------------------------- context
class Planner:
    def __init__(self, data_dir: str, engine=None):
        self.corpus = CorpusIndex(data_dir)
        self.dir = Directory(self.corpus)
        self.engine = engine  # optional MemoryEngine: used to resolve "the corrected NRR"-style references

    # ---- lookups -------------------------------------------------------------------------
    def find_channel(self, phrase: str) -> dict | None:
        p = re.sub(r"[#]|\bchannel\b|\bthe\b", "", phrase.lower()).strip().replace(" ", "-")
        for c in self.corpus.channels:
            if not c.get("is_dm") and p in {c["name"].lower(), c["id"].lower()}:
                return c
        for c in self.corpus.channels:
            if not c.get("is_dm") and p and (p in c["name"].lower() or c["name"].lower() in p):
                return c
        return None

    def find_event(self, phrase: str, as_of: datetime) -> dict | None:
        want = set(textutil.tokens(phrase))
        if not want:
            return None
        best, best_key = None, None
        for e in self.corpus.events.values():
            if parse_dt(e["updated"]) > as_of or e.get("status") == "cancelled":
                continue
            summ = set(textutil.tokens(e.get("summary") or ""))
            extra = set(textutil.tokens(f"{e.get('description') or ''} {e.get('location') or ''}"))
            hit = want & summ
            if not hit and not (want & extra):
                continue
            score = 2 * len(hit) / max(1, len(summ)) + len(hit) / len(want) + 0.2 * len(want & extra) / len(want)
            st = parse_dt((e.get("start") or {}).get("dateTime") or (e.get("start") or {}).get("date"))
            upcoming = st >= as_of
            key = (round(score, 3), upcoming, -abs((st - as_of).total_seconds()))
            if best_key is None or key > best_key:
                best, best_key = e, key
        return best

    def resolve_people(self, names: list[str], via: str | None, as_of: datetime) -> tuple[list[Person], list[dict] | None]:
        out: list[Person] = []
        for n in names:
            hits = self.dir.find(n, via)
            if not hits and via:
                hits = self.dir.find(n)  # reachable some other way
            if not hits:
                return [], clarify(f"I couldn't find anyone called {n.strip().title()}. Who do you mean?")
            if len(hits) > 1:
                opts = " or ".join(p.name for p in hits[:3])
                return [], clarify(f"Which {n.strip().title()} do you mean: {opts}?")
            out.append(hits[0])
        return out, None

    # ---- entry ---------------------------------------------------------------------------
    def plan(self, command: str, as_of_str: str) -> list[dict]:
        as_of = parse_dt(as_of_str)
        c = POLITE.sub("", command.strip()).strip()
        if not c:
            return clarify("What would you like me to do?")
        if self._is_question(c):
            return [{"type": "memory.ask", "args": {"question": command.strip()}}]
        clauses = self._split(c)
        actions: list[dict] = []
        for cl in clauses:
            res = self._one(cl, as_of, whole=c)
            if res and res[0]["type"] == "clarify":
                return res  # never act on part of a command we could not fully resolve
            actions.extend(res or [])
        return actions or clarify("I'm not sure what action you want. Could you rephrase?")

    @staticmethod
    def _is_question(c: str) -> bool:
        if re.match(rf"^{ACTION_VERBS}\b", c, re.I):
            return False
        return c.rstrip().endswith("?") or bool(QUESTION_START.match(c))

    def _split(self, c: str) -> list[str]:
        """Split "A and B" into two actions only when B starts a new action: a verb that always starts
        one, or a messaging verb aimed at someone/some channel ("thank Ben ...", not "ask if she ...")."""
        always = r"(?:remind|book|schedule|set up|open|delete|remove|move|reschedule|push|cancel|email|e-mail|slack|dm)"
        directed = r"(?:thank|tell|message|ping|ask|send|let|notify)"
        pieces = re.split(r"(\s*(?:,\s*and|,|;|\band then\b|\bthen\b|\band\b)\s+)", c, flags=re.I)
        out, cur = [], pieces[0]
        for sep, nxt in zip(pieces[1::2], pieces[2::2]):
            first = nxt.split(None, 2)
            ok = False
            if first and re.match(rf"^{always}$", first[0], re.I):
                ok = True
            elif first and re.match(rf"^{directed}$", first[0], re.I) and len(first) > 1:
                target = first[1].strip(",")
                ok = bool(self.dir.find(target)) or target.lower() == "the" and "channel" in nxt.lower() or target.startswith("#")
            if ok:
                out.append(cur.strip())
                cur = nxt
            else:
                cur += sep + nxt
        out.append(cur.strip())
        return [p for p in out if p]

    def _one(self, c: str, as_of: datetime, whole: str) -> list[dict]:
        low = c.lower()
        if re.match(r"^(delete|remove|erase|wipe|trash|purge)\b", low):
            return self._destructive(c)
        if re.match(r"^cancel\b", low):
            return self._cancel(c, as_of)
        m = re.match(r"^open\s+(.+)$", c, re.I)
        if m:
            app = re.sub(r"^(?:the\s+)?|\s+app$", "", m.group(1).strip().rstrip(".!"), flags=re.I)
            return [{"type": "app.open", "args": {"app": app}}]
        if low.startswith("remind"):
            return self._remind(c, as_of)
        if re.match(r"^(book|schedule|set up|create|add|arrange|put)\b", low) and re.search(r"\b(meeting|call|time|sync|1:1|catch[- ]?up|minutes?|hour|with)\b", low):
            return self._create_event(c, as_of)
        if re.match(r"^(move|reschedule|push|shift|postpone)\b", low):
            return self._update_event(c, as_of)
        if re.match(r"^(e-?mail|mail)\b", low) or re.search(r"^send\b.*\be-?mail\b", low):
            return self._email(c, as_of)
        if re.match(r"^(slack|dm|message|ping|tell|thank|notify|let|ask|send)\b", low):
            return self._slack(c, as_of)
        return clarify(f"I'm not sure what to do with: {c}")

    # ---- destructive ---------------------------------------------------------------------
    def _destructive(self, c: str) -> list[dict]:
        m = re.search(r"\bfrom\s+([A-Z][\w.'-]+(?:\s+[A-Z][\w.'-]+)?)", c)
        what = re.sub(r"^\s*\w+\s+", "", c).rstrip(".!? ")
        if m:
            hits = self.dir.find(m.group(1))
            if len(hits) == 1:
                what = what.replace(m.group(1), hits[0].name)
        verb = c.split()[0].capitalize()
        return [{"type": "confirm", "args": {"summary": f"{verb} {what}? This can't be undone."}}]

    def _cancel(self, c: str, as_of: datetime) -> list[dict]:
        phrase = re.sub(r"^cancel\s+(?:my\s+|the\s+)?", "", c, flags=re.I)
        ev = self.find_event(phrase, as_of)
        if not ev:
            return clarify(f"Which event should I cancel? I couldn't find “{phrase}”.")
        return [{"type": "confirm", "args": {"summary": f"Cancel “{ev['summary']}” (event {ev['id']})? Attendees will be notified.",
                                              "event_id": ev["id"]}}]

    # ---- reminders -----------------------------------------------------------------------
    def _remind(self, c: str, as_of: datetime) -> list[dict]:
        rest = re.sub(r"^remind\s+me\s*", "", c, flags=re.I)
        off = parse_offset(rest)
        if off:
            delta, phrase, m = off
            ev = self.find_event(phrase, as_of)
            if not ev:
                return clarify(f"Which event do you mean by “{phrase}”?")
            start = parse_dt(ev["start"].get("dateTime") or ev["start"]["date"] + "T09:00:00")
            task = strip_spans(rest, [m.span()])
            due = start + delta
        else:
            w = parse_when(rest, as_of)
            if w.day is None and w.at is None:
                return clarify("When should I remind you?")
            day = w.day or (as_of.astimezone(LA).date() + (timedelta(days=0) if w.at and combine(as_of.astimezone(LA).date(), w.at) > as_of else timedelta(days=1)))
            due = combine(day, w.at or __import__("datetime").time(9, 0))
            task = strip_spans(rest, w.spans)
        task = re.sub(r"^(?:to|that|about)\s+", "", task.strip(), flags=re.I).strip(" ,.")
        if not task:
            return clarify("What should the reminder say?")
        return [{"type": "reminder.create", "args": {"text": task, "due": _iso(due)}}]

    # ---- calendar ------------------------------------------------------------------------
    def _create_event(self, c: str, as_of: datetime) -> list[dict]:
        w = parse_when(c, as_of)
        if w.day is None:
            return clarify("Which day should I schedule it?")
        if w.at is None:
            return clarify("What time should I schedule it?")
        names: list[str] = []
        m = re.search(r"\bwith\s+(.+?)(?=\s+(?:tomorrow|today|on|at|about|for|to|next|this|monday|tuesday|wednesday|thursday|friday|saturday|sunday|\d)\b|$)", c, re.I)
        if m:
            names = [n.strip() for n in re.split(r"\s*(?:,|\band\b)\s*", m.group(1)) if n.strip()]
        people, bad = self.resolve_people(names, "email", as_of)
        if bad:
            return bad
        topic = None
        t = re.search(r"\b(?:about|re:?|regarding|to discuss|to talk about|to go over|to review|to cover|to walk through|for)\s+(.+)$", c, re.I)
        if t:
            topic = strip_spans(t.group(1), [])
            topic = re.sub(r"\b(tomorrow|today)\b.*$", "", topic).strip(" .")
        who = ", ".join(p.first for p in people)
        title = f"{topic[0].upper() + topic[1:]}" if topic else (f"Meeting with {who}" if who else "Meeting")
        if topic and who:
            title = f"{who} / Alex: {topic}"
        start = combine(w.day, w.at)
        end = start + (w.duration or timedelta(minutes=60))
        return [{"type": "calendar.create_event", "args": {
            "title": title, "start": _iso(start), "end": _iso(end),
            "attendees": [p.email for p in people if p.email]}}]

    def _update_event(self, c: str, as_of: datetime) -> list[dict]:
        m = re.match(r"^(?:move|reschedule|push|shift|postpone)\s+(?:my\s+|the\s+)?(?P<ev>.+?)\s+(?:to|until|till|for)\s+(?P<when>.+)$", c, re.I)
        shift = re.match(r"^(?:move|push|shift|postpone)\s+(?:my\s+|the\s+)?(?P<ev>.+?)\s+(?:back|forward|ahead|later|earlier|by)\s+(?:by\s+)?(?P<n>\d+|an?|one|two|half an?)\s*(?P<u>hours?|minutes?|mins?|days?)\b", c, re.I)
        if shift:
            ev = self.find_event(shift.group("ev"), as_of)
            if not ev:
                return clarify(f"Which event do you mean by “{shift.group('ev')}”?")
            raw = shift.group("n").lower()
            n = 0.5 if raw.startswith("half") else (int(raw) if raw.isdigit() else {"a": 1, "an": 1, "one": 1, "two": 2}[raw])
            unit = shift.group("u").lower()
            d = timedelta(days=n) if unit.startswith("d") else timedelta(hours=n) if unit.startswith("h") else timedelta(minutes=n)
            if re.search(r"\b(earlier|ahead)\b", c, re.I):
                d = -d
            st, en = parse_dt(ev["start"]["dateTime"]), parse_dt(ev["end"]["dateTime"])
            return [{"type": "calendar.update_event", "args": {"event_id": ev["id"], "start": _iso(st + d), "end": _iso(en + d)}}]
        if not m:
            return clarify("What should I move, and to when?")
        ev = self.find_event(m.group("ev"), as_of)
        if not ev:
            return clarify(f"Which event do you mean by “{m.group('ev')}”?")
        w = parse_when(m.group("when"), as_of)
        if w.day is None and w.at is None:
            return clarify("What new day or time should I use?")
        st, en = parse_dt(ev["start"]["dateTime"]).astimezone(LA), parse_dt(ev["end"]["dateTime"]).astimezone(LA)
        new = combine(w.day or st.date(), w.at or st.timetz().replace(tzinfo=None))
        return [{"type": "calendar.update_event", "args": {"event_id": ev["id"], "start": _iso(new), "end": _iso(new + (en - st))}}]

    # ---- messaging -----------------------------------------------------------------------
    def _take_recipients(self, text: str, via: str | None, as_of: datetime):
        """Longest-prefix match of known people at the start of `text` ('Sarah Patel and ...', 'John the ...')."""
        toks = text.split()
        names, i = [], 0
        while i < len(toks):
            for span in (2, 1):
                cand = " ".join(toks[i:i + span]).strip(",")
                if cand and self.dir.find(cand, via if via else None) and (span == 1 or len(cand.split()) == 2):
                    # a two-word candidate must be a real full-name hit, not first-name + next word
                    if span == 2 and not any(cand.lower() == p.name.lower() for p in self.dir.people):
                        continue
                    names.append(cand)
                    i += span
                    break
            else:
                break
            if i < len(toks) and toks[i].lower() == "and" and i + 1 < len(toks) and self.dir.find(toks[i + 1].strip(","), via):
                i += 1
                continue
            break
        return names, " ".join(toks[i:])

    def _compose(self, payload: str, recipient: Person | None, as_of: datetime) -> str:
        p = payload.strip().rstrip(".")
        p = re.sub(r"^(?:and\s+)?", "", p, flags=re.I)
        m = re.match(r"^(?:ask|check)\s+(?:him|her|them)?\s*(?:if|whether)\s+(.*)$", p, re.I)
        if m:
            return _question_to_you(m.group(1))
        m = re.match(r"^ask\s+(?:him|her|them)?\s*(?:about|for)\s+(.*)$", p, re.I)
        if m:
            return f"Could you share {m.group(1)}?"
        m = re.match(r"^(?:tell|let)\s+(?:him|her|them)?\s*(?:know\s+)?(?:that\s+)?(.*)$", p, re.I)
        if m:
            p = m.group(1)
        p = re.sub(r"^(?:that|to say|saying)\s+", "", p, flags=re.I)
        m = re.match(r"^thank(?:\s+(?:you|\w+))?(?:\s+for\s+(.*))?$", p, re.I)
        if m:
            return f"Thanks{' for ' + m.group(1) if m.group(1) else ''}!"
        if re.match(r"^(?:the|our|my|a|an)\s+[\w\s'-]{2,40}$", p, re.I) and self.engine is not None:
            fact = self._lookup_fact(p, as_of)
            if fact:
                return f"{p[0].upper() + p[1:]}: {fact}"
        return (p[0].upper() + p[1:] + ".") if p else ""

    def _lookup_fact(self, phrase: str, as_of: datetime) -> str | None:
        """Resolve a noun phrase ('the corrected NRR') to its current value using memory: among sentences that
        mention the topic and carry a value, prefer correction wording when the phrase asks for it,
        otherwise the most recent one."""
        eng = self.engine
        q = re.sub(r"^(?:the|our|my)\s+", "", phrase, flags=re.I)
        wants_fix = bool(re.search(r"\b(corrected|updated|latest|current|new|final|fixed)\b", q, re.I))
        topic = re.sub(r"\b(?:corrected|updated|latest|current|new|final|fixed)\b\s*", "", q, flags=re.I).strip()
        terms = set(textutil.tokens(topic))
        hits = eng.retriever.search(f"{topic}", as_of.isoformat(), 20)
        cue = re.compile(r"\b(not|fix(?:ed)?|correct(?:ed|ion)?|actually|now|update[sd]?|instead|after)\b", re.I)
        cands = []
        for h in hits:
            body = h.record.text.partition("] ")[2] or h.record.text
            body = re.sub(r"^[A-Z][\w .'-]{1,30}:\s+", "", body)  # drop the "Speaker:" prefix
            for sent in textutil.split_sentences(body):
                if terms and terms <= set(textutil.tokens(sent)) and textutil.values_in(sent) | set(re.findall(r"\b\d+\b", sent)):
                    cands.append((bool(cue.search(sent)) if wants_fix else True, h.record.delivery_time, sent))
        if not cands:
            return None
        best = max(cands, key=lambda c: (c[0], c[1]))[2]
        words = best.split()
        return " ".join(words[:30]).rstrip(" ,;") + ("" if best.endswith(".") else ".")

    def _subject_for(self, person: Person, payload: str) -> str:
        want = {t for t in textutil.tokens(payload) if len(t) > 3}
        best = None
        gm = self.corpus.data_dir / "connectors/gmail/messages.jsonl"
        for line in gm.read_text().splitlines():
            if not line.strip():
                continue
            m = json.loads(line)
            addrs = " ".join([m.get("from", "")] + list(m.get("to", [])) + list(m.get("cc", []))).lower()
            if person.email and person.email.lower() in addrs and want & set(textutil.tokens(m.get("subject", ""))):
                best = m
        if best:
            return "Re: " + re.sub(r"^(?:re:\s*)+", "", best["subject"], flags=re.I)
        words = [w for w in re.findall(r"[\w'-]+", payload) if w.lower() not in textutil.STOP][:5]
        return " ".join(words).capitalize() or "Following up"

    def _email(self, c: str, as_of: datetime) -> list[dict]:
        rest = re.sub(r"^(?:send\s+(?:an?\s+)?)?(?:e-?mail|mail)\s*(?:to\s+)?", "", c, flags=re.I)
        names, payload = self._take_recipients(rest, "email", as_of)
        if not names:
            m = re.match(r"^([A-Za-z.'-]+)", rest)
            return clarify(f"Who should I email{' (' + m.group(1) + ')' if m else ''}? I couldn't find that person.")
        people, bad = self.resolve_people(names, "email", as_of)
        if bad:
            return bad
        body_core = self._compose(payload, people[0], as_of)
        if not body_core:
            return clarify("What should the email say?")
        greet = ", ".join(p.first for p in people)
        me = self.dir.me.first if self.dir.me else ""
        body = f"Hi {greet},\n\n{body_core}\n\nThanks,\n{me}".strip()
        return [{"type": "gmail.send", "args": {
            "to": [p.email for p in people], "cc": [],
            "subject": self._subject_for(people[0], payload or body_core), "body": body}}]

    def _slack(self, c: str, as_of: datetime) -> list[dict]:
        explicit_email = bool(re.search(r"\b(by\s+)?e-?mail\b", c, re.I))
        ch = re.search(r"\b(?:the\s+)?(#?[\w-]+(?:\s+[\w-]+)?)\s+channel\b|(#[\w-]+)", c, re.I)
        if ch and not re.search(r"\bon\s+slack\b", c, re.I) or (ch and ch.group(2)):
            chan = self.find_channel(ch.group(2) or ch.group(1))
            if chan:
                payload = re.sub(r"^.*?\bchannel\b\s*", "", c, flags=re.I) if ch.group(1) else re.sub(r"^.*?#[\w-]+\s*", "", c)
                text = self._compose(payload, None, as_of)
                return [{"type": "slack.send_message", "args": {"to": chan["id"], "text": text}}] if text else clarify("What should I post?")
        m = re.match(r"^(?:slack|dm|message|ping|tell|thank|notify|let|ask|send)\s+(?P<rest>.+)$", c, re.I)
        verb = c.split()[0].lower()
        rest = m.group("rest") if m else c
        rest = re.sub(r"^(?:a\s+)?(?:message|dm)\s+to\s+", "", rest, flags=re.I)
        names, payload = self._take_recipients(rest, None, as_of)
        if not names:
            return clarify("Who should I message?")
        via = "slack" if re.search(r"\bslack\b|\bdm\b", c, re.I) or verb in ("slack", "dm") else ("email" if explicit_email else None)
        payload = re.sub(r"^(?:on|via|in)\s+slack\s*", "", payload, flags=re.I)
        people, bad = self.resolve_people(names, via, as_of)
        if bad:
            return bad
        if verb == "thank":
            fm = re.search(r"\bfor\s+(.+)$", payload, re.I)
            text = f"Thanks {people[0].first}{' for ' + fm.group(1).rstrip('.') if fm else ''}!"
        else:
            p = payload
            if verb in ("ask",):
                p = "ask " + p
            text = self._compose(p if p else "", people[0], as_of)
            if re.match(r"^about\s", payload, re.I):
                text = f"Hi {people[0].first}, quick note {payload.strip().rstrip('.')}."
        if not text:
            return clarify("What should the message say?")
        person = people[0]
        if via == "email" or (not person.slack_id and person.email):
            return [{"type": "gmail.send", "args": {"to": [person.email], "cc": [], "subject": self._subject_for(person, payload), "body": f"Hi {person.first},\n\n{text}"}}]
        return [{"type": "slack.send_message", "args": {"to": person.slack_id, "text": text}}]


def _question_to_you(clause: str) -> str:
    cl = clause.strip().rstrip(".?")
    m = re.match(r"^(?:he|she|they)(?:'s|\s+has|\s+have|'ve)\s+(going\s+.*)$", cl, re.I)
    if m:
        return f"Are you {m.group(1)}?"
    m = re.match(r"^(?:he|she|they)(?:'s|\s+has|\s+have|'ve)\s+(.*)$", cl, re.I)
    if m:
        return f"Have you {m.group(1)}?"
    m = re.match(r"^(?:he|she|they)\s+(can|could|will|would|should|did|does|do)\s+(.*)$", cl, re.I)
    if m:
        return f"{m.group(1).capitalize()} you {m.group(2)}?"
    m = re.match(r"^(?:he|she|they)\s+(?:is|are|was|were)\s+(.*)$", cl, re.I)
    if m:
        return f"Are you {m.group(1)}?"
    cl = re.sub(r"\b(?:he|she|they)\b", "you", cl, flags=re.I)
    return f"Could you confirm whether {cl}?"


def run_actions(input_path: str, output_path: str, data_dir: str) -> None:
    from .memory_runner import MemoryEngine
    engine = MemoryEngine(data_dir, use_llm=False)
    planner = Planner(data_dir, engine)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for item in load_jsonl(input_path):
            f.write(json.dumps({"id": item["id"], "actions": planner.plan(item["command"], item["as_of"])}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    raise SystemExit("Use `python3 run.py actions ...` from the project root.")
