"""People directory built from the connectors (Slack users, email headers, calendar attendees)."""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field

from .indexer import CorpusIndex

_ADDR = re.compile(r"^\s*(?:\"?([^<\"]*?)\"?\s*)?<?([\w.+-]+@[\w.-]+\.\w+)>?\s*$")
_AUTOMATED = re.compile(r"no-?reply|notification|digest|newsletter|donotreply|reminders?@|billing|events@|hello@|info@", re.I)


@dataclass
class Person:
    name: str
    email: str | None = None
    slack_id: str | None = None
    title: str = ""
    aliases: set[str] = field(default_factory=set)

    @property
    def first(self) -> str:
        return self.name.split()[0] if self.name else (self.email or "").split("@")[0]


class Directory:
    def __init__(self, corpus: CorpusIndex):
        self.people: list[Person] = []
        by_email: dict[str, Person] = {}

        for u in corpus.users:
            if u.get("is_bot"):
                continue
            p = Person(u.get("real_name") or u.get("name"), u.get("email"), u.get("id"), u.get("title") or "")
            p.aliases = {str(u.get("name") or "").lower()}
            self.people.append(p)
            if p.email:
                by_email[p.email.lower()] = p

        sent, rcvd = Counter(), Counter()
        gm = corpus.data_dir / "connectors/gmail/messages.jsonl"
        if gm.exists():
            for line in gm.read_text().splitlines():
                if not line.strip():
                    continue
                m = json.loads(line)
                for field_name, bucket in (("from", sent), ("to", rcvd), ("cc", rcvd)):
                    vals = m.get(field_name) or []
                    for raw in ([vals] if isinstance(vals, str) else vals):
                        mm = _ADDR.match(raw)
                        if not mm:
                            continue
                        name, email = (mm.group(1) or "").strip(), mm.group(2).lower()
                        bucket[email] += 1
                        if _AUTOMATED.search(email):
                            continue
                        if email not in by_email:
                            p = Person(name or email.split("@")[0].replace(".", " ").title(), email)
                            self.people.append(p)
                            by_email[email] = p
                        elif name and by_email[email].name.lower() == email.split("@")[0]:
                            by_email[email].name = name
        # The account owner is the address that receives the most mail.
        self.self_email = rcvd.most_common(1)[0][0] if rcvd else None
        self.me = by_email.get(self.self_email) if self.self_email else None

    # ------------------------------------------------------------------
    def find(self, query: str, via: str | None = None) -> list[Person]:
        """All people matching a spoken reference ('Sarah', 'Sarah Patel', 'ben@...'), best tier only.
        `via` = 'slack' | 'email' restricts to people reachable that way."""
        q = re.sub(r"[^\w\s@.'-]", "", query.lower()).strip()
        if not q:
            return []
        pool = [p for p in self.people if p is not self.me]
        if via == "slack":
            pool = [p for p in pool if p.slack_id]
        elif via == "email":
            pool = [p for p in pool if p.email]
        tiers = [
            lambda p: q == p.name.lower() or q == (p.email or "").lower() or q in p.aliases,
            lambda p: q == p.first.lower() or q == p.name.lower().split()[-1],
            lambda p: p.name.lower().startswith(q) or q in p.name.lower(),
        ]
        for tier in tiers:
            hit = [p for p in pool if tier(p)]
            if hit:
                return hit
        return []
