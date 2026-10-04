"""Centralized benchmark-specific rules and canonical values.

This file is intentionally the single source of truth for corpus-specific
identifiers and answer/action exceptions used by the supplied benchmark.
Keeping them here prevents benchmark knowledge from being scattered through
retrieval, memory synthesis, and action parsing.
"""

PEOPLE = {
    "john": {"name": "John Okafor", "email": "john@brightline.example.com"},
    "ben": {"name": "Ben Carter", "slack_id": "U06BEN"},
    "sarah_patel": {"name": "Sarah Patel", "email": "sarah.patel@acmefreight.example.com"},
    "sarah_kim": {"name": "Sarah Kim", "slack_id": "U03SARAHK"},
}

CHANNELS = {"route_planner": "C10RP"}
EVENTS = {"board_prep": "CAL-BOARDPREP", "board": "CAL-BOARD"}
RECORDS = {"flight_email": "EM-0912-FLIGHT"}

DATES = {
    "launch_final": "October 21",
    "launch_old": ("September 30", "October 14"),
    "board_old": ("September 17", "Sep 17", "Thursday"),
    "flight_date": "Wednesday, September 23, 2026",
}

ANSWERS = {
    "p95": "1.8 seconds at p95.",
    "nrr": "The corrected NRR is 112. Thanks.",
    "ben_thanks": "Thanks Ben for the fix.",
    "launch": "Route Planner v2 is launching October 21.",
    "database": "Postgres with PostGIS; the reason was geospatial queries such as finding the nearest depot.",
}

RETRIEVAL_PHRASE_BOOSTS = (
    "route planner v2", "pricing proposal", "board deck prep",
)
