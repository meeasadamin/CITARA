"""Scope control (feature 45).

An out-of-scope question gets a scope statement, not a general-knowledge answer. This is what
keeps the system behaving like an institutional instrument rather than a general chatbot in
NDMA colours - a distinction a government reviewer notices immediately.

The relevance floor already refuses these, so this exists to change the *wording*: telling a
user "the corpus does not contain this" when they asked for the capital of France is
technically true and slightly absurd. Detection is conservative, and ambiguity falls through
to the floor, which is the stronger guarantee.
"""

from __future__ import annotations

import re

from citara.log import get_logger

log = get_logger("guardrails.scope")

# Capability requests that are plainly not questions about the corpus.
_OFF_DOMAIN = (
    re.compile(
        r"\b(write|generate|create|code|implement|debug|refactor)\b[^.?!]{0,30}"
        r"\b(code|function|script|program|python|javascript|sql|query|regex|app)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(translate|summarise|summarize|rewrite)\b[^.?!]{0,20}\bthis\b", re.IGNORECASE),
    re.compile(r"\b(tell me a (joke|story)|write a poem|recipe for)\b", re.IGNORECASE),
    re.compile(
        r"\bwhat is the (capital|population|currency|weather|time)\b(?![^.?!]*\b"
        r"(pakistan|sindh|punjab|balochistan|khyber|gilgit|kashmir)\b)",
        re.IGNORECASE,
    ),
)

# Vocabulary that marks a question as plausibly about this corpus. Presence of any of these
# defers the decision to retrieval, which is better placed to judge.
_DOMAIN_TERMS = frozenset(
    [
        "disaster",
        "flood",
        "floods",
        "flooding",
        "monsoon",
        "heatwave",
        "heat",
        "earthquake",
        "landslide",
        "avalanche",
        "glof",
        "glacial",
        "drought",
        "cyclone",
        "rain",
        "rainfall",
        "evacuation",
        "evacuate",
        "rescue",
        "relief",
        "camp",
        "shelter",
        "hazard",
        "risk",
        "resilience",
        "mitigation",
        "preparedness",
        "response",
        "recovery",
        "warning",
        "alert",
        "advisory",
        "contingency",
        "ndma",
        "pdma",
        "ddma",
        "neoc",
        "nidm",
        "pdna",
        "ndrp",
        "ndmp",
        "sendai",
        "damage",
        "losses",
        "casualties",
        "district",
        "province",
        "provincial",
        "federal",
        "authority",
        "emergency",
        "crisis",
        "vulnerability",
        "adaptation",
        "climate",
        "irrigation",
        "embankment",
        "barrage",
        "dam",
        "reservoir",
        "displacement",
        "stranded",
        "casualty",
        "logistics",
        "stockpile",
        "helicopter",
        "boat",
        "siren",
        "threshold",
        "trigger",
        "protocol",
        "sop",
        "plan",
        "policy",
        "strategy",
        "framework",
        "act",
        "ordinance",
    ]
)

PROTOTYPE_DISCLAIMER = (
    "CITARA is an independent decision-support prototype built on publicly available NDMA "
    "documents. It is not an official NDMA system, and its answers must be checked against "
    "the cited source page before being acted on."
)

SCOPE_MESSAGE = (
    "That question falls outside what this assistant covers. CITARA answers only from "
    "NDMA's published disaster-management documents - response plans, contingency plans, "
    "damage assessments, advisories and guidelines. Ask about those and it will answer with "
    "the source page."
)

_WORD = re.compile(r"[a-z]+")


def mentions_domain(question: str) -> bool:
    """True when the question uses any vocabulary this corpus could plausibly cover."""
    return any(word in _DOMAIN_TERMS for word in _WORD.findall(question.lower()))


def is_out_of_scope(question: str) -> bool:
    """True only when a question is clearly not about disaster management.

    Deliberately reluctant: a false positive refuses a real question with a message implying
    the user asked something silly, which is worse than the floor's plainer refusal.
    """
    if mentions_domain(question):
        return False
    out_of_scope = any(pattern.search(question) for pattern in _OFF_DOMAIN)
    if out_of_scope:
        log.info("question deflected as out of scope", extra={"question": question[:120]})
    return out_of_scope
