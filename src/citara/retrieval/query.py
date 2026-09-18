"""Query preparation: acronym expansion and history-aware rewriting (features 30-32).

Two problems, solved separately.

**Acronyms.** The corpus writes both "NEOC" and "National Emergency Operations Centre", and a
user types whichever they know. Expansion is applied to the *sparse* query only: BM25 gains
the missing tokens, while the dense query keeps the user's own phrasing, because padding an
embedding input with boilerplate moves it away from what the user actually meant.

**Follow-ups.** "What about Sindh?" is four words that retrieve nothing. Condensing the recent
turns into a standalone question is the single highest-value feature in the retrieval
pipeline, because a panel will produce this pattern within three questions without trying.
"""

from __future__ import annotations

import re
from typing import Protocol

from citara.config import RetrievalSettings
from citara.log import get_logger

log = get_logger("retrieval.query")

# Domain vocabulary, expanded into the sparse query so either form matches.
ACRONYMS: dict[str, str] = {
    "ndma": "National Disaster Management Authority",
    "pdma": "Provincial Disaster Management Authority",
    "ddma": "District Disaster Management Authority",
    "neoc": "National Emergency Operations Centre",
    "peoc": "Provincial Emergency Operations Centre",
    "pdna": "Post Disaster Needs Assessment",
    "drr": "Disaster Risk Reduction",
    "drm": "Disaster Risk Management",
    "ews": "Early Warning System",
    "glof": "Glacial Lake Outburst Flood",
    "ndrp": "National Disaster Response Plan",
    "ndmp": "National Disaster Management Plan",
    "nap": "National Adaptation Plan",
    "insar": "Integrated National Search and Rescue",
    "usar": "Urban Search and Rescue",
    "sar": "Search and Rescue",
    "sop": "Standard Operating Procedure",
    "pmd": "Pakistan Meteorological Department",
    "ffc": "Federal Flood Commission",
    "nidm": "National Institute of Disaster Management",
    "gb": "Gilgit-Baltistan",
    "ajk": "Azad Jammu and Kashmir",
    "kp": "Khyber Pakhtunkhwa",
}

# Reverse direction: the expansion in the query, the acronym in the corpus.
_EXPANSIONS = {phrase.lower(): acronym.upper() for acronym, phrase in ACRONYMS.items()}

_WORD = re.compile(r"[a-z0-9&-]+")
# A question this short, or opening like this, only makes sense given the turn before it.
_FOLLOW_UP_MARKERS = re.compile(
    r"^\s*(what about|how about|and |what of|there|those|these|it |that |they |same for)",
    re.IGNORECASE,
)
_SHORT_QUESTION_WORDS = 6


def expand_acronyms(query: str) -> str:
    """Append expansions (and contractions) so the sparse query matches either form."""
    lowered = query.lower()
    additions: list[str] = []

    for token in _WORD.findall(lowered):
        phrase = ACRONYMS.get(token)
        if phrase and phrase.lower() not in lowered:
            additions.append(phrase)

    for phrase, acronym in _EXPANSIONS.items():
        if phrase in lowered and acronym.lower() not in _WORD.findall(lowered):
            additions.append(acronym)

    return f"{query} {' '.join(additions)}".strip() if additions else query


def looks_like_follow_up(question: str) -> bool:
    """True when a question cannot stand on its own."""
    if _FOLLOW_UP_MARKERS.match(question):
        return True
    return len(question.split()) <= _SHORT_QUESTION_WORDS and "?" in question


class Rewriter(Protocol):
    """Condenses conversation history plus a new question into a standalone query."""

    def rewrite(self, question: str, history: list[str]) -> str: ...


class HeuristicRewriter:
    """History-aware rewriting without an LLM.

    Carries the previous turn's content words into the new question. Cruder than a model, but
    it costs nothing, never fails, and keeps retrieval evaluable offline - so the fallback
    when a provider is unreachable is degraded rather than broken.
    """

    _STOPWORDS = frozenset(
        (
            "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
            "is", "are", "was", "were", "do", "does", "did", "the", "a", "an",
            "of", "for", "to", "in", "on", "at", "by", "with", "from", "about",
            "and", "or", "as", "that", "this", "these", "those", "it", "its",
            "their", "there", "here", "me", "my", "our", "your", "please",
            "tell", "explain", "give", "show",
        )
    )  # fmt: skip

    def rewrite(self, question: str, history: list[str]) -> str:
        if not history or not looks_like_follow_up(question):
            return question
        previous = history[-1]
        carried = [
            word
            for word in _WORD.findall(previous.lower())
            if word not in self._STOPWORDS and len(word) > 2
        ]
        if not carried:
            return question
        return f"{question.rstrip('?').strip()} {' '.join(carried)}"


def prepare(
    question: str,
    history: list[str],
    settings: RetrievalSettings,
    rewriter: Rewriter | None = None,
) -> tuple[str, str]:
    """Produce the dense and sparse queries for one turn.

    Returns ``(dense_query, sparse_query)``. Only the most recent turns feed rewriting: an
    unbounded window makes a conversation that began on floods contaminate every later
    earthquake question (feature 31).
    """
    window = history[-settings.history_turns :] if settings.history_turns else []
    rewriter = rewriter or HeuristicRewriter()
    dense_query = rewriter.rewrite(question, window) if window else question

    if dense_query != question:
        log.info(
            "query rewritten",
            extra={"original": question, "rewritten": dense_query, "turns": len(window)},
        )

    sparse_query = expand_acronyms(dense_query) if settings.expand_acronyms else dense_query
    return dense_query, sparse_query
