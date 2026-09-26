"""A model scoring another model's answers (features 68, 69).

Two questions are asked about every generated answer, and they are deliberately kept apart
because they fail in different ways.

**Faithfulness** asks whether each claim in the answer can be read in the passages that were
retrieved. The judge is given the passages and nothing else: no gold answer, no question, and
an explicit instruction that a claim which is true in the world but absent from the passages
is unsupported. That is the whole point of the measurement. A system that answers correctly
from the model's own memory has failed at the thing this project exists to do, and a judge
allowed to use its own knowledge would score that failure as a success.

**Answer relevance** asks whether the answer addresses the question that was asked, which is
a separate property from being true. The judge sees the question and the answer, and not the
passages, so a faithful answer to a question nobody asked still scores badly.

The judge is a model, so it is fallible, and two things keep it honest. It returns the claims
it extracted rather than a bare number, so a surprising score can be read back and checked by
a human. And the provider it runs on is recorded with every score, because the default judge
is the same model family as the generator, and a model scoring its own output is the weakest
form of this measurement - ``--judge groq`` runs it again on another company's model.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from citara.config import Settings
from citara.generation.providers import GeminiProvider, GroqProvider, Provider
from citara.log import get_logger

log = get_logger("evaluation.judge")

# Claims are scored one by one rather than as a single verdict on the answer, because "mostly
# supported" is the interesting case and a single verdict hides it.
_FAITHFULNESS_SYSTEM = """You check whether an answer stays inside the evidence it was given.

You will be shown numbered passages and an answer written from them.

Break the answer into its separate factual claims. A claim is a single checkable assertion:
a figure, a date, a name, an obligation, a step in a procedure. Ignore hedging, headings,
and sentences that only refer to the passages themselves.

For each claim, decide whether the passages state it or directly entail it.

Rules you must follow:
- Use only the passages. Your own knowledge is not evidence.
- A claim that is true in the world but absent from the passages is UNSUPPORTED.
- A claim that generalises beyond what a passage says is UNSUPPORTED.
- A figure that differs from the passage in value, unit or scope is UNSUPPORTED.
- If the answer makes no factual claims at all, return an empty list.

Reply with JSON only, in this shape:
{"claims": [{"claim": "<the claim, quoted or paraphrased>", "supported": true,
             "why": "<passage number and why, in one short sentence>"}]}"""

_RELEVANCE_SYSTEM = """You judge whether an answer addresses the question that was asked.

You are not checking whether the answer is true, and you are not checking its sources. A
truthful answer to a different question is a bad answer. A refusal that explains why the
question cannot be answered from the available documents is addressing the question, and
scores 1.0.

Score on this scale, and use only these three values:
- 1.0 - answers the question that was asked, completely.
- 0.5 - addresses the question but leaves a substantial part of it unanswered, or answers a
  narrower or broader question than the one asked.
- 0.0 - does not address the question.

Reply with JSON only, in this shape:
{"score": 1.0, "why": "<one short sentence>"}"""

# Judging is a bigger request than answering: the judge reads every retrieved passage as
# well as the answer, where the assistant only writes a capped reply. The assistant's 30 s
# is right for a reader who is waiting, and too short here - it cost a run to a 504.
_JUDGE_TIMEOUT_S = 90.0
# The assistant's 1024-token cap is sized for a short answer. A verdict on every claim in
# that answer, each with its reason, is a longer reply than the answer it judges.
_JUDGE_MAX_TOKENS = 3072

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)
_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


class JudgeUnavailable(RuntimeError):
    """The judge could not be reached, or did not answer in a readable form."""


def parse_json_object(text: str) -> dict[str, object]:
    """The first JSON object in *text*, however the model wrapped it.

    Models fence their JSON, preface it with a sentence, or do both, and a scoring run that
    dies on a stray code fence has thrown away the requests it already spent.
    """
    stripped = _FENCE.sub("", text).strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        match = _OBJECT.search(stripped)
        if match is None:
            raise JudgeUnavailable(f"no JSON object in judge reply: {text[:200]}") from None
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError as error:
            raise JudgeUnavailable(f"unreadable judge reply: {text[:200]}") from error
    if not isinstance(parsed, dict):
        raise JudgeUnavailable(f"judge replied with {type(parsed).__name__}, not an object")
    return parsed


@dataclass(frozen=True)
class ClaimVerdict:
    """One factual claim from an answer, and whether the passages carry it."""

    claim: str
    supported: bool
    why: str = ""


@dataclass(frozen=True)
class FaithfulnessScore:
    """The share of an answer's claims the evidence supports.

    ``score`` is None when the judge found no claims to check. That is not a perfect score
    and must not be averaged as one: it means the answer asserted nothing, which for this
    system is its own kind of failure.
    """

    claims: tuple[ClaimVerdict, ...] = ()

    @property
    def score(self) -> float | None:
        if not self.claims:
            return None
        return sum(1 for claim in self.claims if claim.supported) / len(self.claims)

    @property
    def unsupported(self) -> tuple[str, ...]:
        return tuple(claim.claim for claim in self.claims if not claim.supported)


@dataclass(frozen=True)
class RelevanceScore:
    """How well an answer addressed the question, on the three-point scale."""

    score: float
    why: str = ""


def _passages(evidence: list[tuple[str, str]]) -> str:
    """Numbered passages, in the same shape the answer's markers refer to."""
    return "\n\n".join(
        f"[{index}] {citation}\n{text}" for index, (citation, text) in enumerate(evidence, start=1)
    )


class Judge:
    """Scores generated answers for faithfulness and relevance."""

    def __init__(self, provider: Provider, name: str, model: str) -> None:
        self.provider = provider
        self.name = name
        self.model = model

    def faithfulness(self, answer: str, evidence: list[tuple[str, str]]) -> FaithfulnessScore:
        """Which of the answer's claims the retrieved passages support (feature 68)."""
        if not evidence:
            # Nothing was retrieved, so nothing in the answer can be supported by it. This is
            # a real result, not a missing measurement: every claim is unsupported.
            return FaithfulnessScore(
                claims=(ClaimVerdict(answer.strip()[:200], False, "no passages were retrieved"),)
            )
        user = f"PASSAGES\n{_passages(evidence)}\n\nANSWER\n{answer}"
        payload = parse_json_object(self._ask(_FAITHFULNESS_SYSTEM, user))
        raw = payload.get("claims", [])
        if not isinstance(raw, list):
            raise JudgeUnavailable("judge returned claims that are not a list")
        verdicts = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            verdicts.append(
                ClaimVerdict(
                    claim=str(item.get("claim", "")).strip(),
                    supported=bool(item.get("supported", False)),
                    why=str(item.get("why", "")).strip(),
                )
            )
        return FaithfulnessScore(claims=tuple(verdicts))

    def relevance(self, question: str, answer: str) -> RelevanceScore:
        """Whether the answer addressed the question asked (feature 69)."""
        user = f"QUESTION\n{question}\n\nANSWER\n{answer}"
        payload = parse_json_object(self._ask(_RELEVANCE_SYSTEM, user))
        try:
            raw = float(payload.get("score", 0.0))  # type: ignore[arg-type]
        except (TypeError, ValueError) as error:
            raise JudgeUnavailable(f"judge returned a non-numeric score: {payload}") from error
        # Snapped to the scale it was given rather than trusted: a judge that answers 0.85 has
        # ignored the instruction, and averaging its improvisation would blur the measurement.
        score = min((1.0, 0.5, 0.0), key=lambda point: abs(point - raw))
        return RelevanceScore(score=score, why=str(payload.get("why", "")).strip())

    def _ask(self, system: str, user: str) -> str:
        return self.provider.generate(system, user)


def build_judge(settings: Settings, name: str = "gemini") -> Judge:
    """A judge on *name*, pinned to the evaluation model and temperature.

    The judge runs through the same provider wrapper as the assistant, so it inherits the one
    retry layer, the request budget and the quota cooldowns rather than opening a second,
    uncounted path to the same APIs.
    """
    evaluation = settings.evaluation
    if name == "gemini":
        if settings.google_api_key is None:
            raise JudgeUnavailable("GOOGLE_API_KEY is not configured")
        model = evaluation.judge_model
        pinned = settings.model_copy(
            update={
                "generation": settings.generation.model_copy(
                    update={
                        "primary_model": model,
                        "temperature": evaluation.judge_temperature,
                        "request_timeout_s": _JUDGE_TIMEOUT_S,
                        "max_output_tokens": _JUDGE_MAX_TOKENS,
                    }
                )
            }
        )
        provider: Provider = GeminiProvider(pinned)
    elif name == "groq":
        if settings.groq_api_key is None:
            raise JudgeUnavailable("GROQ_API_KEY is not configured")
        model = settings.generation.fallback_model
        pinned = settings.model_copy(
            update={
                "generation": settings.generation.model_copy(
                    update={
                        "temperature": evaluation.judge_temperature,
                        "request_timeout_s": _JUDGE_TIMEOUT_S,
                        "max_output_tokens": _JUDGE_MAX_TOKENS,
                    }
                )
            }
        )
        provider = GroqProvider(pinned)
    else:
        raise JudgeUnavailable(f"unknown judge: {name}")

    log.info("judge ready", extra={"judge": name, "model": model})
    return Judge(provider=provider, name=name, model=model)
