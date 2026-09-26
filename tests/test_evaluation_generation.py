"""Tests for the generation suite: faithfulness, relevance and latency (features 68, 69, 71)."""

from __future__ import annotations

import pytest

from citara.chunking.models import Chunk
from citara.config import Settings
from citara.evaluation.generation import (
    GenerationReport,
    QuestionResult,
    evaluation_settings,
    render_markdown,
    run_generation,
)
from citara.evaluation.judge import (
    ClaimVerdict,
    FaithfulnessScore,
    Judge,
    JudgeUnavailable,
    declines,
    parse_json_object,
)
from citara.evaluation.metrics import mean, percentile
from citara.evaluation.models import GoldQuestion, GoldSet, GoldSource
from citara.generation.models import Citation, GeneratedAnswer
from citara.retrieval.models import RetrievedChunk

# -- fakes ----------------------------------------------------------------------------------


class ScriptedProvider:
    """A provider that replies with whatever the test queued, in order."""

    name = "scripted"

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.prompts: list[tuple[str, str]] = []

    def generate(self, system: str, user: str) -> str:
        self.prompts.append((system, user))
        if not self.replies:
            raise AssertionError("the judge asked more questions than the test scripted")
        return self.replies.pop(0)

    def stream(self, system: str, user: str):  # pragma: no cover - the judge never streams
        raise NotImplementedError

    def warm_up(self) -> None:  # pragma: no cover - nothing to warm
        return None


class ScriptedJudge(Judge):
    """Fixed verdicts, so the run loop can be tested without a model."""

    def __init__(self, faithfulness: float | None = 1.0, relevance: float = 1.0) -> None:
        super().__init__(provider=ScriptedProvider([]), name="scripted", model="test-judge")
        self._faithfulness = faithfulness
        self._relevance = relevance
        self.scored: list[str] = []

    def faithfulness(self, answer: str, evidence: list[tuple[str, str]]) -> FaithfulnessScore:
        self.scored.append(answer)
        if self._faithfulness is None:
            return FaithfulnessScore(claims=())
        supported = ClaimVerdict("a supported claim", True)
        if self._faithfulness == 1.0:
            return FaithfulnessScore(claims=(supported,))
        return FaithfulnessScore(claims=(supported, ClaimVerdict("an invented figure", False)))

    def relevance(self, question: str, answer: str):
        from citara.evaluation.judge import RelevanceScore

        return RelevanceScore(score=self._relevance)


class ScriptedAnswerer:
    """Returns a queued answer per question, keyed on the question text."""

    def __init__(self, answers: dict[str, GeneratedAnswer | Exception]) -> None:
        self.answers = answers
        self.asked: list[str] = []

    def answer(
        self,
        question: str,
        history: list[str] | None = None,
        doc_ids: list[str] | None = None,
        year: int | None = None,
        session_id: str = "",
    ) -> GeneratedAnswer:
        self.asked.append(question)
        reply = self.answers[question]
        if isinstance(reply, Exception):
            raise reply
        return reply


def evidence(content: str = "35,000 people died in the 1935 Quetta earthquake.") -> RetrievedChunk:
    chunk = Chunk(
        chunk_id="c1",
        doc_id="ndmp2",
        title="NDMP II 2023",
        filename="ndmp2.pdf",
        page_start=21,
        page_end=24,
        total_pages=200,
        year=2023,
        ordinal=0,
        content=content,
    )
    return RetrievedChunk(chunk=chunk, rerank_score=0.9)


def generated(
    text: str = "35,000 people died [1].", retrieval_ms: float = 3200.0
) -> GeneratedAnswer:
    return GeneratedAnswer(
        text=text,
        mode="generated",
        citations=[Citation(1, "NDMP II 2023, pp. 21-24", "c1", "ndmp2", 21, 24, used=True)],
        provider="gemini",
        model="gemini-3.5-flash-lite",
        evidence=[evidence()],
        retrieval_ms=retrieval_ms,
        generation_ms=1800.0,
    )


def refused() -> GeneratedAnswer:
    return GeneratedAnswer(text="Not in the corpus.", mode="refused", retrieval_ms=3100.0)


def gold_set() -> GoldSet:
    return GoldSet(
        questions=[
            GoldQuestion(
                id="a1",
                question="How many died in the Quetta earthquake?",
                category="table",
                expected_answer="35,000",
                sources=[GoldSource(doc_id="ndmp2", page_start=21, page_end=24)],
            ),
            GoldQuestion(
                id="a2",
                question="How are early warnings disseminated?",
                category="procedural",
                expected_answer="through PDMAs",
                sources=[GoldSource(doc_id="ndrp", page_start=47, page_end=47)],
            ),
            GoldQuestion(
                id="u1",
                question="What is the Gwadar heatwave threshold?",
                category="unanswerable",
                answerable=False,
            ),
        ]
    )


# -- reading what the judge said --------------------------------------------------------------


def test_a_fenced_reply_is_still_read() -> None:
    """Models fence their JSON; a run must not die on a code fence it already paid for."""
    assert parse_json_object('```json\n{"score": 1.0}\n```') == {"score": 1.0}


def test_a_reply_with_a_preamble_is_still_read() -> None:
    text = 'Here is my assessment:\n{"score": 0.5, "why": "partial"}'
    assert parse_json_object(text)["score"] == 0.5


def test_a_reply_with_no_json_is_refused_rather_than_guessed() -> None:
    with pytest.raises(JudgeUnavailable):
        parse_json_object("I think the answer is quite good, actually.")


def test_a_reply_that_is_not_an_object_is_refused() -> None:
    with pytest.raises(JudgeUnavailable):
        parse_json_object("[1, 2, 3]")


# -- the two scores ---------------------------------------------------------------------------


def test_faithfulness_is_the_share_of_claims_the_evidence_carries() -> None:
    score = FaithfulnessScore(
        claims=(
            ClaimVerdict("35,000 died", True),
            ClaimVerdict("the figure is disputed", True),
            ClaimVerdict("the epicentre was 8 km deep", False),
        )
    )
    assert score.score == pytest.approx(2 / 3)
    assert score.unsupported == ("the epicentre was 8 km deep",)


def test_an_answer_that_asserts_nothing_scores_none_rather_than_one() -> None:
    """No claims is not a perfect score, and averaging it as one would flatter the system."""
    assert FaithfulnessScore(claims=()).score is None


def test_nothing_retrieved_means_nothing_in_the_answer_is_supported() -> None:
    judge = Judge(ScriptedProvider([]), "scripted", "test")
    score = judge.faithfulness("Some claim about flood damage.", [])
    assert score.score == 0.0
    assert score.unsupported


def test_the_faithfulness_prompt_carries_the_passages_and_the_answer() -> None:
    provider = ScriptedProvider(['{"claims": [{"claim": "x", "supported": true}]}'])
    judge = Judge(provider, "scripted", "test")
    judge.faithfulness("35,000 died [1].", [("NDMP II, p. 21", "35,000 people died")])

    _, user = provider.prompts[0]
    assert "35,000 people died" in user
    assert "35,000 died [1]." in user


def test_a_judge_that_improvises_a_score_is_snapped_to_the_scale_it_was_given() -> None:
    """A judge answering 0.85 ignored its instructions; averaging that blurs the measure."""
    provider = ScriptedProvider(['{"score": 0.85, "why": "mostly"}'])
    judge = Judge(provider, "scripted", "test")
    assert judge.relevance("q", "a").score == 1.0

    provider = ScriptedProvider(['{"score": 0.4}'])
    assert Judge(provider, "scripted", "test").relevance("q", "a").score == 0.5


def test_a_non_numeric_score_is_refused() -> None:
    judge = Judge(ScriptedProvider(['{"score": "good"}']), "scripted", "test")
    with pytest.raises(JudgeUnavailable):
        judge.relevance("q", "a")


def test_the_relevance_judge_is_not_shown_the_evidence() -> None:
    """Relevance is about the question, not the sources; showing passages invites correctness."""
    provider = ScriptedProvider(['{"score": 1.0}'])
    Judge(provider, "scripted", "test").relevance("How many died?", "35,000 died [1].")

    system, user = provider.prompts[0]
    assert "PASSAGES" not in user
    assert "not checking whether the answer is true" in system


# -- shared statistics ------------------------------------------------------------------------


def test_a_missing_score_is_skipped_rather_than_counted_as_zero() -> None:
    assert mean([1.0, None, 0.5]) == pytest.approx(0.75)
    assert mean([None, None]) is None


def test_the_percentile_reports_a_value_something_actually_took() -> None:
    assert percentile([10.0, 20.0, 30.0, 40.0], 0.5) == 30.0
    assert percentile([], 0.95) == 0.0


# -- the run --------------------------------------------------------------------------------


def test_a_measured_run_switches_the_cache_off() -> None:
    """A cached answer takes 0.4 s and no model call; with the cache on, latency measures it."""
    settings = evaluation_settings(Settings(_env_file=None))  # type: ignore[call-arg]
    assert settings.resilience.enable_cache is False
    assert settings.resilience.session_query_cap >= 1000


def test_every_question_is_asked_and_generated_answers_are_scored() -> None:
    gold = gold_set()
    answerer = ScriptedAnswerer(
        {
            gold.questions[0].question: generated(),
            gold.questions[1].question: generated(),
            gold.questions[2].question: refused(),
        }
    )
    judge = ScriptedJudge()
    report = run_generation(
        answerer,
        judge,
        gold,
        Settings(_env_file=None),  # type: ignore[call-arg]
    )

    assert len(answerer.asked) == 3
    assert len(report.scored) == 2
    assert report.faithfulness == 1.0
    assert report.relevance == 1.0


def test_a_refusal_is_counted_but_not_scored_for_faithfulness() -> None:
    """A refusal makes no claims, so faithfulness is undefined rather than zero."""
    gold = gold_set()
    answerer = ScriptedAnswerer(
        {
            gold.questions[0].question: generated(),
            gold.questions[1].question: refused(),
            gold.questions[2].question: refused(),
        }
    )
    judge = ScriptedJudge()
    report = run_generation(
        answerer,
        judge,
        gold,
        Settings(_env_file=None),  # type: ignore[call-arg]
    )

    assert len(judge.scored) == 1
    assert [r.id for r in report.wrongly_refused] == ["a2"]
    assert report.wrongly_answered == []


def test_answering_a_question_the_corpus_cannot_answer_is_reported() -> None:
    gold = gold_set()
    answerer = ScriptedAnswerer(
        {
            gold.questions[0].question: generated(),
            gold.questions[1].question: generated(),
            gold.questions[2].question: generated(),
        }
    )
    report = run_generation(
        answerer,
        ScriptedJudge(),
        gold,
        Settings(_env_file=None),  # type: ignore[call-arg]
    )
    assert [r.id for r in report.wrongly_answered] == ["u1"]


def test_one_failing_question_does_not_end_the_run() -> None:
    """The run has already spent quota on the questions before it."""
    gold = gold_set()
    answerer = ScriptedAnswerer(
        {
            gold.questions[0].question: generated(),
            gold.questions[1].question: RuntimeError("provider exhausted"),
            gold.questions[2].question: refused(),
        }
    )
    report = run_generation(
        answerer,
        ScriptedJudge(),
        gold,
        Settings(_env_file=None),  # type: ignore[call-arg]
    )

    assert len(report.results) == 3
    failed = next(r for r in report.results if r.id == "a2")
    assert "provider exhausted" in failed.error
    assert failed not in report.scored
    # A question that never ran is not a refusal either.
    assert failed not in report.wrongly_refused


def test_generation_latency_ignores_questions_that_never_reached_a_model() -> None:
    """A refusal spends no generation time; folding its zero in reports a speed we do not have."""
    gold = gold_set()
    answerer = ScriptedAnswerer(
        {
            gold.questions[0].question: generated(retrieval_ms=3000.0),
            gold.questions[1].question: generated(retrieval_ms=3400.0),
            gold.questions[2].question: refused(),
        }
    )
    report = run_generation(
        answerer,
        ScriptedJudge(),
        gold,
        Settings(_env_file=None),  # type: ignore[call-arg]
    )

    p50 = report.latency(0.5)
    assert p50["generation_ms"] == 1800.0
    # Retrieval covers all three: the refused question's 3100 ms is the median of the set.
    assert p50["retrieval_ms"] == 3100.0


def test_unsupported_claims_are_carried_into_the_report() -> None:
    gold = gold_set()
    answerer = ScriptedAnswerer(
        {
            gold.questions[0].question: generated(),
            gold.questions[1].question: generated(),
            gold.questions[2].question: refused(),
        }
    )
    report = run_generation(
        answerer,
        ScriptedJudge(faithfulness=0.5),
        gold,
        Settings(_env_file=None),  # type: ignore[call-arg]
    )
    assert report.faithfulness == 0.5
    assert report.scored[0].unsupported == ("an invented figure",)


def test_a_timing_that_cannot_be_true_is_named_rather_than_averaged() -> None:
    """A laptop that suspends mid-run carries the sleep into the elapsed-time counter."""
    gold = gold_set()
    slept = generated(retrieval_ms=15_324_147.0)
    answerer = ScriptedAnswerer(
        {
            gold.questions[0].question: generated(retrieval_ms=3200.0),
            gold.questions[1].question: slept,
            gold.questions[2].question: refused(),
        }
    )
    report = run_generation(
        answerer,
        ScriptedJudge(),
        gold,
        Settings(_env_file=None),  # type: ignore[call-arg]
    )

    assert [r.id for r in report.untimed] == ["a2"]
    # The percentile sees only the honest readings.
    assert report.latency(0.95)["retrieval_ms"] < 5000
    # Its score still counts: the clock was wrong, the answer was not.
    assert len(report.scored) == 2
    assert "a2" in render_markdown(report, gold)


def test_the_phrasings_the_prompt_asks_for_are_recognised() -> None:
    """The assistant is told to say so plainly when the evidence falls short; it varies."""
    assert declines("The provided evidence does not contain the total housing damages.")
    assert declines("Based on the provided evidence, the text does not state what countries.")
    assert declines("Based on the provided evidence, there is no mention of what people should do.")
    assert declines("The documents do not specify a threshold for Gwadar.")


def test_a_real_answer_is_not_mistaken_for_a_decline() -> None:
    """A false positive throws away a real answer, so the detector stays narrow."""
    assert not declines("GLOF stands for Glacial Lake Outburst Flood [1].")
    assert not declines("NDMA treats 1 July to 30 September as the monsoon season [1][2].")
    assert not declines("The plan states that no district may deploy without NDMA tasking [1].")


def test_an_answer_that_declines_is_counted_apart_from_the_ones_that_answered() -> None:
    """ "The evidence does not contain X" cannot contradict its evidence, so it scores 1.0."""
    gold = gold_set()
    answerer = ScriptedAnswerer(
        {
            gold.questions[0].question: generated("35,000 died [1]."),
            gold.questions[1].question: generated(
                "The provided evidence does not contain the dissemination steps."
            ),
            gold.questions[2].question: refused(),
        }
    )
    report = run_generation(
        answerer,
        ScriptedJudge(),
        gold,
        Settings(_env_file=None),  # type: ignore[call-arg]
    )

    assert [r.id for r in report.substantive] == ["a1"]
    assert [r.id for r in report.declined_on_evidence] == ["a2"]
    # Averaged over the answer that answered, not over both.
    assert report.faithfulness == 1.0
    assert sum(1 for r in report.substantive if r.faithfulness is not None) == 1


def test_declining_on_an_unanswerable_question_is_not_a_wrong_answer() -> None:
    """The gate let the evidence through and the model still said no. That is correct."""
    gold = gold_set()
    answerer = ScriptedAnswerer(
        {
            gold.questions[0].question: generated(),
            gold.questions[1].question: generated(),
            gold.questions[2].question: generated(
                "The provided evidence does not contain a Gwadar threshold."
            ),
        }
    )
    report = run_generation(
        answerer,
        ScriptedJudge(),
        gold,
        Settings(_env_file=None),  # type: ignore[call-arg]
    )
    assert report.wrongly_answered == []
    assert [r.id for r in report.declined_on_evidence] == ["u1"]


def test_the_artifact_leads_with_how_often_it_answered_at_all() -> None:
    answered = scored_result("a1", 1.0, 1.0)
    declined = QuestionResult(
        id="a2", question="q", category="table", answerable=True, mode="generated", declined=True
    )
    missed = QuestionResult(
        id="a3", question="q", category="table", answerable=True, mode="refused"
    )
    markdown = render_markdown(report_with([answered, declined, missed]), gold_set())

    assert "What happened to the answerable questions" in markdown
    assert "Declined on the evidence" in markdown
    assert "1 of 3 (33%)" in markdown


# -- the artifact ------------------------------------------------------------------------------


def report_with(results: list[QuestionResult]) -> GenerationReport:
    return GenerationReport(
        judge="groq",
        judge_model="openai/gpt-oss-120b",
        generator_model="gemini-3.5-flash-lite",
        fingerprint="abc123",
        generated_at="2026-09-25T12:00:00+00:00",
        results=results,
    )


def scored_result(
    question_id: str, faithfulness: float, relevance: float, category: str = "table"
) -> QuestionResult:
    return QuestionResult(
        id=question_id,
        question="q",
        category=category,
        answerable=True,
        mode="generated",
        faithfulness=faithfulness,
        relevance=relevance,
        claims_checked=2,
        retrieval_ms=3200.0,
        generation_ms=1800.0,
        fully_cited=True,
    )


def test_the_artifact_reports_the_numbers_it_was_given() -> None:
    report = report_with([scored_result("a1", 1.0, 1.0), scored_result("a2", 0.5, 0.5)])
    markdown = render_markdown(report, gold_set())

    assert "0.750" in markdown  # faithfulness
    assert "openai/gpt-oss-120b" in markdown
    assert "abc123" in markdown
    assert "Do not hand-edit" in markdown


def test_the_artifact_names_the_questions_that_went_wrong() -> None:
    wrong = QuestionResult(
        id="u1", question="q", category="unanswerable", answerable=False, mode="generated"
    )
    missed = QuestionResult(
        id="a9", question="q", category="table", answerable=True, mode="refused"
    )
    markdown = render_markdown(
        report_with([scored_result("a1", 1.0, 1.0), wrong, missed]), gold_set()
    )

    assert "u1" in markdown
    assert "a9" in markdown


def test_the_artifact_says_when_the_judge_shared_the_generator_s_family() -> None:
    """Self-scoring is the weakest form of this measurement and must be stated, not buried."""
    report = report_with([scored_result("a1", 1.0, 1.0)])
    report.judge = "gemini"
    markdown = render_markdown(report, gold_set())
    assert "same model family" in markdown
    assert "--judge groq" in markdown


def test_the_artifact_names_the_independent_judge_when_there_was_one() -> None:
    markdown = render_markdown(report_with([scored_result("a1", 1.0, 1.0)]), gold_set())
    assert "a different company's model" in markdown
    assert "same model family" not in markdown


def test_the_judge_defaults_to_the_provider_that_did_not_write_the_answers() -> None:
    """Self-scoring is the weakest reading, so it is not what an unconfigured run gets."""
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.evaluation.judge_provider == "groq"
    assert settings.generation.primary_model.startswith("gemini")


def test_the_answer_and_its_citations_are_kept_for_checking() -> None:
    gold = gold_set()
    answerer = ScriptedAnswerer(
        {
            gold.questions[0].question: generated("35,000 died [1]."),
            gold.questions[1].question: refused(),
            gold.questions[2].question: refused(),
        }
    )
    report = run_generation(
        answerer,
        ScriptedJudge(),
        gold,
        Settings(_env_file=None),  # type: ignore[call-arg]
    )
    scored = report.scored[0]
    assert scored.answer_text == "35,000 died [1]."
    assert scored.evidence_citations == ("NDMP II 2023, pp. 21-24",)


def test_an_empty_run_renders_without_inventing_numbers() -> None:
    markdown = render_markdown(report_with([]), gold_set())
    assert "n/a" in markdown
    assert "0.000" not in markdown
