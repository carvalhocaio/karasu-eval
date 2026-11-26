import pytest

from karasu_eval.client import JudgeRawVerdict
from karasu_eval.graders.llm_judge import (
    grade_llm_criterion,
    validate_evidence_quotes,
)
from karasu_eval.models import CriterionConfig, EvalCase
from tests.stubs import StubClient

RESPONSE = "Aplique o desfolhante quando 60% a 70% das maçãs estiverem abertas."
CONTEXT = "Não aplicar com chuva prevista nas 6 horas seguintes."


@pytest.fixture
def case() -> EvalCase:
    return EvalCase(
        id="cotton-001",
        crop="cotton",
        category="defoliation_window",
        user_query="Quando aplicar desfolhante?",
        protocol_context=CONTEXT,
        expected_decision="APPROVE_WITH_WINDOW",
        required_facts=["60% a 70%"],
        max_words=40,
    )


@pytest.fixture
def criterion() -> CriterionConfig:
    return CriterionConfig(
        id="protocol_correctness",
        description="Protocol fidelity.",
        oracle_type="reference_judge",
        severity="blocker",
        scale_min=1,
        scale_max=5,
        pass_threshold=4.0,
        scale_anchors={1: "Wrong.", 5: "Right."},
    )


def verdict(score: float, quotes: list[str]) -> JudgeRawVerdict:
    return JudgeRawVerdict(score=score, evidence_quotes=quotes, rationale="Judged.")


def test_quote_from_response_is_valid() -> None:
    assert validate_evidence_quotes(RESPONSE, CONTEXT, ["60% a 70% das maçãs"])


def test_quote_from_protocol_context_is_valid() -> None:
    assert validate_evidence_quotes(RESPONSE, CONTEXT, ["chuva prevista"])


def test_validation_ignores_case_and_whitespace() -> None:
    assert validate_evidence_quotes(RESPONSE, CONTEXT, ["  60%  A   70% DAS maçãs "])


def test_empty_quotes_are_invalid() -> None:
    assert not validate_evidence_quotes(RESPONSE, CONTEXT, [])


def test_single_fabricated_quote_invalidates_all() -> None:
    assert not validate_evidence_quotes(
        RESPONSE, CONTEXT, ["60% a 70%", "aplique com 80% das maçãs"]
    )


def test_valid_quote_above_threshold_passes(
    case: EvalCase, criterion: CriterionConfig
) -> None:
    client = StubClient(verdict(5.0, ["60% a 70%"]))
    result = grade_llm_criterion(criterion, RESPONSE, case, client)
    assert result.verdict == "PASS"
    assert result.score == 5.0
    assert result.criterion_id == "protocol_correctness"


def test_score_equal_to_threshold_passes(
    case: EvalCase, criterion: CriterionConfig
) -> None:
    client = StubClient(verdict(4.0, ["60% a 70%"]))
    assert grade_llm_criterion(criterion, RESPONSE, case, client).verdict == "PASS"


def test_score_just_below_threshold_fails(
    case: EvalCase, criterion: CriterionConfig
) -> None:
    client = StubClient(verdict(3.99, ["60% a 70%"]))
    assert grade_llm_criterion(criterion, RESPONSE, case, client).verdict == "FAIL"


def test_fabricated_quote_falls_back_to_review(
    case: EvalCase, criterion: CriterionConfig
) -> None:
    fabricated = ["aplique com 80% das maçãs"]
    client = StubClient(verdict(5.0, fabricated))
    result = grade_llm_criterion(criterion, RESPONSE, case, client)
    assert result.verdict == "REVIEW"
    assert result.score is None
    assert result.evidence_quotes == fabricated
    assert "unverified" in result.rationale.lower()


def test_missing_quotes_fall_back_to_review(
    case: EvalCase, criterion: CriterionConfig
) -> None:
    client = StubClient(verdict(5.0, []))
    result = grade_llm_criterion(criterion, RESPONSE, case, client)
    assert result.verdict == "REVIEW"
    assert result.score is None
