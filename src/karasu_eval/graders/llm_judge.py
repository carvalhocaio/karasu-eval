from karasu_eval.client import ModelClient
from karasu_eval.models import CriterionConfig, CriterionResult, EvalCase


def normalize_text(text: str) -> str:
    return " ".join(text.split()).casefold()


def validate_evidence_quotes(
    response: str, protocol_context: str, quotes: list[str]
) -> bool:
    if not quotes:
        return False
    sources = (normalize_text(response), normalize_text(protocol_context))
    return all(
        normalize_text(quote) != ""
        and any(normalize_text(quote) in source for source in sources)
        for quote in quotes
    )


def grade_llm_criterion(
    criterion: CriterionConfig,
    response: str,
    case: EvalCase,
    client: ModelClient,
) -> CriterionResult:
    raw = client.judge(criterion, case, response)

    if not validate_evidence_quotes(
        response, case.protocol_context, raw.evidence_quotes
    ):
        return CriterionResult(
            criterion_id=criterion.id,
            verdict="REVIEW",
            score=None,
            evidence_quotes=raw.evidence_quotes,
            rationale=(
                "Judge evidence quote could not be verified verbatim in the response "
                f"or protocol context; unverified verdict needs human review. "
                f"Judge rationale: {raw.rationale}"
            ),
        )

    return CriterionResult(
        criterion_id=criterion.id,
        verdict="PASS" if raw.score >= criterion.pass_threshold else "FAIL",
        score=raw.score,
        evidence_quotes=raw.evidence_quotes,
        rationale=raw.rationale,
    )
