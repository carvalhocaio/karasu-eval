from typing import Literal

from pydantic import BaseModel

from karasu_eval.models import EvaluationContract, RunSummary, Severity, Verdict


class RegressionItem(BaseModel):
    case_id: str
    criterion_id: str
    severity: Severity
    baseline_verdict: Verdict
    candidate_verdict: Verdict


class GateReport(BaseModel):
    baseline_name: str
    candidate_name: str
    decision: Literal["PASS", "FAIL", "REVIEW"]
    exit_code: int
    reason: str
    new_blocker_failures: list[RegressionItem]
    new_non_blocker_failures: list[RegressionItem]
    carryover_failures: list[RegressionItem]
    fixed_failures: list[RegressionItem]
    review_cases: list[RegressionItem]


def _index_verdicts(summary: RunSummary) -> dict[tuple[str, str], Verdict]:
    return {
        (evaluation.case_id, result.criterion_id): result.verdict
        for evaluation in summary.evaluations
        for result in evaluation.results
    }


def evaluate_release_gate(
    contract: EvaluationContract,
    baseline: RunSummary,
    candidate: RunSummary,
    baseline_name: str = "baseline",
    candidate_name: str = "candidate",
) -> GateReport:
    severity_by_criterion: dict[str, Severity] = {
        c.id: c.severity for c in contract.criteria
    }
    baseline_verdicts = _index_verdicts(baseline)
    candidate_verdicts = _index_verdicts(candidate)

    new_blocker_failures: list[RegressionItem] = []
    new_non_blocker_failures: list[RegressionItem] = []
    carryover_failures: list[RegressionItem] = []
    fixed_failures: list[RegressionItem] = []
    review_cases: list[RegressionItem] = []

    for key, baseline_verdict in baseline_verdicts.items():
        candidate_verdict = candidate_verdicts.get(key)
        if candidate_verdict is None:
            continue
        case_id, criterion_id = key
        severity = severity_by_criterion[criterion_id]
        item = RegressionItem(
            case_id=case_id,
            criterion_id=criterion_id,
            severity=severity,
            baseline_verdict=baseline_verdict,
            candidate_verdict=candidate_verdict,
        )
        if candidate_verdict == "REVIEW":
            review_cases.append(item)
        elif baseline_verdict == "PASS" and candidate_verdict == "FAIL":
            if severity == "blocker":
                new_blocker_failures.append(item)
            else:
                new_non_blocker_failures.append(item)
        elif baseline_verdict == "FAIL" and candidate_verdict == "FAIL":
            carryover_failures.append(item)
        elif baseline_verdict == "FAIL" and candidate_verdict == "PASS":
            fixed_failures.append(item)

    if new_blocker_failures:
        decision: Literal["PASS", "FAIL", "REVIEW"] = "FAIL"
        exit_code = 1
        reason = (
            f"Blocked release: {len(new_blocker_failures)} new blocker regression(s)."
        )
    elif review_cases:
        decision = "REVIEW"
        exit_code = 2
        reason = f"Manual review required: {len(review_cases)} unverified verdict(s)."
    else:
        decision = "PASS"
        exit_code = 0
        reason = (
            f"Release approved: 0 new blocker regressions, "
            f"{len(fixed_failures)} failure(s) fixed."
        )

    return GateReport(
        baseline_name=baseline_name,
        candidate_name=candidate_name,
        decision=decision,
        exit_code=exit_code,
        reason=reason,
        new_blocker_failures=new_blocker_failures,
        new_non_blocker_failures=new_non_blocker_failures,
        carryover_failures=carryover_failures,
        fixed_failures=fixed_failures,
        review_cases=review_cases,
    )
