from karasu_eval.client import ModelClient
from karasu_eval.graders.deterministic import grade_deterministic_criterion
from karasu_eval.graders.llm_judge import grade_llm_criterion
from karasu_eval.models import (
    CaseEvaluation,
    CriterionConfig,
    CriterionResult,
    EvalCase,
    EvaluationContract,
    RunSummary,
)


def grade_criterion(
    criterion: CriterionConfig,
    response: str,
    case: EvalCase,
    client: ModelClient,
) -> CriterionResult:
    if criterion.oracle_type == "deterministic":
        return grade_deterministic_criterion(criterion, response, case)
    return grade_llm_criterion(criterion, response, case, client)


def evaluate_dataset(
    contract: EvaluationContract,
    cases: list[EvalCase],
    system_prompt: str,
    client: ModelClient,
) -> RunSummary:
    evaluations: list[CaseEvaluation] = []
    for case in cases:
        response = client.generate(system_prompt, case)
        results = [
            grade_criterion(criterion, response, case, client).model_copy(
                update={"case_id": case.id}
            )
            for criterion in contract.criteria
        ]
        evaluations.append(
            CaseEvaluation(case_id=case.id, response=response, results=results)
        )
    return RunSummary(
        contract_version=contract.version,
        blocker_criterion_ids=[
            criterion.id
            for criterion in contract.criteria
            if criterion.severity == "blocker"
        ],
        evaluations=evaluations,
    )
