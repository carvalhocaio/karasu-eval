from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, Field, model_validator

OracleType = Literal["deterministic", "reference_judge", "rubric_judge"]
Severity = Literal["blocker", "non_blocker"]
Verdict = Literal["PASS", "FAIL", "REVIEW"]
CropType = Literal["cotton", "soybean"]


class CriterionConfig(BaseModel):
    id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    oracle_type: OracleType
    severity: Severity
    scale_min: int
    scale_max: int
    pass_threshold: float
    scale_anchors: dict[int, str] = Field(default_factory=dict[int, str])

    @model_validator(mode="after")
    def validate_scale_and_anchors(self) -> Self:
        if self.scale_min >= self.scale_max:
            raise ValueError("scale_min must be strictly less than scale_max")
        if not (self.scale_min <= self.pass_threshold <= self.scale_max):
            raise ValueError("pass_threshold must lie within [scale_min, scale_max]")
        if (
            self.oracle_type in ("reference_judge", "rubric_judge")
            and len(self.scale_anchors) < 2
        ):
            raise ValueError("LLM judge criteria require at least two scale_anchors")
        return self


class EvaluationContract(BaseModel):
    version: str
    domain: str
    criteria: list[CriterionConfig] = Field(min_length=1)


class EvalCase(BaseModel):
    id: str = Field(min_length=1)
    crop: CropType
    category: str = Field(min_length=1)
    user_query: str = Field(min_length=1)
    protocol_context: str = Field(min_length=1)
    expected_decision: str = Field(min_length=1)
    required_facts: list[str] = Field(default_factory=list)
    forbidden_phrases: list[str] = Field(default_factory=list)
    max_words: int = Field(gt=0)


class CriterionResult(BaseModel):
    case_id: str | None = None
    criterion_id: str
    verdict: Verdict
    score: float | None = None
    evidence_quotes: list[str] = Field(default_factory=list)
    rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_evidence_requirement(self) -> Self:
        if (
            self.score is not None
            and self.verdict in ("PASS", "FAIL")
            and not self.evidence_quotes
        ):
            raise ValueError(
                "Scored PASS/FAIL verdicts require at least one evidence quote"
            )
        return self


def load_contract(path: Path) -> EvaluationContract:
    raw_data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return EvaluationContract.model_validate(raw_data)


def load_eval_cases(path: Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            cases.append(EvalCase.model_validate_json(stripped))
        except Exception as exc:
            raise ValueError(f"Invalid eval case at line {line_number}: {exc}") from exc
    return cases


class CaseEvaluation(BaseModel):
    case_id: str
    response: str
    results: list[CriterionResult]


class RunSummary(BaseModel):
    contract_version: str
    blocker_criterion_ids: list[str]
    evaluations: list[CaseEvaluation]

    @property
    def all_results(self) -> list[CriterionResult]:
        return [result for item in self.evaluations for result in item.results]

    @property
    def blocker_failures(self) -> int:
        return sum(
            1
            for result in self.all_results
            if result.verdict == "FAIL"
            and result.criterion_id in self.blocker_criterion_ids
        )

    @property
    def review_count(self) -> int:
        return sum(1 for result in self.all_results if result.verdict == "REVIEW")

    @property
    def pass_rate(self) -> float:
        results = self.all_results
        if not results:
            return 0.0
        return sum(1 for result in results if result.verdict == "PASS") / len(results)
