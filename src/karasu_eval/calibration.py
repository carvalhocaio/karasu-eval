from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from karasu_eval.models import CriterionResult


class HumanLabel(BaseModel):
    case_id: str = Field(min_length=1)
    criterion_id: str = Field(min_length=1)
    expected_verdict: Literal["PASS", "FAIL"]


class CalibrationReport(BaseModel):
    criterion_id: str
    total_paired: int
    skipped_review: int
    tp: int
    fp: int
    tn: int
    fn: int
    tpr: float
    tnr: float
    cohens_kappa: float
    is_calibrated: bool


def load_human_labels(path: Path) -> list[HumanLabel]:
    labels: list[HumanLabel] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            labels.append(HumanLabel.model_validate_json(stripped))
        except Exception as exc:
            raise ValueError(
                f"Invalid human label at line {line_number}: {exc}"
            ) from exc
    return labels


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _cohens_kappa(tp: int, fp: int, tn: int, fn: int) -> float:
    total = tp + fp + tn + fn
    if total == 0:
        return 0.0
    observed = (tp + tn) / total
    expected = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (total * total)
    if expected >= 1.0:
        return 0.0
    return (observed - expected) / (1.0 - expected)


def compute_calibration(
    human_labels: list[HumanLabel],
    predictions: list[CriterionResult],
    min_kappa: float = 0.75,
) -> list[CalibrationReport]:
    pred_index = {
        (pred.case_id, pred.criterion_id): pred
        for pred in predictions
        if pred.case_id is not None
    }
    labels_by_criterion: dict[str, list[HumanLabel]] = {}
    for item in human_labels:
        labels_by_criterion.setdefault(item.criterion_id, []).append(item)

    reports: list[CalibrationReport] = []
    for criterion_id, items in labels_by_criterion.items():
        tp = fp = tn = fn = skipped_review = 0
        for item in items:
            pred = pred_index.get((item.case_id, item.criterion_id))
            if pred is None:
                continue
            if pred.verdict == "REVIEW":
                skipped_review += 1
                continue
            if item.expected_verdict == "PASS" and pred.verdict == "PASS":
                tp += 1
            elif item.expected_verdict == "FAIL" and pred.verdict == "PASS":
                fp += 1
            elif item.expected_verdict == "FAIL" and pred.verdict == "FAIL":
                tn += 1
            else:
                fn += 1

        kappa = _cohens_kappa(tp, fp, tn, fn)
        reports.append(
            CalibrationReport(
                criterion_id=criterion_id,
                total_paired=tp + fp + tn + fn,
                skipped_review=skipped_review,
                tp=tp,
                fp=fp,
                tn=tn,
                fn=fn,
                tpr=_safe_rate(tp, tp + fn),
                tnr=_safe_rate(tn, tn + fp),
                cohens_kappa=kappa,
                is_calibrated=kappa >= min_kappa,
            )
        )
    return reports
