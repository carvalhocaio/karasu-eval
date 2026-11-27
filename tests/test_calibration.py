from pathlib import Path

import pytest

from karasu_eval.calibration import (
    HumanLabel,
    compute_calibration,
    load_human_labels,
)
from karasu_eval.models import CriterionResult, Verdict


def label(case_id: str, criterion_id: str, expected: Verdict) -> HumanLabel:
    assert expected in ("PASS", "FAIL")
    return HumanLabel(
        case_id=case_id,
        criterion_id=criterion_id,
        expected_verdict=expected,
    )


def pred(case_id: str, criterion_id: str, verdict: Verdict) -> CriterionResult:
    return CriterionResult(
        case_id=case_id,
        criterion_id=criterion_id,
        verdict=verdict,
        score=None if verdict == "REVIEW" else (5.0 if verdict == "PASS" else 1.0),
        evidence_quotes=[] if verdict == "REVIEW" else ["quote"],
        rationale="Judged.",
    )


def test_load_human_labels_parses_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "labels.jsonl"
    path.write_text(
        '{"case_id": "c1", "criterion_id": "p", "expected_verdict": "PASS"}\n'
        "\n"
        '{"case_id": "c2", "criterion_id": "p", "expected_verdict": "FAIL"}\n',
        encoding="utf-8",
    )
    labels = load_human_labels(path)
    assert len(labels) == 2
    assert labels[0].case_id == "c1"
    assert labels[1].expected_verdict == "FAIL"


def test_load_human_labels_invalid_line_raises_value_error(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"case_id": "c1"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        load_human_labels(path)


def test_compute_calibration_perfect_agreement() -> None:
    labels = [
        label("c1", "protocol_correctness", "PASS"),
        label("c2", "protocol_correctness", "PASS"),
        label("c3", "protocol_correctness", "FAIL"),
        label("c4", "protocol_correctness", "FAIL"),
    ]
    preds = [
        pred("c1", "protocol_correctness", "PASS"),
        pred("c2", "protocol_correctness", "PASS"),
        pred("c3", "protocol_correctness", "FAIL"),
        pred("c4", "protocol_correctness", "FAIL"),
    ]
    reports = compute_calibration(labels, preds, min_kappa=0.75)
    assert len(reports) == 1
    report = reports[0]
    assert report.criterion_id == "protocol_correctness"
    assert report.total_paired == 4
    assert report.skipped_review == 0
    assert (report.tp, report.fp, report.tn, report.fn) == (2, 0, 2, 0)
    assert report.tpr == pytest.approx(1.0)
    assert report.tnr == pytest.approx(1.0)
    assert report.cohens_kappa == pytest.approx(1.0)
    assert report.is_calibrated is True


def test_compute_calibration_excludes_review_from_matrix() -> None:
    labels = [
        label("c1", "agronomic_clarity", "PASS"),
        label("c2", "agronomic_clarity", "FAIL"),
        label("c3", "agronomic_clarity", "PASS"),
    ]
    preds = [
        pred("c1", "agronomic_clarity", "PASS"),
        pred("c2", "agronomic_clarity", "FAIL"),
        pred("c3", "agronomic_clarity", "REVIEW"),
    ]
    reports = compute_calibration(labels, preds, min_kappa=0.75)
    report = reports[0]
    assert report.total_paired == 2
    assert report.skipped_review == 1
    assert (report.tp, report.fp, report.tn, report.fn) == (1, 0, 1, 0)
    assert report.cohens_kappa == pytest.approx(1.0)
    assert report.is_calibrated is True


def test_compute_calibration_zero_denominator_returns_zero_kappa() -> None:
    labels = [
        label("c1", "protocol_correctness", "PASS"),
        label("c2", "protocol_correctness", "PASS"),
    ]
    preds = [
        pred("c1", "protocol_correctness", "PASS"),
        pred("c2", "protocol_correctness", "PASS"),
    ]
    report = compute_calibration(labels, preds, min_kappa=0.75)[0]
    assert report.tpr == pytest.approx(1.0)
    assert report.tnr == pytest.approx(0.0)
    assert report.cohens_kappa == pytest.approx(0.0)
    assert report.is_calibrated is False


def test_compute_calibration_all_review_returns_zero_metrics() -> None:
    labels = [label("c1", "protocol_correctness", "PASS")]
    preds = [pred("c1", "protocol_correctness", "REVIEW")]
    report = compute_calibration(labels, preds)[0]
    assert report.total_paired == 0
    assert report.skipped_review == 1
    assert report.tpr == 0.0
    assert report.tnr == 0.0
    assert report.cohens_kappa == 0.0
    assert report.is_calibrated is False


def test_compute_calibration_partial_agreement_and_threshold() -> None:
    labels = [
        label("c1", "crit", "PASS"),
        label("c2", "crit", "PASS"),
        label("c3", "crit", "PASS"),
        label("c4", "crit", "FAIL"),
        label("c5", "crit", "FAIL"),
    ]
    preds = [
        pred("c1", "crit", "PASS"),
        pred("c2", "crit", "PASS"),
        pred("c3", "crit", "FAIL"),
        pred("c4", "crit", "FAIL"),
        pred("c5", "crit", "FAIL"),
    ]
    # tp=2, fp=0, tn=2, fn=1 => po=0.8, pe=(2*3 + 3*2)/25=12/25=0.48
    # kappa = (0.8 - 0.48) / (1 - 0.48) = 0.32 / 0.52 = 8/13 ~= 0.61538
    report = compute_calibration(labels, preds, min_kappa=0.75)[0]
    assert (report.tp, report.fp, report.tn, report.fn) == (2, 0, 2, 1)
    assert report.tpr == pytest.approx(2 / 3)
    assert report.tnr == pytest.approx(1.0)
    assert report.cohens_kappa == pytest.approx(8 / 13)
    assert report.is_calibrated is False
