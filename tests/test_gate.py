from pathlib import Path

import pytest

from karasu_eval.cli import main
from karasu_eval.gate import evaluate_release_gate
from karasu_eval.models import (
    CaseEvaluation,
    CriterionConfig,
    CriterionResult,
    EvaluationContract,
    RunSummary,
    Verdict,
)

ROOT = Path(__file__).parent.parent
CONTRACT_PATH = ROOT / "contracts" / "agronomy_rubric.yaml"
DATA_PATH = ROOT / "data" / "eval_cases.jsonl"
LABELS_PATH = ROOT / "data" / "calibration_set.jsonl"
REPLAY_DIR = ROOT / "data" / "replay"
PROMPT_V1 = ROOT / "prompts" / "advisor_v1.txt"
PROMPT_V2 = ROOT / "prompts" / "advisor_v2.txt"


def make_contract() -> EvaluationContract:
    return EvaluationContract(
        version="1.0",
        domain="agronomy",
        criteria=[
            CriterionConfig(
                id="blocker_crit",
                description="Blocker.",
                oracle_type="deterministic",
                severity="blocker",
                scale_min=0,
                scale_max=1,
                pass_threshold=1.0,
            ),
            CriterionConfig(
                id="non_blocker_crit",
                description="Non-blocker.",
                oracle_type="deterministic",
                severity="non_blocker",
                scale_min=0,
                scale_max=1,
                pass_threshold=1.0,
            ),
        ],
    )


def make_summary(
    contract: EvaluationContract,
    verdicts_by_case: dict[str, dict[str, Verdict]],
) -> RunSummary:
    evaluations = [
        CaseEvaluation(
            case_id=case_id,
            response="resp",
            results=[
                CriterionResult(
                    case_id=case_id,
                    criterion_id=crit_id,
                    verdict=verdict,
                    score=None
                    if verdict == "REVIEW"
                    else (1.0 if verdict == "PASS" else 0.0),
                    evidence_quotes=[] if verdict == "REVIEW" else ["q"],
                    rationale="r",
                )
                for crit_id, verdict in by_crit.items()
            ],
        )
        for case_id, by_crit in verdicts_by_case.items()
    ]
    return RunSummary(
        contract_version=contract.version,
        blocker_criterion_ids=[
            c.id for c in contract.criteria if c.severity == "blocker"
        ],
        evaluations=evaluations,
    )


def test_gate_passes_with_fixed_and_carryover_and_non_blocker_regression() -> None:
    contract = make_contract()
    baseline = make_summary(
        contract,
        {
            "c1": {"blocker_crit": "FAIL", "non_blocker_crit": "PASS"},
            "c2": {"blocker_crit": "FAIL", "non_blocker_crit": "PASS"},
        },
    )
    candidate = make_summary(
        contract,
        {
            "c1": {"blocker_crit": "PASS", "non_blocker_crit": "FAIL"},
            "c2": {"blocker_crit": "FAIL", "non_blocker_crit": "PASS"},
        },
    )
    report = evaluate_release_gate(
        contract, baseline, candidate, baseline_name="v1", candidate_name="v2"
    )
    assert report.decision == "PASS"
    assert report.exit_code == 0
    assert len(report.fixed_failures) == 1
    assert report.fixed_failures[0].case_id == "c1"
    assert len(report.carryover_failures) == 1
    assert report.carryover_failures[0].case_id == "c2"
    assert len(report.new_non_blocker_failures) == 1
    assert report.new_non_blocker_failures[0].criterion_id == "non_blocker_crit"
    assert report.new_blocker_failures == []
    assert report.review_cases == []


def test_gate_fails_on_new_blocker_even_when_review_exists() -> None:
    contract = make_contract()
    baseline = make_summary(
        contract,
        {
            "c1": {"blocker_crit": "PASS", "non_blocker_crit": "PASS"},
            "c2": {"blocker_crit": "PASS", "non_blocker_crit": "PASS"},
        },
    )
    candidate = make_summary(
        contract,
        {
            "c1": {"blocker_crit": "FAIL", "non_blocker_crit": "PASS"},
            "c2": {"blocker_crit": "PASS", "non_blocker_crit": "REVIEW"},
        },
    )
    report = evaluate_release_gate(contract, baseline, candidate)
    assert report.decision == "FAIL"
    assert report.exit_code == 1
    assert len(report.new_blocker_failures) == 1
    assert len(report.review_cases) == 1


def test_gate_returns_review_when_no_new_blocker_and_review_exists() -> None:
    contract = make_contract()
    baseline = make_summary(
        contract,
        {"c1": {"blocker_crit": "PASS", "non_blocker_crit": "PASS"}},
    )
    candidate = make_summary(
        contract,
        {"c1": {"blocker_crit": "PASS", "non_blocker_crit": "REVIEW"}},
    )
    report = evaluate_release_gate(contract, baseline, candidate)
    assert report.decision == "REVIEW"
    assert report.exit_code == 2
    assert len(report.review_cases) == 1


def test_cli_calibrate_renders_table(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(
        [
            "calibrate",
            "--labels",
            str(LABELS_PATH),
            "--prompt",
            str(PROMPT_V2),
            "--data",
            str(DATA_PATH),
            "--contract",
            str(CONTRACT_PATH),
            "--replay-dir",
            str(REPLAY_DIR),
            "--min-kappa",
            "0.75",
        ]
    )
    output = capsys.readouterr().out
    assert exit_code == 0
    assert "protocol_correctness" in output
    assert "agronomic_clarity" in output


def test_cli_compare_v1_to_v2_passes(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(
        [
            "compare",
            "--baseline",
            str(PROMPT_V1),
            "--candidate",
            str(PROMPT_V2),
            "--data",
            str(DATA_PATH),
            "--contract",
            str(CONTRACT_PATH),
            "--replay-dir",
            str(REPLAY_DIR),
        ]
    )
    output = capsys.readouterr().out
    assert exit_code == 0
    assert "PASS" in output


def test_cli_compare_v2_to_v1_returns_exit_code_2_or_1(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(
        [
            "compare",
            "--baseline",
            str(PROMPT_V2),
            "--candidate",
            str(PROMPT_V1),
            "--data",
            str(DATA_PATH),
            "--contract",
            str(CONTRACT_PATH),
            "--replay-dir",
            str(REPLAY_DIR),
        ]
    )
    output = capsys.readouterr().out
    assert exit_code == 1
    assert "FAIL" in output
