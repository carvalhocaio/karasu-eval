import json
from pathlib import Path

import pytest

from karasu_eval.cli import main
from karasu_eval.client import JudgeRawVerdict, ReplayClient
from karasu_eval.models import (
    CriterionConfig,
    EvalCase,
    EvaluationContract,
    load_contract,
    load_eval_cases,
)
from karasu_eval.runner import evaluate_dataset
from tests.stubs import StubClient

ROOT = Path(__file__).parent.parent
CONTRACT_PATH = ROOT / "src" / "karasu_eval" / "contracts" / "agronomy_rubric.yaml"
DATA_PATH = ROOT / "data" / "eval_cases.jsonl"
REPLAY_DIR = ROOT / "data" / "replay"
PROMPT_PATH = ROOT / "prompts" / "advisor_v1.txt"


@pytest.fixture
def case() -> EvalCase:
    return EvalCase(
        id="cotton-001",
        crop="cotton",
        category="defoliation_window",
        user_query="Quando aplicar desfolhante?",
        protocol_context="Aplicar com 60% a 70% de maçãs abertas.",
        expected_decision="APPROVE_WITH_WINDOW",
        max_words=40,
    )


@pytest.fixture
def judge_criterion() -> CriterionConfig:
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


@pytest.fixture
def replay_dir(tmp_path: Path, case: EvalCase) -> Path:
    record = {
        "generations": {case.id: "recorded answer"},
        "verdicts": {
            case.id: {
                "protocol_correctness": {
                    "score": 5.0,
                    "evidence_quotes": ["recorded answer"],
                    "rationale": "Matches.",
                }
            }
        },
    }
    (tmp_path / "v1.json").write_text(json.dumps(record), encoding="utf-8")
    return tmp_path


def test_replay_client_loads_generation(replay_dir: Path, case: EvalCase) -> None:
    client = ReplayClient(replay_dir, "v1")
    assert client.generate("system", case) == "recorded answer"


def test_replay_client_loads_judge_verdict(
    replay_dir: Path, case: EvalCase, judge_criterion: CriterionConfig
) -> None:
    client = ReplayClient(replay_dir, "v1")
    raw = client.judge(judge_criterion, case, "recorded answer")
    assert raw.score == 5.0
    assert raw.evidence_quotes == ["recorded answer"]


def test_replay_client_missing_generation_raises_key_error(
    replay_dir: Path, case: EvalCase
) -> None:
    client = ReplayClient(replay_dir, "v1")
    missing = case.model_copy(update={"id": "ghost-001"})
    with pytest.raises(KeyError, match="ghost-001"):
        client.generate("system", missing)


def test_replay_client_missing_verdict_raises_key_error(
    replay_dir: Path, case: EvalCase, judge_criterion: CriterionConfig
) -> None:
    client = ReplayClient(replay_dir, "v1")
    other = judge_criterion.model_copy(update={"id": "agronomic_clarity"})
    with pytest.raises(KeyError, match="agronomic_clarity"):
        client.judge(other, case, "recorded answer")


def test_replay_client_missing_file_raises_key_error(tmp_path: Path) -> None:
    with pytest.raises(KeyError, match="v9"):
        ReplayClient(tmp_path, "v9")


def test_evaluate_dataset_routes_criteria_by_oracle_type(
    case: EvalCase, judge_criterion: CriterionConfig
) -> None:
    deterministic = CriterionConfig(
        id="concision",
        description="Short.",
        oracle_type="deterministic",
        severity="non_blocker",
        scale_min=0,
        scale_max=1,
        pass_threshold=1.0,
    )
    contract = EvaluationContract(
        version="t", domain="d", criteria=[judge_criterion, deterministic]
    )
    client = StubClient(
        JudgeRawVerdict(score=5.0, evidence_quotes=["short"], rationale="ok"),
        generation="short",
    )
    summary = evaluate_dataset(contract, [case], "system", client)
    assert client.judged_criteria == ["protocol_correctness"]
    results = summary.evaluations[0].results
    assert [r.criterion_id for r in results] == ["protocol_correctness", "concision"]
    assert summary.evaluations[0].response == "short"
    assert summary.blocker_failures == 0
    assert summary.pass_rate == 1.0


def test_end_to_end_replay_run_exposes_regressions() -> None:
    contract = load_contract(CONTRACT_PATH)
    cases = load_eval_cases(DATA_PATH)
    client = ReplayClient(REPLAY_DIR, "advisor_v1")
    summary = evaluate_dataset(
        contract, cases, PROMPT_PATH.read_text(encoding="utf-8"), client
    )
    assert len(cases) == 6
    assert len(summary.evaluations) == 6
    assert summary.blocker_failures >= 1
    assert summary.review_count == 1
    assert 0.0 < summary.pass_rate < 1.0


def test_cli_run_renders_summary_and_flags_blocker_failures(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(
        [
            "run",
            "--prompt",
            str(PROMPT_PATH),
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
    assert "cotton-001" in output
    assert "protocol_correctness" in output
