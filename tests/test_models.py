import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from karasu_eval.models import (
    CriterionConfig,
    CriterionResult,
    EvalCase,
    load_contract,
    load_eval_cases,
)


def test_criterion_config_rejects_threshold_outside_scale() -> None:
    with pytest.raises(ValidationError):
        CriterionConfig(
            id="protocol_correctness",
            description="Fidelity to agronomic protocol",
            oracle_type="reference_judge",
            severity="blocker",
            scale_min=1,
            scale_max=5,
            pass_threshold=6.0,
            scale_anchors={1: "Wrong protocol", 5: "Exact protocol"},
        )


def test_criterion_config_requires_anchors_for_llm_judges() -> None:
    with pytest.raises(ValidationError):
        CriterionConfig(
            id="agronomic_clarity",
            description="Clarity for field technician",
            oracle_type="rubric_judge",
            severity="non_blocker",
            scale_min=1,
            scale_max=5,
            pass_threshold=4.0,
            scale_anchors={},
        )


def test_eval_case_normalizes_and_validates_fields() -> None:
    case = EvalCase(
        id="cotton-001",
        crop="cotton",
        category="defoliation_window",
        user_query="Quando aplicar desfolhante no algodão?",
        protocol_context="Aplicar quando 60% a 70% das maçãs estiverem abertas.",
        expected_decision="APPROVE_WITH_WINDOW",
        required_facts=["60% a 70% das maçãs abertas"],
        forbidden_phrases=["garantia de produtividade"],
        max_words=120,
    )

    assert case.crop == "cotton"
    assert case.max_words == 120


def test_criterion_result_requires_evidence_when_judged_pass_or_fail() -> None:
    with pytest.raises(ValidationError):
        CriterionResult(
            criterion_id="protocol_correctness",
            verdict="PASS",
            score=5.0,
            evidence_quotes=[],
            rationale="Matches protocol.",
        )


def test_load_contract_and_cases_from_files(tmp_path: Path) -> None:
    contract_file = tmp_path / "rubric.yaml"
    contract_file.write_text(
        """
version: "1.0"
domain: "agronomy"
criteria:
  - id: safety_disclaimer
    description: "Must advise consulting an agronomist"
    oracle_type: deterministic
    severity: blocker
    scale_min: 0
    scale_max: 1
    pass_threshold: 1.0
    scale_anchors: {}
""".strip(),
        encoding="utf-8",
    )

    cases_file = tmp_path / "cases.jsonl"
    case_payload = {
        "id": "soy-001",
        "crop": "soybean",
        "category": "sanitary_break",
        "user_query": "Posso plantar soja em julho no MT?",
        "protocol_context": "Vazio sanitário proibindo cultivo.",
        "expected_decision": "DENY",
        "required_facts": ["vazio sanitário"],
        "forbidden_phrases": [],
        "max_words": 100,
    }
    cases_file.write_text(
        json.dumps(case_payload, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    contract = load_contract(contract_file)
    cases = load_eval_cases(cases_file)

    assert contract.domain == "agronomy"
    assert len(contract.criteria) == 1
    assert len(cases) == 1
    assert cases[0].id == "soy-001"
