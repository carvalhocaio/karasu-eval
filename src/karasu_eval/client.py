import json
import os
from pathlib import Path
from typing import Protocol

from openai import OpenAI
from pydantic import BaseModel

from karasu_eval.models import CriterionConfig, EvalCase


class JudgeRawVerdict(BaseModel):
    score: float
    evidence_quotes: list[str]
    rationale: str


class ModelClient(Protocol):
    def generate(self, system_prompt: str, case: EvalCase) -> str: ...

    def judge(
        self, criterion: CriterionConfig, case: EvalCase, response: str
    ) -> JudgeRawVerdict: ...


class ReplayClient:
    def __init__(self, replay_dir: Path, prompt_version: str) -> None:
        record_path = replay_dir / f"{prompt_version}.json"
        if not record_path.is_file():
            raise KeyError(f"No replay file for prompt version: {record_path}")
        record = json.loads(record_path.read_text(encoding="utf-8"))
        self.prompt_version = prompt_version
        self.generations: dict[str, str] = record["generations"]
        self.verdicts: dict[str, dict[str, JudgeRawVerdict]] = {
            case_id: {
                criterion_id: JudgeRawVerdict.model_validate(raw)
                for criterion_id, raw in by_criterion.items()
            }
            for case_id, by_criterion in record["verdicts"].items()
        }

    def generate(self, system_prompt: str, case: EvalCase) -> str:
        if case.id not in self.generations:
            raise KeyError(
                f"Missing replay generation for case '{case.id}' "
                f"in prompt version '{self.prompt_version}'"
            )
        return self.generations[case.id]

    def judge(
        self, criterion: CriterionConfig, case: EvalCase, response: str
    ) -> JudgeRawVerdict:
        verdict = self.verdicts.get(case.id, {}).get(criterion.id)
        if verdict is None:
            raise KeyError(
                f"Missing replay verdict for case '{case.id}' and criterion "
                f"'{criterion.id}' in prompt version '{self.prompt_version}'"
            )
        return verdict


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Environment variable {name} is required for live mode")
    return value


class OpenAIClient:
    def __init__(self) -> None:
        self.client = OpenAI(api_key=_require_env("OPENAI_API_KEY"))
        self.generator_model = _require_env("EVAL_GENERATOR_MODEL")
        self.judge_model = _require_env("EVAL_JUDGE_MODEL")

    def generate(self, system_prompt: str, case: EvalCase) -> str:
        completion = self.client.chat.completions.create(
            model=self.generator_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"Contexto do protocolo:\n{case.protocol_context}\n\n"
                        f"Pergunta:\n{case.user_query}"
                    ),
                },
            ],
        )
        return completion.choices[0].message.content or ""

    def judge(
        self, criterion: CriterionConfig, case: EvalCase, response: str
    ) -> JudgeRawVerdict:
        anchors = "\n".join(
            f"{score}: {meaning}"
            for score, meaning in sorted(criterion.scale_anchors.items())
        )
        completion = self.client.beta.chat.completions.parse(
            model=self.judge_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict agronomy evaluator. Score the response "
                        f"on criterion '{criterion.id}': {criterion.description}\n"
                        f"Scale {criterion.scale_min}-{criterion.scale_max}. "
                        f"Anchors:\n{anchors}\n"
                        "Return verbatim evidence quotes copied exactly from the "
                        "response or the protocol context. Never paraphrase quotes."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Protocol context:\n{case.protocol_context}\n\n"
                        f"Expected decision: {case.expected_decision}\n"
                        f"Required facts: {case.required_facts}\n\n"
                        f"Question:\n{case.user_query}\n\n"
                        f"Response:\n{response}"
                    ),
                },
            ],
            response_format=JudgeRawVerdict,
        )
        parsed = completion.choices[0].message.parsed
        if parsed is None:
            raise RuntimeError(f"Judge returned no parsed verdict for '{criterion.id}'")
        return parsed
