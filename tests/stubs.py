from karasu_eval.client import JudgeRawVerdict
from karasu_eval.models import CriterionConfig, EvalCase


class StubClient:
    def __init__(self, verdict: JudgeRawVerdict, generation: str = "stub") -> None:
        self.verdict = verdict
        self.generation = generation
        self.judged_criteria: list[str] = []

    def generate(self, system_prompt: str, case: EvalCase) -> str:
        return self.generation

    def judge(
        self, criterion: CriterionConfig, case: EvalCase, response: str
    ) -> JudgeRawVerdict:
        self.judged_criteria.append(criterion.id)
        return self.verdict
