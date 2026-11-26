import argparse
from collections.abc import Sequence
from pathlib import Path

from rich.console import Console
from rich.table import Table

from karasu_eval.client import ModelClient, OpenAIClient, ReplayClient
from karasu_eval.models import RunSummary, load_contract, load_eval_cases
from karasu_eval.runner import evaluate_dataset

DEFAULT_CONTRACT = Path(__file__).parent / "contracts" / "agronomy_rubric.yaml"
VERDICT_STYLES = {"PASS": "green", "FAIL": "red", "REVIEW": "yellow"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="karasu-eval")
    subcommands = parser.add_subparsers(dest="command", required=True)
    run = subcommands.add_parser("run")
    run.add_argument("--prompt", type=Path, default=Path("prompts/advisor_v1.txt"))
    run.add_argument("--data", type=Path, default=Path("data/eval_cases.jsonl"))
    run.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    run.add_argument("--replay-dir", type=Path, default=Path("data/replay"))
    run.add_argument("--live", action="store_true")
    return parser


def render_summary(summary: RunSummary, console: Console) -> None:
    criterion_ids = [result.criterion_id for result in summary.evaluations[0].results]
    table = Table(title=f"Evaluation run (contract v{summary.contract_version})")
    table.add_column("case")
    for criterion_id in criterion_ids:
        table.add_column(criterion_id)
    for evaluation in summary.evaluations:
        cells = [
            f"[{VERDICT_STYLES[result.verdict]}]{result.verdict}[/]"
            for result in evaluation.results
        ]
        table.add_row(evaluation.case_id, *cells)
    console.print(table)
    console.print(
        f"pass rate: {summary.pass_rate:.0%} | "
        f"blocker failures: {summary.blocker_failures} | "
        f"needs review: {summary.review_count}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    prompt_path: Path = args.prompt
    client: ModelClient = (
        OpenAIClient() if args.live else ReplayClient(args.replay_dir, prompt_path.stem)
    )
    summary = evaluate_dataset(
        load_contract(args.contract),
        load_eval_cases(args.data),
        prompt_path.read_text(encoding="utf-8"),
        client,
    )
    render_summary(summary, Console(width=200))
    return 1 if summary.blocker_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
