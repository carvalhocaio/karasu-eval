import argparse
from collections.abc import Sequence
from pathlib import Path

from rich.console import Console
from rich.table import Table

from karasu_eval.calibration import (
    CalibrationReport,
    compute_calibration,
    load_human_labels,
)
from karasu_eval.client import ModelClient, OpenAIClient, ReplayClient
from karasu_eval.gate import GateReport, evaluate_release_gate
from karasu_eval.models import (
    EvaluationContract,
    RunSummary,
    load_contract,
    load_eval_cases,
)
from karasu_eval.runner import evaluate_dataset

DEFAULT_CONTRACT = Path("contracts/agronomy_rubric.yaml")
DEFAULT_DATA = Path("data/eval_cases.jsonl")
DEFAULT_REPLAY_DIR = Path("data/replay")
DEFAULT_LABELS = Path("data/calibration_set.jsonl")
VERDICT_STYLES = {"PASS": "green", "FAIL": "red", "REVIEW": "yellow"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="karasu-eval")
    subcommands = parser.add_subparsers(dest="command", required=True)

    run = subcommands.add_parser("run")
    run.add_argument("--prompt", type=Path, default=Path("prompts/advisor_v1.txt"))
    run.add_argument("--data", type=Path, default=DEFAULT_DATA)
    run.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    run.add_argument("--replay-dir", type=Path, default=DEFAULT_REPLAY_DIR)
    run.add_argument("--live", action="store_true")

    calibrate = subcommands.add_parser("calibrate")
    calibrate.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    calibrate.add_argument(
        "--prompt", type=Path, default=Path("prompts/advisor_v1.txt")
    )
    calibrate.add_argument("--data", type=Path, default=DEFAULT_DATA)
    calibrate.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    calibrate.add_argument("--replay-dir", type=Path, default=DEFAULT_REPLAY_DIR)
    calibrate.add_argument("--min-kappa", type=float, default=0.75)
    calibrate.add_argument("--live", action="store_true")

    compare = subcommands.add_parser("compare")
    compare.add_argument(
        "--baseline", type=Path, default=Path("prompts/advisor_v1.txt")
    )
    compare.add_argument(
        "--candidate", type=Path, default=Path("prompts/advisor_v2.txt")
    )
    compare.add_argument("--data", type=Path, default=DEFAULT_DATA)
    compare.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    compare.add_argument("--replay-dir", type=Path, default=DEFAULT_REPLAY_DIR)
    compare.add_argument("--live", action="store_true")

    return parser


def _make_client(live: bool, replay_dir: Path, prompt_path: Path) -> ModelClient:
    if live:
        return OpenAIClient()
    return ReplayClient(replay_dir, prompt_path.stem)


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


def render_calibration(reports: list[CalibrationReport], console: Console) -> None:
    table = Table(title="Judge calibration")
    for col in (
        "criterion",
        "paired",
        "skipped",
        "TP",
        "FP",
        "TN",
        "FN",
        "TPR",
        "TNR",
        "kappa",
        "status",
    ):
        table.add_column(col)
    for report in reports:
        status = (
            "[green]CALIBRATED[/]" if report.is_calibrated else "[red]UNCALIBRATED[/]"
        )
        table.add_row(
            report.criterion_id,
            str(report.total_paired),
            str(report.skipped_review),
            str(report.tp),
            str(report.fp),
            str(report.tn),
            str(report.fn),
            f"{report.tpr:.2f}",
            f"{report.tnr:.2f}",
            f"{report.cohens_kappa:.2f}",
            status,
        )
    console.print(table)


def _criterion_pass_rates(
    contract: EvaluationContract, summary: RunSummary
) -> dict[str, float]:
    rates: dict[str, float] = {}
    for criterion in contract.criteria:
        matching = [r for r in summary.all_results if r.criterion_id == criterion.id]
        rates[criterion.id] = (
            sum(1 for r in matching if r.verdict == "PASS") / len(matching)
            if matching
            else 0.0
        )
    return rates


def render_comparison(
    contract: EvaluationContract,
    baseline: RunSummary,
    candidate: RunSummary,
    report: GateReport,
    console: Console,
) -> None:
    base_rates = _criterion_pass_rates(contract, baseline)
    cand_rates = _criterion_pass_rates(contract, candidate)
    table = Table(
        title=f"Release gate: {report.baseline_name} -> {report.candidate_name}"
    )
    table.add_column("criterion")
    table.add_column("severity")
    table.add_column(report.baseline_name)
    table.add_column(report.candidate_name)
    table.add_column("delta")
    for criterion in contract.criteria:
        base_rate = base_rates[criterion.id]
        cand_rate = cand_rates[criterion.id]
        delta = cand_rate - base_rate
        table.add_row(
            criterion.id,
            criterion.severity,
            f"{base_rate:.0%}",
            f"{cand_rate:.0%}",
            f"{delta:+.0%}",
        )
    console.print(table)
    style = VERDICT_STYLES[report.decision]
    console.print(
        f"new blocker: {len(report.new_blocker_failures)} | "
        f"new non-blocker: {len(report.new_non_blocker_failures)} | "
        f"carryover: {len(report.carryover_failures)} | "
        f"fixed: {len(report.fixed_failures)} | "
        f"review: {len(report.review_cases)}"
    )
    console.print(f"decision: [{style}]{report.decision}[/] — {report.reason}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    console = Console(width=200)
    contract = load_contract(args.contract)
    cases = load_eval_cases(args.data)

    if args.command == "run":
        prompt_path: Path = args.prompt
        summary = evaluate_dataset(
            contract,
            cases,
            prompt_path.read_text(encoding="utf-8"),
            _make_client(args.live, args.replay_dir, prompt_path),
        )
        render_summary(summary, console)
        return 1 if summary.blocker_failures else 0

    if args.command == "calibrate":
        prompt_path = args.prompt
        summary = evaluate_dataset(
            contract,
            cases,
            prompt_path.read_text(encoding="utf-8"),
            _make_client(args.live, args.replay_dir, prompt_path),
        )
        reports = compute_calibration(
            load_human_labels(args.labels),
            summary.all_results,
            min_kappa=args.min_kappa,
        )
        render_calibration(reports, console)
        return 0

    baseline_path: Path = args.baseline
    candidate_path: Path = args.candidate
    baseline = evaluate_dataset(
        contract,
        cases,
        baseline_path.read_text(encoding="utf-8"),
        _make_client(args.live, args.replay_dir, baseline_path),
    )
    candidate = evaluate_dataset(
        contract,
        cases,
        candidate_path.read_text(encoding="utf-8"),
        _make_client(args.live, args.replay_dir, candidate_path),
    )
    report = evaluate_release_gate(
        contract,
        baseline,
        candidate,
        baseline_name=baseline_path.stem,
        candidate_name=candidate_path.stem,
    )
    render_comparison(contract, baseline, candidate, report, console)
    return report.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
