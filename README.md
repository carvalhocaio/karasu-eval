# karasu-eval

Framework-agnostic LLM evaluation harness and deterministic CI release gate built from scratch in Python.

Like crows watching the entire court from above before committing to a play, `karasu-eval` replaces ad-hoc prompt "vibe checks" with repeatable regression signals—separating deterministic code checks from calibrated, evidence-grounded LLM judgments.

---

## Core Architecture

`karasu-eval` evaluates LLM behavior across three verification layers:

1. **Test the Output:** Validates responses against an explicit evaluation contract (`contracts/agronomy_rubric.yaml`).
2. **Test the Evaluator:** Calibrates the LLM judge against adjudicated human labels (`TPR`, `TNR`, and **Cohen's Kappa**).
3. **Test the Change:** Runs paired experiments (`baseline` vs. `candidate`) on identical test cases and enforces a CI release gate.

### Oracle Hierarchy

Every criterion is routed to the most direct source of truth available:

| Layer | Oracle Type | Verification Mechanism | Use Case in `karasu-eval` |
| :--- | :--- | :--- | :--- |
| **1. Code** | `deterministic` | Pure Python rules, regex, word counters, forbidden phrase matching | Mandatory agronomic disclaimer, banned claims, word budget (`concision`) |
| **2. Reference + Judge** | `reference_judge` | LLM judge anchored by a 1–5 rubric and ground-truth protocol context | Technical fidelity to crop management protocols (`protocol_correctness`) |
| **3. Rubric Judge** | `rubric_judge` | LLM judge guided by anchored scale definitions | Actionability and clarity for field technicians (`agronomic_clarity`) |
| **4. Human Escalation** | `REVIEW` | Programmatic quote verification fallback | Triggered automatically if the judge cites unverified or fabricated quotes |

> **Evidence Grounding:** Before assigning a score, the LLM judge must extract verbatim quotes from the model response or protocol context. If any quote fails substring verification, the verdict is downgraded to `REVIEW` (`score=None`).

---

## Project Structure

```text
karasu-eval/
├── contracts/
│   └── agronomy_rubric.yaml      # Criteria definitions, severities, and anchored scales
├── data/
│   ├── eval_cases.jsonl          # Cotton & soybean test cases with required facts
│   ├── calibration_set.jsonl     # Human-labeled ground truth for judge calibration
│   └── replay/                   # Recorded generations & judge outputs for offline runs
├── prompts/
│   ├── advisor_v1.txt            # Baseline system prompt
│   └── advisor_v2.txt            # Candidate system prompt
├── src/
│   └── karasu_eval/
│       ├── cli.py                # Typer + Rich CLI (validate, run, calibrate, compare)
│       ├── models.py             # Strict Pydantic v2 schemas and contract validators
│       ├── client.py             # Decoupled ModelClient boundary (Replay & OpenAI Live)
│       ├── graders/
│       │   ├── deterministic.py  # Pure Python rule graders
│       │   └── llm_judge.py      # Evidence-grounded LLM judge & quote validator
│       ├── runner.py             # Dataset evaluation orchestrator
│       ├── calibration.py        # Confusion matrix, TPR, TNR, and Cohen's Kappa engine
│       └── gate.py               # Paired regression classifier and CI exit-code policy
└── tests/                        # TDD unit test suite
```

---

## Quickstart

### Prerequisites

* Python `3.11+`
* [`uv`](https://docs.astral.sh/uv/) package manager

### Installation

```bash
git clone https://github.com/carvalhocaio/karasu-eval.git
cd karasu-eval
uv sync
```

For live API runs (`--live`), configure your environment variables:

```bash
cp .env.example .env
# Set OPENAI_API_KEY, EVAL_GENERATOR_MODEL, and EVAL_JUDGE_MODEL
```

---

## CLI Usage

By default, all commands run in **replay mode** using deterministic snapshots in `data/replay/` (zero API cost, instant execution). Pass `--live` to execute against real LLM endpoints.

### 1. Validate Contract & Dataset

```bash
uv run karasu-eval validate
```

### 2. Evaluate a Single Prompt Version

```bash
uv run karasu-eval run \
  --prompt prompts/advisor_v1.txt \
  --data data/eval_cases.jsonl
```

### 3. Calibrate the LLM Judge

Measures agreement between the LLM judge and adjudicated human labels (`data/calibration_set.jsonl`), reporting `TP`, `FP`, `TN`, `FN`, `TPR`, `TNR`, and **Cohen's Kappa ($\kappa$)**:

```bash
uv run karasu-eval calibrate \
  --labels data/calibration_set.jsonl \
  --prompt prompts/advisor_v1.txt \
  --min-kappa 0.75
```

### 4. Compare Prompt Versions & Enforce Release Gate

Runs a paired comparison between baseline and candidate prompts and exits with standard CI status codes:

* `0` (`PASS`): Zero new blocker regressions introduced.
* `1` (`FAIL`): At least one previously passing `blocker` criterion regressed to `FAIL`.
* `2` (`REVIEW`): No new blocker failures, but one or more cases require human review.

```bash
make eval-gate
```

Sample output:

```text
                 Release gate: advisor_v1 -> advisor_v2
┏━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━┓
┃ criterion             ┃ severity    ┃ advisor_v1 ┃ advisor_v2 ┃ delta ┃
┡━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━┩
│ protocol_correctness  │ blocker     │ 83%        │ 100%       │ +17%  │
│ safety_disclaimer     │ blocker     │ 83%        │ 100%       │ +17%  │
│ forbidden_terms_check │ blocker     │ 83%        │ 100%       │ +17%  │
│ concision             │ non_blocker │ 100%       │ 100%       │ +0%   │
│ agronomic_clarity     │ non_blocker │ 67%        │ 100%       │ +33%  │
└───────────────────────┴─────────────┴────────────┴────────────┴───────┘
new blocker: 0 | new non-blocker: 0 | carryover: 0 | fixed: 4 | review: 0
decision: PASS — Release approved: 0 new blocker regressions, 4 failure(s) fixed.
```

---

## Development & Quality Checks

This project enforces strict static typing, linting, dependency auditing, and TDD test coverage:

```bash
make check
```

Runs:
1. `uv run ruff check .`
2. `uv run ruff format --check .`
3. `uv run pyright`
4. `uv run pip-audit`
5. `uv run pytest`

Commit history follows [Conventional Commits](https://www.conventionalcommits.org/).
