# Testing And Merge Rules

This repo treats CI as the primary defense against behavioral regressions.

## Local Commands

Install dependencies, including the test group:

```bash
uv sync --group dev
```

Run the required fast suite locally:

```bash
uv run pytest -q
```

Run the same four CI lanes locally:

```bash
uv run pytest tests/unit -q
uv run pytest tests/contracts -q
uv run pytest tests/integration -q
uv run pytest tests/smoke -q
```

Run the tiny offline smoke path through the supported entrypoint:

```bash
uv run python main.py --mode synthetic --run-id local-smoke --checkpoint-dir /tmp/casf-smoke
```

## Supported Entrypoints

- `main.py` is the primary supported training entrypoint.
- `run_job.sh` is the supported SLURM wrapper contract and is exercised in CI in synthetic mode through `CONTINUAL_LEARNING_*` environment overrides.
- `experiments/legacy/3B_train.py` is legacy experimental code and is not part of the supported path.
- The defaults in `run_job.sh` are cluster-specific examples, not portability guarantees.

## Checkpoint Resume Contract

Before Milestone 2, checkpoint and resume only claim metadata-level restore:

- `checkpoint()` writes model, tokenizer, memory registry, config, and `last_period` metadata.
- `resume()` restores `memory_registry.json` and `last_period.txt`.
- The smoke test proves metadata restoration plus non-crashing continuation on a fresh trainer instance.
- Full training-state recovery, optimizer/scheduler restore, and interruption-safe recovery are deferred to Milestone 2.

## Required Merge Contract

Before merge:

1. The PR maps to a specific roadmap box, or updates `ROADMAP.md` first if the roadmap needs to change.
2. The PR description states the problem, behavior change, tests, artifact impact, and non-goals.
3. Every behavior change is covered by tests at the right layer.
4. If checkpoint, config, runner, dataset, or launcher contracts change, the tests and docs for those contracts change in the same PR.
5. Required CI is green on the PR branch. The required behavioral suite is `unit`, `contract`, `integration`, and `smoke`.

Use GitHub checks, not guesswork:

```bash
gh pr checks <pr-number>
```

Merge through the pull request after the required checks pass:

```bash
gh pr merge <pr-number> --squash --delete-branch
```

## Bug-To-Test Rule

If a bug escapes, the fix must add a permanent regression test in the same PR unless the behavior cannot be exercised in required CI. If it cannot be exercised in required CI, do not claim it as guaranteed coverage.

## Scope Rules For Trainer Work

- Keep one roadmap box per PR.
- Do not weaken CI or remove tests without replacing them with stronger coverage.
- Prefer synthetic, offline fixtures for required CI coverage.
- Keep the supported entrypoint and SLURM wrapper contract under smoke-test coverage whenever launch behavior changes.
