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

Run the tiny offline smoke path through the supported entrypoint:

```bash
uv run python main.py --mode synthetic --run-id local-smoke --checkpoint-dir /tmp/casf-smoke
```

## Supported Entrypoints

- `main.py` is the supported top-level training entrypoint.
- `run_job.sh` is the supported batch launcher and must keep pointing at `main.py --mode real`.
- `3B_train.py` is legacy experimental code and is not part of the supported path.

## Required Merge Contract

Before merge:

1. The PR maps to a specific roadmap box, or updates `ROADMAP.md` first if the roadmap needs to change.
2. The PR description states the problem, behavior change, tests, artifact impact, and non-goals.
3. Every behavior change is covered by tests at the right layer.
4. If checkpoint, config, runner, dataset, or launcher contracts change, the tests and docs for those contracts change in the same PR.
5. Required CI is green on the PR branch. Today that required gate is `fast-tests`.

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
- Keep the supported entrypoint under smoke-test coverage whenever launch behavior changes.
