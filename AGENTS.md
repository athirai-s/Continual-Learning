# Agent Working Rules

These rules exist so a long-running agent does not have to rely on distant chat context to remember the merge standard.

## Durable Sources Of Truth

Before starting substantial work, read:
- `ROADMAP.md` for milestone order, PR contracts, merge expectations, and testing strategy
- `README.md` or `docs/testing.md` for local workflow once those docs exist

If a rule in chat conflicts with repo state, update repo state in the same PR or stop and clarify.

## Active Plan Mirror

At the start of any non-trivial task, copy a compact version of the following checklist into the active plan or TODOs for the current session:

1. Roadmap box and scope are explicit.
2. Required tests for this box are known before editing.
3. Supported entrypoint remains the tested path.
4. Artifact, checkpoint, config, or runner contract changes require test and doc updates.
5. No merge until required CI is green.

If the task changes shape, refresh the active plan so these items stay near the working context.

## Merge Discipline

Do not treat human review as the primary regression defense.

Before merge:
- map the PR to a roadmap box or update the roadmap first
- add or update tests for every behavior change
- update docs and contract tests when artifact or runner behavior changes
- verify required checks with `gh pr checks`
- merge with `gh pr merge` only after required checks pass

Do not merge around failing required checks just because the change looks safe.

## Drift Control

If a bug escapes:
- add a permanent regression test
- update the roadmap or docs if the missing expectation was process-related

If a PR changes the merge contract, test pyramid, or supported entrypoint:
- update `ROADMAP.md` and any relevant contributor docs in the same PR
