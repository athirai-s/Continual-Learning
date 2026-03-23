# Training Loop Roadmap

This roadmap is for turning the current repo into a training system that is trustworthy, testable in CI, and useful for repeated continual-learning experiments.

The immediate priority is not "add more trainer features." The immediate priority is to make behavioral regressions hard to ship and to make future PRs independently verifiable by tests.

## How To Use This Roadmap

- Another single-threaded agent should pick the next unticked PR in order.
- Each PR is expected to be small enough to review and test in isolation.
- A PR is not complete unless its acceptance criteria and required tests are satisfied.
- No PR should weaken CI or delete tests without replacing them with stronger coverage.
- If a PR discovers that this roadmap is missing a prerequisite, update this document in the same PR before implementation.
- Before non-trivial work, mirror the compact active checklist from `AGENTS.md` into the session plan or TODOs so merge-critical expectations stay near the current working context.

## Non-Negotiable Principles

1. CI is load-bearing for behavioral correctness. Human review is not the primary regression defense.
2. Fast tests must use local synthetic fixtures. Pulling remote models or datasets in required CI is not acceptable.
3. Every escaped bug should create a permanent regression test.
4. The trainer is only valuable if it is both correct and decision-useful.
5. Early milestones should bias toward trustworthiness and observability over feature breadth.

## Merge Policy

The branch should not be considered mergeable unless required CI checks pass.

Required checks after Milestone 1:
- unit and contract tests
- CPU integration tests
- tiny end-to-end smoke test
- checkpoint metadata resume smoke test

Allowed optional checks:
- longer dataset adapter checks against real data
- performance tracking jobs
- GPU smoke jobs

Optional checks may inform work, but required checks gate merges.

## Pre-Merge Expectations

Before merging a PR, all of the following should be true:

1. Scope is grounded.
- The PR maps to a specific roadmap box, or it updates the roadmap first if the dependency graph changed.
- The PR description states the problem, proposed change, acceptance criteria, required tests, and non-goals.

2. Behavior is proved, not asserted.
- Every behavior change is covered by tests at the right layer.
- New bugs found during the PR add a permanent regression test before merge.
- If behavior cannot be tested in required CI, the PR should not claim it as guaranteed.

3. Required CI is green.
- Required checks must pass on the PR branch.
- Flaky or quarantined tests do not count as evidence unless the quarantine policy says otherwise.
- Optional jobs may inform the merge decision, but they do not replace required checks.

4. Artifacts and contracts are updated.
- If the PR changes checkpoint contents, artifact schema, config schema, or runner behavior, the relevant contract tests and docs are updated in the same PR.
- If the PR changes the supported entrypoint or launch path, smoke tests must cover the supported path before merge.

5. Merge is done through the PR, not by bypass.
- Use `gh pr checks` to confirm required checks are green.
- Merge with `gh pr merge` only after the merge contract above is satisfied.
- Do not merge around failing required checks just because the change "looks safe."

## How We Prevent Future Models From Forgetting

Do not rely on chat history to preserve standards. Put the standards in repo and platform surfaces that future agents will actually read or be blocked by.

Required institutional memory:

1. Repo-local sources of truth
- `ROADMAP.md` for milestone order, PR contracts, and merge expectations
- `AGENTS.md` for always-on repo instructions
- `README.md` or `docs/testing.md` for local developer workflow

2. PR-time forcing functions
- `.github/pull_request_template.md` should require:
  - roadmap box
  - behavior change
  - tests added or updated
  - artifact or checkpoint impact
  - non-goals
- CODEOWNERS or required review rules should cover the critical surfaces if the team wants human signoff in addition to CI

3. Merge-time forcing functions
- branch protection should require the fast behavioral CI suite
- the supported entrypoint smoke test must be a required check
- merges should happen through `gh pr merge` against protected branches rather than direct branch pushes

4. Drift control
- if a PR changes the merge contract, test pyramid, or supported entrypoint, it must update this roadmap and the corresponding docs in the same PR
- if a bug escapes, the follow-up PR must add the missing regression test and, if needed, update the roadmap or contributor docs

The goal is simple: a future model should be able to read the repo, inspect CI, and infer the correct merge standard without needing this conversation.

## Active Checklist For Long Sessions

This is the compact checklist that should be mirrored into the active plan or TODOs during non-trivial work:

1. Roadmap box and scope are explicit.
2. Required tests for this box are known before editing.
3. Supported entrypoint remains the tested path.
4. Artifact, checkpoint, config, or runner contract changes require test and doc updates.
5. No merge until required CI is green.

This checklist is not a substitute for the full merge policy above. Its purpose is to keep the critical expectations near the working set during long sessions.

## Test Pyramid For This Repo

Behavior-first test layers:

1. Unit tests
- Pure logic with no filesystem, network, or model downloads.
- Examples: `MemoryRegistry`, `ContradictionDetector`, passage filtering, config validation, small trainer helpers.

2. Contract tests
- Assert stable interfaces and artifact shapes.
- Only cover public seams that future PRs are expected to change independently.
- Examples: every `TemporalDataset` implementation satisfies the dataset contract; checkpoints contain required files; metrics logs have a stable schema.

3. Integration tests
- Exercise multiple modules together on local synthetic fixtures.
- Examples: tiny training loop on CPU, checkpoint artifact generation, period transition updating memory registry, evaluation against a saved checkpoint.

4. End-to-end smoke tests
- Run the actual top-level training entrypoint on a tiny local setup.
- Must be deterministic enough to catch broken wiring.
- Must not require network access or large models.

This repo does not currently need full production-style E2E testing. The closest equivalent is a tiny deterministic full-pipeline smoke test, and that should be mandatory in CI.

## Current Baseline

The current repo has:
- a custom trainer in `trainer.py`
- config scaffolding in `train_config.py`
- a train entrypoint in `main.py`
- a separate older experiment path in `3B_train.py`
- dataset adapters under `casf_dataset_api/download_dataset_scripts/data/`

The biggest gaps are:
- no meaningful automated tests
- no CI safety net
- incomplete checkpoint/resume semantics
- little structured observability
- no trustworthy tiny offline path for smoke-testing the real training loop

## Milestone Overview

- [x] Milestone 0: Testable Project Skeleton
- [x] Milestone 1: Behavioral Safety Net In CI
- [ ] Milestone 2: Trustworthy Checkpoint And Resume
- [ ] Milestone 3: Reproducible And Informative Experiments
- [ ] Milestone 4: Measurement-Led Performance Work
- [ ] Milestone 5: Continual-Learning Training Quality

## Milestone 0: Testable Project Skeleton

### Goal

Create the local fixtures, test harness, and top-level entrypoint structure needed to test the actual training loop in CI without network or heavyweight model dependencies.

### Delivered Value

After this milestone, future trainer work can be validated automatically instead of relying on manual runs.

### Exit Criteria

- `pytest` runs successfully in repo CI.
- The repo has a local synthetic dataset fixture and a tiny local model/tokenizer fixture.
- The training entrypoint can be executed in a tiny offline test mode.
- Required CI can run without downloading a model or dataset.

### PR M0.1: Add Test Harness And CI Skeleton

- [x] `M0.1` Add `pytest`-based test harness, test layout, and required CI workflow.
- Problem:
  The repo currently has no automated regression barrier.
- Proposed change:
  Add `tests/`, `pytest` configuration, shared test utilities, and a GitHub Actions workflow that runs the fast suite on every PR.
- Likely files:
  `pyproject.toml`, `.github/workflows/ci.yml`, `tests/conftest.py`, `tests/smoke/test_repo_sanity.py`
- Acceptance criteria:
  - `uv run pytest -q` works in a clean environment.
  - CI runs the fast suite on pull requests.
  - The workflow is written so new test files are picked up automatically.
  - The workflow does not require secrets, GPUs, or network-only fixtures.
- Required tests:
  - a repo sanity test that imports the local modules needed by the fast suite
  - a smoke test that the test helpers load
- Verification:
  - `uv run pytest -q`
- Non-goals:
  - no trainer behavior assertions yet
  - no coverage threshold yet

### PR M0.2: Add Synthetic Fixtures For Offline Training Tests

- [x] `M0.2` Introduce synthetic dataset, tokenizer, and tiny causal LM fixtures for CPU-only tests.
- Problem:
  Required CI cannot depend on Hugging Face downloads or large real datasets.
- Proposed change:
  Create local fixtures that satisfy the trainer's expectations:
  - a `SyntheticTemporalDataset`
  - a lightweight tokenizer stub or tiny local tokenizer fixture
  - a tiny causal LM fixture that can train for a few steps on CPU
- Likely files:
  `tests/helpers/synthetic_dataset.py`, `tests/helpers/fake_tokenizer.py`, `tests/helpers/tiny_lm.py`, `tests/conftest.py`
- Acceptance criteria:
  - fixtures require no network access
  - fixtures are fast enough for required CI
  - the synthetic dataset can represent changed and unchanged probes plus train passages
  - the tiny LM supports forward pass, loss, and save/load needed by smoke tests
- Required tests:
  - fixture construction tests
  - one trainer-adjacent test proving the fixtures are usable together
- Verification:
  - `uv run pytest tests/integration/test_synthetic_backend.py -q`
- Non-goals:
  - no real-dataset correctness
  - no GPU benchmarking

### PR M0.3: Extract A Testable Training Runner

- [x] `M0.3` Refactor the top-level entrypoint so tests can invoke the real training flow with injected dependencies.
- Problem:
  `main.py` is currently hardcoded around a specific remote model and dataset selection, which blocks realistic smoke tests.
- Proposed change:
  Extract a callable training runner from `main.py` that accepts:
  - config
  - model and tokenizer factories or injected instances
  - dataset factory or injected dataset
  - output directory
  Define an explicit offline test injection contract for the CLI wrapper, for example a `--mode synthetic` or equivalent config-file driven path that selects:
  - synthetic model backend
  - synthetic tokenizer backend
  - synthetic dataset backend
  - explicit output directory
  Keep `main.py` as a thin CLI wrapper around that runner.
- Likely files:
  `main.py`, new `train_runner.py`, related tests
- Acceptance criteria:
  - the core training flow is invocable without patching globals
  - `main.py` remains usable as a human entrypoint
  - the CLI wrapper has an explicit documented contract for offline synthetic execution
  - tests can run the real training flow with synthetic fixtures
- Required tests:
  - runner smoke test using synthetic fixtures
  - CLI wrapper smoke test that uses the explicit synthetic execution contract rather than monkeypatching globals
- Verification:
  - `uv run pytest tests/smoke/test_train_runner.py -q`
- Non-goals:
  - no checkpoint semantics expansion yet
  - no config-system redesign yet

### PR M0.4: Cut Over Supported Launch Scripts Early

- [x] `M0.4` Move supported launch scripts onto the tested training path as soon as the injectable runner exists.
- Problem:
  The repo currently has a split-brain launch path: `main.py` and `trainer.py` represent the evolving training system, but `run_job.sh` still launches `3B_train.py`.
- Proposed change:
  After `M0.3`, update supported launch scripts to invoke the new runner path. If `3B_train.py` is kept temporarily, mark it clearly as a non-supported experiment path and keep it out of the default launch flow.
- Likely files:
  `run_job.sh`, `main.py`, `3B_train.py`, docs, tests
- Acceptance criteria:
  - the supported batch-launch path points at the same entrypoint exercised by smoke tests
  - `3B_train.py` is either clearly deprecated or explicitly marked experimental
  - a smoke test protects the supported entrypoint
- Required tests:
  - supported entrypoint smoke test
- Verification:
  - `uv run pytest tests/smoke/test_train_runner.py -q`
- Non-goals:
  - no removal of historical experimental code that still has archival value

### PR M0.5: Document Required CI And Definition Of Done

- [x] `M0.5` Add contributor-facing testing and merge rules.
- Problem:
  Even with CI files present, the repo still lacks an explicit rule set for what every trainer PR must prove.
- Proposed change:
  Add a short contributor document covering:
  - required CI checks
  - bug-to-test policy
  - expectations for trainer, dataset, and checkpoint changes
  - how to run the fast suite locally
- Likely files:
  `README.md` or `docs/testing.md`
- Acceptance criteria:
  - the rules are short, concrete, and consistent with this roadmap
  - the local test command is documented
  - the policy clearly says that required CI gates merges
- Required tests:
  - none beyond doc linting if introduced
- Verification:
  - manual review of docs plus existing CI run
- Non-goals:
  - no broad contributor guide rewrite

## Milestone 1: Behavioral Safety Net In CI

### Goal

Turn the new test harness into a real behavioral barrier that catches trainer, dataset, artifact, and resume regressions before merge.

### Delivered Value

After this milestone, the repo should have a mandatory fast suite that exercises the real training path end to end on tiny local fixtures.

This milestone is intentionally not exhaustive. It should protect the public seams most likely to regress: trainer state flow, dataset adapter contract, checkpoint artifacts, and the supported entrypoint.

### Exit Criteria

- required CI includes unit, contract, integration, and tiny end-to-end smoke tests
- checkpoint metadata reload and continuation smoke are tested in CI
- dataset adapters have contract coverage without requiring remote data in required CI
- a broken training loop should be caught before merge

### PR M1.1: Add Unit Tests For Core Pure Logic

- [x] `M1.1` Add unit tests for memory, contradiction detection, config validation, and passage filtering.
- Problem:
  Core state transitions are currently unguarded and could regress silently.
- Proposed change:
  Add focused unit tests for:
  - `MemoryRegistry.write`, `get_active`, `get_at`, and `history`
  - `ContradictionDetector.check`
  - `TrainConfig.validate`
  - `PassageFilter`
- Likely files:
  `tests/unit/test_memory_registry.py`, `tests/unit/test_contradiction_detector.py`, `tests/unit/test_train_config.py`, `tests/unit/test_passage_filter.py`
- Acceptance criteria:
  - tests cover nominal cases and at least one edge case per component
  - tests do not require network, temp models, or real datasets
- Required tests:
  - all of the above
- Verification:
  - `uv run pytest tests/unit -q`
- Non-goals:
  - no trainer loop behavior yet

### PR M1.2: Add Dataset Contract Tests

- [x] `M1.2` Add contract tests for the `TemporalDataset` interface and adapter behavior.
- Problem:
  Dataset adapters are central to the repo but currently have no stable test contract.
- Proposed change:
  Define adapter contract tests that assert, for each dataset class:
  - `load()` accepts the expected split names
  - `get_probes()` returns `Probe` objects with required fields populated
  - `get_train_passages()` behaves correctly for train-capable splits
  - contradiction-pair methods return the documented shape
  Use monkeypatching or local fake backing data so required CI stays offline, including tiny temporary zip fixtures for `TemporalWikiDataset`.
- Likely files:
  `tests/contracts/test_dataset_contracts.py`, adapter-specific fixtures
- Acceptance criteria:
  - the test contract is explicit and reusable across dataset classes
  - required CI does not fetch remote datasets
  - adapters fail loudly when they violate the shared contract
- Required tests:
  - one contract test per dataset implementation
  - one negative-path test for invalid split handling
- Verification:
  - `uv run pytest tests/contracts/test_dataset_contracts.py -q`
- Non-goals:
  - no validation of real dataset contents
  - no dataset-download integration in required CI

### PR M1.3: Add Trainer Integration Tests

- [x] `M1.3` Add CPU integration tests for the custom trainer on synthetic fixtures.
- Problem:
  Unit tests alone will not catch broken training wiring, step accounting, or artifact generation.
- Proposed change:
  Add integration tests that run a tiny training period and assert:
  - loss is produced
  - checkpoints are written
  - the memory registry is updated
  - expected artifact files exist
  - trainer outputs have a stable schema
- Likely files:
  `tests/integration/test_trainer_loop.py`, `tests/contracts/test_checkpoint_artifacts.py`
- Acceptance criteria:
  - tests execute the real `CASFTrainer` on CPU
  - tests use the extracted runner or trainer directly, not mocks of the loop
  - artifact assertions are explicit and versionable
- Required tests:
  - tiny single-period train test
  - checkpoint artifact shape test
  - memory registry update test
- Verification:
  - `uv run pytest tests/integration/test_trainer_loop.py tests/contracts/test_checkpoint_artifacts.py -q`
- Non-goals:
  - no resume yet beyond artifact presence
  - no performance assertions yet

### PR M1.4: Add Checkpoint Metadata Resume Smoke Tests

- [x] `M1.4` Add required smoke tests proving that checkpoint metadata can be restored on tiny local fixtures.
- Problem:
  Checkpoint files may exist while resume semantics are still broken.
- Proposed change:
  Add tests that:
  - train for a short run
  - checkpoint
  - construct a new runner or trainer
  - load the checkpoint metadata supported by current code
  - continue execution without crashing
  The test should assert exactly what "resume" means at this stage and fail if behavior changes. At this milestone, resume only claims metadata reload from fully written checkpoints plus non-crashing continuation on a fresh trainer; interrupted-write safety and full state restoration are explicitly deferred to Milestone 2.
- Likely files:
  `tests/smoke/test_resume.py`, supporting helpers
- Acceptance criteria:
  - metadata resume smoke runs in required CI
  - the test asserts restored registry and last-period metadata that the current implementation claims to support
  - the test documents current limitations instead of implying full recovery
- Required tests:
  - save then metadata-resume smoke test
  - missing-metadata negative-path test
- Verification:
  - `uv run pytest tests/smoke/test_resume.py -q`
- Non-goals:
  - no full optimizer and scheduler restore yet
  - no exact-same-batch continuation guarantee yet
  - no interrupted-write or corruption recovery yet

### PR M1.5: Make Behavioral CI Required

- [x] `M1.5` Promote the fast behavioral suite to the required merge gate.
- Problem:
  Tests do not protect the repo unless they are both stable and required.
- Proposed change:
  Update CI so the required fast suite includes:
  - unit tests
  - dataset contract tests
  - trainer integration tests
  - checkpoint metadata resume smoke tests
  Add a separate optional lane for slower jobs if needed.
- Likely files:
  `.github/workflows/ci.yml`, docs updates
- Acceptance criteria:
  - required checks correspond to the Milestone 1 suite
  - slower non-blocking jobs are clearly separated
  - CI output makes it obvious which failures are merge-blocking
- Required tests:
  - CI workflow validation if tooling is added
  - existing fast suite remains green
- Verification:
  - green CI run on a PR branch
- Non-goals:
  - no external dashboard integration
  - no GPU runners as a requirement

## Milestone 2: Trustworthy Checkpoint And Resume

### Goal

Upgrade checkpointing from "files exist" to a checkpoint-boundary resume contract that is explicit, tested, and cheap to trust after interruption or preemption.

### Delivered Value

Interrupted runs become cheap to recover from safe optimizer-step boundaries, and future experimentation can rely on a declared resume contract instead of restarting from scratch.

### Exit Criteria

- checkpoints are versioned and only become visible as complete after finalize
- single-writer semantics are explicit for supported environments
- step accounting and checkpoint cadence are defined in optimizer steps
- resume restores trainer state, RNG state, and run cursor at safe boundaries
- compatibility and integrity validation fail fast on unsafe resume attempts

### PR M2.1: Add Versioned Checkpoints With Atomic Finalization

- [x] `M2.1` Replace in-place checkpoint writes with versioned checkpoint directories and an atomic `latest.json` pointer.
- Problem:
  Current checkpoints write directly into a target directory, which makes partial or ambiguous checkpoint state too easy to confuse with a valid resume source.
- Proposed change:
  Write each checkpoint into a unique temp directory under the run root, finalize it into a unique versioned directory such as `checkpoints/ckpt-000123/`, then atomically replace `latest.json` to point at the new checkpoint. Restrict full Milestone 2 support to environments where an advisory OS file lock on the run-root lock file is available and reliable. Clean stale temp directories during startup and checkpoint scans instead of trying to hide them.
- Likely files:
  `trainer.py`, `train_runner.py`, new checkpoint helpers, tests
- Acceptance criteria:
  - finalized checkpoint directories are versioned and never overwritten in place
  - `latest.json` always points at a fully written checkpoint
  - concurrent writers fail clearly when the run-root lock is already held
  - unsupported locking environments fail explicitly instead of claiming safe resume
  - stale temp directories are cleaned or reported deterministically
- Required tests:
  - checkpoint finalize test
  - advisory-lock behavior test
  - stale-temp cleanup test
- Verification:
  - targeted integration and smoke tests
- Non-goals:
  - no distributed or multi-writer checkpointing
  - no fallback locking scheme that cannot guarantee single-writer safety

### PR M2.2: Fix Step Accounting And Add Safe Checkpoint Cadence

- [x] `M2.2` Separate micro-steps from optimizer-steps, flush tail accumulation correctly, and checkpoint only at safe optimizer-step boundaries.
- Problem:
  The current loop can drop the final partial accumulation window, blur micro-step and optimizer-step semantics, and only save progress at coarse boundaries.
- Proposed change:
  Define and implement correct step semantics for:
  - micro-step versus optimizer-step counters
  - final incomplete accumulation windows
  - logged and persisted step values
  - configurable checkpoint cadence in optimizer steps
  Only emit checkpoints after completed optimizer steps with gradients cleared, while still checkpointing at training-unit end.
- Likely files:
  `trainer.py`, tests
- Acceptance criteria:
  - no gradients are silently discarded at the end of an epoch
  - reported progress matches actual optimizer updates
  - configured checkpoint cadence fires only on safe optimizer-step boundaries
  - tests cover divisible and non-divisible accumulation cases
- Required tests:
  - accumulation edge-case tests
  - checkpoint-cadence tests
  - trainer summary correctness tests
- Verification:
  - targeted trainer integration tests
- Non-goals:
  - no throughput tuning yet
  - no mid-accumulation checkpoint support

### PR M2.3: Restore Trainer State And Resume Within A Unit

- [ ] `M2.3` Persist trainer state, RNG state, and a current-unit cursor so resume can continue from the next safe checkpoint boundary.
- Problem:
  Restoring only model weights and metadata is not enough to continue a partially completed unit faithfully after preemption.
- Proposed change:
  Persist model, optimizer, scheduler, RNG state, current unit, completed-unit cursor, next safe optimizer-step cursor within the current unit, and step counters. At unit start, materialize a runner-owned snapshot of the filtered ordered training inputs for that unit. Resume should continue from the next safe optimizer-step boundary using that persisted unit snapshot instead of re-deriving batch order from loader randomness.
- Likely files:
  `trainer.py`, `train_runner.py`, checkpoint helpers, tests
- Acceptance criteria:
  - resume semantics are explicit and tested at checkpoint boundaries
  - the runner can continue within the current unit or skip to the next unfinished unit
  - RNG-driven behavior is restored under supported conditions
  - data-position behavior is driven by the persisted unit snapshot rather than incidental loader state
- Required tests:
  - within-unit resume test
  - next-unit continuation test
  - synthetic split-run versus uninterrupted equivalence test at a checkpoint boundary
- Verification:
  - targeted resume tests
- Non-goals:
  - no data-loader performance tuning
  - no promise of distributed resume semantics

### PR M2.4: Add Resume Compatibility Validation And Integrity Checks

- [ ] `M2.4` Add an explicit `resume_compatibility` contract, checkpoint integrity validation, and clear failure modes.
- Problem:
  Resume can still be unsafe if the invocation changes training semantics or if the checkpoint does not prove it came from compatible data and artifacts.
- Proposed change:
  Add a checkpoint manifest with:
  - schema version
  - model id or path
  - ordered training plan
  - a declared `resume_compatibility` block covering the training-semantic settings that would make resume unsafe if changed, including batch size, gradient accumulation, learning-rate and scheduler settings, max length, passage-filter settings, checkpoint cadence, and any other fields the trainer actually branches on
  - source-specific dataset identity for the currently supported datasets
  For TemporalWiki, use content digests of the source archives or files. For TSQA and TGQA, use dataset fingerprint or revision from the loaded Hugging Face split. Full Milestone 2 resume is supported only for datasets with a defined identity adapter. Keep M1 checkpoints readable, but explicitly treat them as metadata-only rather than full trainer-state checkpoints.
- Likely files:
  new checkpoint manifest helper, `trainer.py`, tests
- Acceptance criteria:
  - corrupt or incomplete checkpoints fail with a clear error
  - mismatched resume-compatibility settings fail before training continues
  - dataset identity mismatches fail for the supported datasets
  - M1 checkpoints are still readable but cannot claim full trainer-state recovery
- Required tests:
  - manifest validation tests
  - config-mismatch negative-path tests
  - dataset-identity mismatch negative-path tests
  - corruption and missing-file negative-path tests
- Verification:
  - targeted contract tests
- Non-goals:
  - no full provenance system
  - no automatic migration framework unless a real compatibility consumer appears

## Milestone 3: Reproducible And Informative Experiments

### Goal

Make each run easy to compare, inspect, and reproduce without adding heavy external tooling too early.

### Delivered Value

After this milestone, a run directory should tell you what happened, why it happened, and how to compare it to another run.

### Exit Criteria

- run artifacts have a stable, additive layout
- metrics are logged in a machine-readable way
- key experiment metadata is persisted
- lightweight evaluation hooks run at useful boundaries

### PR M3.1: Add An Additive Run-Root Layout And Stable Manifest

- [ ] `M3.1` Standardize the run root around stable top-level artifacts without breaking Milestone 2 checkpoint paths.
- Problem:
  Ad hoc run layouts make comparison and automation brittle.
- Proposed change:
  Keep the Milestone 2 checkpoint layout valid and add durable top-level artifacts around it:
  - `run_manifest.json`
  - `metrics/`
  - `periods/<unit>/`
  Readers should keep supporting the Milestone 2-only layout for at least one milestone transition so active runs remain resumable and inspectable while the new structure lands.
- Likely files:
  `trainer.py`, `train_runner.py`, new run-artifact helpers, tests
- Acceptance criteria:
  - every run root has a manifest
  - artifact paths are predictable
  - old Milestone 2 checkpoint paths remain valid
  - tests assert the directory schema and compatibility window
- Required tests:
  - manifest and layout contract tests
- Verification:
  - contract and integration tests
- Non-goals:
  - no external tracking service yet

### PR M3.2: Add Structured Metrics Logging

- [ ] `M3.2` Emit step and period metrics as JSONL or similarly simple structured logs.
- Problem:
  Printed logs are hard to compare and automate against.
- Proposed change:
  Write a minimal structured metrics contract under `metrics/` with:
  - `train_step` events
  - `period_end` events
  - `checkpoint` events
  The initial schema should stay intentionally small and machine-readable without requiring an external service.
- Likely files:
  `trainer.py`, logging helpers, tests
- Acceptance criteria:
  - metrics schema is versioned or explicitly contracted
  - logs remain human-readable enough to inspect locally
  - no external service dependency is introduced
- Required tests:
  - metrics schema contract test
  - log emission integration test
- Verification:
  - targeted tests plus local sample run
- Non-goals:
  - no W&B or MLflow requirement

### PR M3.3: Persist Reproducibility Metadata

- [ ] `M3.3` Record seeds and reproducibility metadata in every run manifest.
- Problem:
  Experiment comparisons are not trustworthy unless the causal inputs are recorded.
- Proposed change:
  Add `TrainConfig.seed` and persist the metadata needed to explain and compare a run, including:
  - seed
  - git commit and dirty flag
  - python, torch, and transformers versions
  - dataset selection
  - model id
  - ordered training plan
  - checkpoint schema version
- Likely files:
  run-manifest helpers, `trainer.py`, `main.py`, `train_config.py`, tests
- Acceptance criteria:
  - metadata is captured automatically
  - tests fail if required metadata fields disappear
- Required tests:
  - manifest-field presence tests
- Verification:
  - targeted contract tests
- Non-goals:
  - no environment snapshot overreach beyond what is useful

### PR M3.4: Add Lightweight Evaluation Hooks

- [ ] `M3.4` Run project-relevant evaluation at useful boundaries and persist results.
- Problem:
  Training loss alone is not enough to guide continual-learning decisions.
- Proposed change:
  Define the evaluation split contract per dataset family, then add evaluation hooks at useful boundaries with results written into structured artifacts. Use a thin generation adapter around model plus tokenizer so the existing evaluator code can be reused without changing the Hugging Face model class. For TemporalWiki, the contract must explicitly evaluate `changed` and `unchanged` separately and persist both outputs rather than relying on the dataset's last-loaded split. TSQA and TGQA should use `val` where available.
- Likely files:
  `trainer.py`, `main.py`, evaluator wiring, adapter helpers, tests
- Acceptance criteria:
  - evaluation cadence is configurable
  - split selection is explicit for each dataset family
  - evaluation outputs are saved in a stable format
  - tiny offline tests can exercise the hook path
- Required tests:
  - evaluation hook smoke test
  - split-selection tests proving `changed` and `unchanged` are both evaluated for TemporalWiki
  - artifact contract test for eval outputs
- Verification:
  - integration tests on synthetic fixtures
- Non-goals:
  - no full benchmark suite yet

## Milestone 4: Measurement-Led Performance Work

### Goal

Do only the performance work that is justified by measurement after the earlier milestones make the trainer trustworthy and observable.

### Delivered Value

Performance work becomes evidence-based instead of anecdotal, and the repo avoids starting a speculative optimization program before a measured bottleneck exists.

### Exit Criteria

- the repo can measure where step time goes
- any optimization work is justified by a measured bottleneck and benchmarked against a baseline
- performance-oriented changes are tested and guarded against correctness regressions

### PR M4.1: Add Timing And Throughput Instrumentation

- [ ] `M4.1` Measure step time, data time, checkpoint time, and effective throughput.
- Problem:
  Performance work is guesswork without measurement.
- Proposed change:
  Extend structured metrics with timing fields for data loading, forward/backward, optimizer step, checkpoint time, and throughput. Keep the instrumentation lightweight enough to leave enabled for ordinary runs.
- Likely files:
  `trainer.py`, metrics helpers, tests
- Acceptance criteria:
  - timings appear in structured logs
  - instrumentation overhead is small for normal runs
  - tests cover the presence and shape of timing fields
- Required tests:
  - timing-field contract tests
- Verification:
  - local sample run plus tests
- Non-goals:
  - no optimization yet

### PR M4.2: Address The Dominant Measured Bottleneck

- [ ] `M4.2` Implement one measured optimization against the largest observed bottleneck.
- Problem:
  Optimization work should be driven by evidence, not by generic trainer folklore.
- Proposed change:
  Use the instrumentation from `M4.1` to identify the dominant bottleneck in real runs, then implement exactly one focused optimization. Candidate areas include data preparation, collation, loader settings, or checkpoint cadence, but the chosen change must match the measured hotspot and keep the behavior contracts from Milestones 2 and 3 intact.
- Likely files:
  `trainer.py`, config, helpers, tests
- Acceptance criteria:
  - the PR names the measured bottleneck explicitly
  - before/after timing evidence is captured
  - correctness tests stay green
  - any changed behavioral contract is documented and tested
- Required tests:
  - integration tests for unchanged trainer outputs
  - targeted tests for the chosen optimization surface
- Verification:
  - benchmark note in PR plus test run
- Non-goals:
  - no speculative bundle of multiple optimizations
  - no distributed data loading

## Milestone 5: Continual-Learning Training Quality

### Goal

Improve the trainer so it better matches the actual continual-learning research question instead of being only a generic LM fine-tuning loop.

### Delivered Value

The training loop becomes more faithful to period-based factual updating and more useful for research conclusions.

### Exit Criteria

- multi-unit training is first-class rather than hidden behind a one-period constant
- resumed runs can skip completed units deterministically
- the supported training path matches the path exercised by CI

### PR M5.1: Add A Runner-Owned Training Plan And Multi-Unit Orchestration

- [ ] `M5.1` Replace the hidden one-period run with a runner-owned training plan and tested multi-unit orchestration.
- Problem:
  The current path is hardcoded to `["aug_sep"]`, which does not represent continual learning.
- Proposed change:
  Introduce a runner-owned `TrainingPlan` object and keep the ordered unit sequence in the training or orchestration layer rather than in dataset modules. For TemporalWiki, declare the default sequence in the training-plan layer as:
  - `aug_sep`
  - `sep_oct`
  - `oct_nov`
  - `nov_dec`
  Record the chosen ordered plan in manifests and checkpoints so resumed runs can skip completed units deterministically. Keep dataset and memory internals unchanged unless a concrete dependency is proven during implementation.
- Likely files:
  `main.py`, `train_runner.py`, `trainer.py`, tests
- Acceptance criteria:
  - the supported training sequence is declared in one training-owned place
  - orchestration uses the training plan rather than a hidden constant
  - completed units are recorded and skipped correctly on resume
  - outputs clearly separate per-unit results
- Required tests:
  - multi-period synthetic training test
  - period-order contract test
- Verification:
  - integration tests
- Non-goals:
  - no support for every imaginable curriculum schedule
  - no refactor of dataset or memory modules unless a concrete dependency is proven

### PR M5.2: Add Project-Relevant Continual-Learning Reports Only If The Baseline Review Cycle Needs Them

- [ ] `M5.2` Emit summary reports only if structured logs and eval artifacts are insufficient for comparing baseline continual-learning runs.
- Problem:
  If raw structured metrics are too cumbersome to compare, the repo will need a stable higher-level summary artifact.
- Proposed change:
  Add one narrow summary artifact that combines existing metrics and evaluation outputs into a stable comparison-friendly format. Skip this PR if the baseline decision workflow is already served by the Milestone 3 artifacts.
- Likely files:
  report helpers, evaluator wiring, tests
- Acceptance criteria:
  - the need for a summary artifact is stated concretely in the PR description
  - reports are generated automatically if this PR is implemented
  - report schema is stable and tested
  - reports are comparable across runs
- Required tests:
  - report artifact contract tests
  - integration smoke test
- Verification:
  - local sample run plus tests
- Non-goals:
  - no polished dashboard UI

### PR M5.3: Retire Redundant Entrypoints Once The Supported Path Is Stable

- [ ] `M5.3` Remove or isolate redundant entrypoints once the supported path has stayed stable through CI-backed use.
- Problem:
  Even after early launch-path cutover, old entrypoints can continue to drift unless the repo narrows the supported surface.
- Proposed change:
  Move `3B_train.py` under `experiments/legacy/` or otherwise isolate it outside the supported surface once the `main.py` plus `run_job.sh` path has stayed stable and tested.
- Likely files:
  `main.py`, `3B_train.py`, `run_job.sh`, docs, tests
- Acceptance criteria:
  - the repo has one clearly supported path
  - remaining experimental scripts are clearly outside the supported surface
  - CI smoke tests cover the supported path
- Required tests:
  - supported entrypoint smoke test
- Verification:
  - CI smoke test plus local dry run
- Non-goals:
  - no rewrite of historical notebooks or exploratory artifacts

## Deferred Until A Real Need Emerges

The following should not be pulled into early milestones unless a concrete need appears:

- distributed training
- mixed-precision expansion beyond what is needed for the active hardware path
- external experiment tracking platforms
- hyperparameter sweep orchestration frameworks
- async checkpoint systems
- automatic checkpoint migration frameworks
- large-scale benchmark dashboards
- contradiction-aware training policies until the baseline path and evaluation loop are stable and a concrete hypothesis exists

## Definition Of Done For Any Trainer PR

A trainer-affecting PR is only done if:
- behavior changes are described concretely
- required tests are added or updated
- fast required CI passes
- any new artifact schema is tested
- checkpoint compatibility impact is stated
- non-goals are explicit

## Expected GitHub Flow

The intended flow, once the early roadmap items land, is:

1. Create a branch for one roadmap box.
2. Implement the box and its required tests.
3. Open a PR that links the roadmap box and fills the PR template completely.
4. Use `gh pr checks` to verify required CI status.
5. Fix any failing required checks before asking for merge.
6. Merge with `gh pr merge` only after the pre-merge expectations in this document are satisfied.

## First PR Order

The default execution order is:

1. `M0.1`
2. `M0.2`
3. `M0.3`
4. `M0.4`
5. `M0.5`
6. `M1.1`
7. `M1.2`
8. `M1.3`
9. `M1.4`
10. `M1.5`
11. Continue in milestone order unless a dependency forces a roadmap update.
