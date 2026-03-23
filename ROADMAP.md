# Training Loop Roadmap

This roadmap is for turning the current repo into a training system that is trustworthy, testable in CI, and useful for repeated continual-learning experiments.

The immediate priority is not "add more trainer features." The immediate priority is to make behavioral regressions hard to ship and to make future PRs independently verifiable by tests.

## How To Use This Roadmap

- Another single-threaded agent should pick the next unticked PR in order.
- Each PR is expected to be small enough to review and test in isolation.
- A PR is not complete unless its acceptance criteria and required tests are satisfied.
- No PR should weaken CI or delete tests without replacing them with stronger coverage.
- If a PR discovers that this roadmap is missing a prerequisite, update this document in the same PR before implementation.

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
- checkpoint and resume smoke test

Allowed optional checks:
- longer dataset adapter checks against real data
- performance tracking jobs
- GPU smoke jobs

Optional checks may inform work, but required checks gate merges.

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
- Examples: tiny training loop on CPU, checkpoint save and resume, period transition updating memory registry, evaluation against a saved checkpoint.

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

- [ ] Milestone 0: Testable Project Skeleton
- [ ] Milestone 1: Behavioral Safety Net In CI
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

- [ ] `M0.1` Add `pytest`-based test harness, test layout, and required CI workflow.
- Problem:
  The repo currently has no automated regression barrier.
- Proposed change:
  Add `tests/`, `pytest` configuration, shared test utilities, and a GitHub Actions workflow that runs the fast suite on every PR.
- Likely files:
  `pyproject.toml`, `.github/workflows/ci.yml`, `tests/conftest.py`, `tests/test_sanity.py`
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

- [ ] `M0.2` Introduce synthetic dataset, tokenizer, and tiny causal LM fixtures for CPU-only tests.
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
  - `uv run pytest tests/helpers -q`
- Non-goals:
  - no real-dataset correctness
  - no GPU benchmarking

### PR M0.3: Extract A Testable Training Runner

- [ ] `M0.3` Refactor the top-level entrypoint so tests can invoke the real training flow with injected dependencies.
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
  - `uv run pytest tests/test_train_runner.py -q`
- Non-goals:
  - no checkpoint semantics expansion yet
  - no config-system redesign yet

### PR M0.4: Cut Over Supported Launch Scripts Early

- [ ] `M0.4` Move supported launch scripts onto the tested training path as soon as the injectable runner exists.
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
  - `uv run pytest tests/test_train_runner.py -q`
- Non-goals:
  - no removal of historical experimental code that still has archival value

### PR M0.5: Document Required CI And Definition Of Done

- [ ] `M0.5` Add contributor-facing testing and merge rules.
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
- checkpoint save and resume are tested in CI
- dataset adapters have contract coverage without requiring remote data in required CI
- a broken training loop should be caught before merge

### PR M1.1: Add Unit Tests For Core Pure Logic

- [ ] `M1.1` Add unit tests for memory, contradiction detection, config validation, and passage filtering.
- Problem:
  Core state transitions are currently unguarded and could regress silently.
- Proposed change:
  Add focused unit tests for:
  - `MemoryRegistry.write`, `get_active`, `get_at`, and `history`
  - `ContradictionDetector.check`
  - `TrainConfig.validate`
  - `PassageFilter`
- Likely files:
  `tests/test_memory.py`, `tests/test_contradiction.py`, `tests/test_config.py`, `tests/test_passage_filter.py`
- Acceptance criteria:
  - tests cover nominal cases and at least one edge case per component
  - tests do not require network, temp models, or real datasets
- Required tests:
  - all of the above
- Verification:
  - `uv run pytest tests/test_memory.py tests/test_contradiction.py tests/test_config.py tests/test_passage_filter.py -q`
- Non-goals:
  - no trainer loop behavior yet

### PR M1.2: Add Dataset Contract Tests

- [ ] `M1.2` Add contract tests for the `TemporalDataset` interface and adapter behavior.
- Problem:
  Dataset adapters are central to the repo but currently have no stable test contract.
- Proposed change:
  Define adapter contract tests that assert, for each dataset class:
  - `load()` accepts the expected split names
  - `get_probes()` returns `Probe` objects with required fields populated
  - `get_train_passages()` behaves correctly for train-capable splits
  - contradiction-pair methods return the documented shape
  Use monkeypatching or local fake backing data so required CI stays offline.
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

- [ ] `M1.3` Add CPU integration tests for the custom trainer on synthetic fixtures.
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

### PR M1.4: Add Checkpoint And Resume Smoke Tests

- [ ] `M1.4` Add required smoke tests proving that checkpointed runs can resume on tiny local fixtures.
- Problem:
  Checkpoint files may exist while resume semantics are still broken.
- Proposed change:
  Add tests that:
  - train for a short run
  - checkpoint
  - construct a new runner or trainer
  - load the checkpointed state supported by current code
  - continue execution without crashing
  The test should assert exactly what "resume" means at this stage and fail if behavior changes. At this milestone, resume only claims correctness for fully written checkpoints produced by a successful checkpoint call; interrupted-write safety is explicitly deferred to Milestone 2.
- Likely files:
  `tests/smoke/test_resume.py`, supporting helpers
- Acceptance criteria:
  - resume smoke runs in required CI
  - the test asserts restored artifacts and progress state that the current implementation claims to support
  - the test documents current limitations if resume is partial
- Required tests:
  - save then resume smoke test
  - missing-metadata negative-path test
- Verification:
  - `uv run pytest tests/smoke/test_resume.py -q`
- Non-goals:
  - no full optimizer and scheduler restore yet
  - no exact-same-batch continuation guarantee yet
  - no interrupted-write or corruption recovery yet

### PR M1.5: Make Behavioral CI Required

- [ ] `M1.5` Promote the fast behavioral suite to the required merge gate.
- Problem:
  Tests do not protect the repo unless they are both stable and required.
- Proposed change:
  Update CI so the required fast suite includes:
  - unit tests
  - dataset contract tests
  - trainer integration tests
  - checkpoint and resume smoke tests
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

Upgrade checkpointing from "files exist" to "resume is faithful enough to trust after interruption or preemption."

### Delivered Value

Interrupted runs become cheap to recover, and future experimentation can rely on resume semantics instead of restarting from scratch.

### Exit Criteria

- checkpoints save all state required by the declared resume contract
- checkpoint format is versioned
- resume tests verify state restoration, not just file presence
- partial training progress is not silently lost on resume

### PR M2.1: Save Full Trainer State With Atomic Checkpoint Writes

- [ ] `M2.1` Save optimizer, scheduler, global step, and epoch progress in checkpoints using an atomic write protocol.
- Problem:
  Current checkpoints only save model, tokenizer, memory registry, config, and last period.
- Proposed change:
  Extend checkpoint contents to save the state needed for faithful continuation of the current trainer design, and write checkpoints via a temp-path plus finalize step so partially written checkpoints are never mistaken for valid ones.
- Likely files:
  `trainer.py`, new checkpoint helpers, tests
- Acceptance criteria:
  - checkpoint manifest explicitly lists saved state
  - checkpoint directories are only made visible as complete once all required files are written
  - restore path reloads the saved trainer state
  - tests cover interrupted or incomplete checkpoint creation
- Required tests:
  - round-trip save and restore of trainer state
  - negative-path test for incomplete checkpoint writes
- Verification:
  - targeted integration tests
- Non-goals:
  - no distributed checkpointing

### PR M2.2: Fix Step Accounting And Tail-Step Semantics

- [ ] `M2.2` Make gradient accumulation, final partial steps, and reported progress correct.
- Problem:
  The current loop can drop the final partial accumulation window and may report progress loosely.
- Proposed change:
  Define and implement correct step semantics for:
  - optimizer step boundaries
  - final incomplete accumulation windows
  - logged global step values
  - returned training summaries
- Likely files:
  `trainer.py`, tests
- Acceptance criteria:
  - no gradients are silently discarded at the end of an epoch
  - reported progress matches actual optimizer updates
  - tests cover divisible and non-divisible accumulation cases
- Required tests:
  - accumulation edge-case tests
  - trainer summary correctness tests
- Verification:
  - targeted trainer integration tests
- Non-goals:
  - no throughput tuning yet

### PR M2.3: Restore RNG And Data-Position Semantics

- [ ] `M2.3` Make resume deterministic enough for the declared contract.
- Problem:
  Resuming training can change data order or RNG-driven behavior in uncontrolled ways.
- Proposed change:
  Save and restore:
  - RNG state where practical
  - sampler or data-position state for the current trainer contract
  If exact batch continuation is not supported, document the weaker guarantee precisely and test it.
- Likely files:
  `trainer.py`, checkpoint helpers, tests
- Acceptance criteria:
  - resume semantics are explicit and tested
  - the contract says whether continuation is exact or approximate
  - data-position behavior is no longer accidental
- Required tests:
  - deterministic resume test under supported conditions
  - contract test for documented weaker behavior if exact continuation is not implemented
- Verification:
  - targeted resume tests
- Non-goals:
  - no data-loader performance changes

### PR M2.4: Add Checkpoint Integrity Validation And Clear Failure Modes

- [ ] `M2.4` Add checkpoint manifest validation, integrity checks, and explicit failure behavior for corrupt artifacts.
- Problem:
  Checkpoint corruption or partial state can be mistaken for a valid resume source unless integrity is checked explicitly.
- Proposed change:
  Add a checkpoint manifest that validates the presence and integrity of required files on load. If the schema changes, version it at that point; do not build a migration system before there is a real compatibility consumer.
- Likely files:
  new checkpoint manifest helper, `trainer.py`, tests
- Acceptance criteria:
  - corrupt or incomplete checkpoints fail with a clear error
  - load-time validation is explicit rather than accidental
  - schema versioning is introduced only when the schema actually changes
- Required tests:
  - manifest validation tests
  - corruption and missing-file negative-path tests
- Verification:
  - targeted contract tests
- Non-goals:
  - no automatic migration framework unless a real need emerges
  - no broad backward-compatibility promise before it is needed

## Milestone 3: Reproducible And Informative Experiments

### Goal

Make each run easy to compare, inspect, and reproduce without adding heavy external tooling too early.

### Delivered Value

After this milestone, a run directory should tell you what happened, why it happened, and how to compare it to another run.

### Exit Criteria

- run artifacts have a stable layout
- metrics are logged in a machine-readable way
- key experiment metadata is persisted
- lightweight evaluation hooks run at useful boundaries

### PR M3.1: Add Stable Run Manifest And Directory Layout

- [ ] `M3.1` Standardize run directories and write a machine-readable manifest for every run.
- Problem:
  Ad hoc run layouts make comparison and automation brittle.
- Proposed change:
  Define a stable layout for configs, checkpoints, metrics, summaries, and environment metadata.
- Likely files:
  `trainer.py`, new run-artifact helpers, tests
- Acceptance criteria:
  - every run directory has a manifest
  - artifact paths are predictable
  - tests assert the directory schema
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
  Write structured metrics for:
  - step loss
  - optimizer step count
  - elapsed time
  - tokens or examples processed if available
  - checkpoint timings if available
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

- [ ] `M3.3` Record seeds, code revision, dataset selection, and model identifiers in every run.
- Problem:
  Experiment comparisons are not trustworthy unless the causal inputs are recorded.
- Proposed change:
  Add reproducibility metadata to the run manifest.
- Likely files:
  run-manifest helpers, `trainer.py`, `main.py`, tests
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
  Define the evaluation split contract per dataset family, then add evaluation hooks at useful boundaries with results written into structured artifacts. For TemporalWiki, the contract must explicitly evaluate `changed` and `unchanged` separately and persist both outputs rather than relying on the dataset's last-loaded split.
- Likely files:
  `trainer.py`, `main.py`, evaluator wiring, tests
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
  Add lightweight timing instrumentation around the training loop and checkpoint path.
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
  Use the instrumentation from `M4.1` to identify the dominant bottleneck in real runs, then implement exactly one focused optimization. Candidate areas include data preparation, collation, loader settings, or checkpoint cadence, but the chosen change must match the measured hotspot.
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

- TemporalWiki period sequencing is first-class rather than hardcoded
- evaluation reflects plasticity and stability goals
- the supported training path matches the path exercised by CI

### PR M5.1: Move Period Sequencing Into The Training-Owned Configuration Layer

- [ ] `M5.1` Replace the hardcoded one-period run with configuration-driven orchestration over the existing supported TemporalWiki period sequence.
- Problem:
  The current path is hardcoded to `["aug_sep"]`, which does not represent continual learning.
- Proposed change:
  Add explicit support for ordered period training and period-aware checkpoints, and declare the supported training sequence in one training-owned config or runner module rather than in a hidden constant inside `main.py`. Keep dataset and memory internals unchanged unless a concrete dependency is proven during implementation.
- Likely files:
  `main.py`, `train_runner.py`, `trainer.py`, `train_config.py`, tests
- Acceptance criteria:
  - the supported training sequence is declared in one training-owned place
  - training orchestration uses configured period order rather than a hidden constant in `main.py`
  - period order is explicit and tested
  - outputs clearly separate per-period results
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
  Fully retire or isolate redundant entrypoints such as `3B_train.py` once the supported path is stable and tested.
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
