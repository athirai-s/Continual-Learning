# Continual-Learning: SMF + CASM

## Current status
[UPDATE THIS at the end of every session]
Current phase: Phase 5 — CASM checkpointing
Last completed: Step 5 — Full CASM checkpoint/resume: save_pretrained now persists slot_usage_counts in casm_memory.pt; load_memory_into recreates extra slots (contradiction-branched), resizes router output layer before loading state dict, and restores slot_usage_counts with backward-compat fallback; _slot_usage_counts reset to zero after period-end registry sync (prevents double-counting on period-boundary resume); _model_slot_to_registry_slot_id persisted in trainer_state.pt and restored on resume; validate_checkpoint_method_compatibility added to artifacts/checkpointing.py and called in resume(); 27 new tests, 196 unit tests passing (1 pre-existing Windows lock failure unchanged)
Next task: Step 6 — evaluation + metrics: per-period reporting for CASM (plasticity, stability, contradiction_acc, routing_acc)

## Known pre-existing test failures (Windows — not caused by this implementation)
23 tests fail on Windows before any of our changes. Two root causes:

1. **Advisory file locks unsupported on Windows** — `RunRootLock` in `artifacts/checkpointing.py` raises `CheckpointLockUnsupportedError` on this platform. Directly breaks:
   - `tests/unit/test_checkpointing.py::test_run_root_lock_rejects_second_writer`
   - `tests/smoke/test_run_locking.py::test_run_training_fails_fast_when_run_root_is_already_locked`
   - All 6 `tests/smoke/test_resume.py` tests that exercise the full runner (runner acquires the lock on entry)

2. **End-to-end training loop depends on the file lock** — the runner fails before writing any artifacts, so every test that asserts on output files also fails:
   - `tests/smoke/test_train_runner.py` (3 tests)
   - `tests/smoke/test_eval_hooks.py` (1 test)
   - `tests/smoke/test_launchers.py` (1 test)
   - `tests/contracts/test_checkpoint_artifacts.py`, `test_eval_artifacts.py`, `test_metrics_schema.py`, `test_run_artifacts.py` (4 tests, 2 from run_artifacts)
   - `tests/contracts/test_run_manifest_metadata.py` (1 test)
   - `tests/integration/test_metrics_logging.py`, `test_reproducibility.py`, `test_training_plan_orchestration.py` (3 tests)

These failures are unrelated to SMF/CASM work and were confirmed present on the base commit before this branch.

## Architecture rules
- TrainConfig.method is the ONLY method switch
- Do NOT fork train_runner.py — one entrypoint only
- run_training() owns the outer loop
- trainer.train_period() does per-period learning
- trainer.checkpoint() saves all state
- CASM must save registry + router state in checkpoints

## Build order
0. Stabilize shared foundation
1. train_config.py — SMF + CASM fields + validation
2. smf_model.py — frozen backbone + sparse memory
3. trainer.py — optimizer branching + SMF train step
4. casf_dataset_api/memory.py — versioned slots
5. casm_model.py — slot bank + router
6. trainer.py — CASM train_period() branch
7. artifacts/checkpointing.py — persist registry + router
8. evaluation + metrics

## Key files
- training/train_config.py
- training/trainer.py
- training/train_runner.py
- training/training_plan.py
- casf_dataset_api/memory.py
- casf_dataset_api/contra*.py
- artifacts/checkpointing.py

## Reference docs
@docs/implementation_guide.md
@docs/architecture_overview.md

## Ignore these files
Ignore AGENTS.md and ROADMAP.md — they are not part of this implementation.