# Continual-Learning: SMF + CASM

## Current status
[UPDATE THIS at the end of every session]
Current phase: Phase 1 — TrainConfig and method-specific validation
Last completed: Step 1 — Added SMF + CASM fields, validation, from_dict, and tests (43/43 passing)
Next task: Step 2 — smf_model.py: frozen backbone + sparse memory module

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