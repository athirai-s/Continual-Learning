"""Step 4: CASM on synthetic with the stronger v2 config.

Changes from the previous 3B synthetic run (train_carc_synthetic.py --method casm):
    casm_top_k                : 3   -> 1     (with per-period masking, 1 slot gets
                                              100% gradient per step instead of 37.5%)
    casm_router_temperature   : 0.3 -> 0.05  (sharper routing, fewer ties)
    casm_num_injection_layers : 1   -> 4     (spread memory influence across layers)

Held fixed (already at consensus values):
    casm_memory_size          : 512   (4x the 1B-notebook 128)
    casm_slots_per_period     : 8     (period-deterministic masking on)
    casm_num_slots            : 32    = 8 * 4 periods
    casm_router_type          : 'similarity'  (match the 1B notebook run so the
                                              config change is the only variable)

Dataset: synthetic (2018 -> 2020 -> 2022 -> 2024).  This is the contradiction-rich
benchmark where CASM's mechanism is actually activated (449/449 sequential
contradictions, per diagnostics/contradiction_structure.json), unlike
TemporalWiki (~5 contradictions across 4 periods).

Before launching, run diagnose_casm_checkpoint.py on an existing CASM
checkpoint.  If the dead-slot audit flags a CLUSTERED period, drop
CASM_SLOTS_PER_PERIOD from 8 to 4 or 6 here.
"""

from training.train_config import TrainConfig
from training.train_runner import (
    build_real_model_and_tokenizer,
    build_synthetic_dataset,
    load_real_model_and_tokenizer,
    run_training,
)

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
CHECKPOINT_DIR = "/scratch1/ashanmug/checkpoints"
PERIODS = ["2018", "2020", "2022", "2024"]

CASM_SLOTS_PER_PERIOD = 8
CASM_NUM_SLOTS = CASM_SLOTS_PER_PERIOD * len(PERIODS)

cfg = TrainConfig(
    run_id="step4_casm_synth_3b_v2",
    model_name=MODEL_NAME,
    method="casm",
    dataset_name="synthetic",
    precision="bfloat16",
    batch_size=4,
    grad_accum_steps=4,
    learning_rate=2e-3,
    epochs_per_period=5,
    warmup_steps=50,
    max_passages_per_period=400,
    min_passage_length=0,
    log_every_n_steps=10,
    eval_after_each_period=True,
    seed=42,
    # CASM — stronger v2 config
    casm_num_slots=CASM_NUM_SLOTS,
    casm_slots_per_period=CASM_SLOTS_PER_PERIOD,
    casm_memory_size=512,
    casm_num_injection_layers=4,
    casm_top_k=1,
    casm_router_temperature=0.05,
    casm_router_hidden_size=512,
    casm_router_type="similarity",
    casm_sparsity_weight=0.001,
    casm_overlap_weight=0.001,
    # Must stay False — branching adds slots dynamically, incompatible with
    # the period-deterministic slot map (casm_slots_per_period).
    casm_branch_on_contradiction=False,
)

cfg.validate()
print("Step 4: CASM synthetic v2 — stronger less-diffuse config")
print(f"Model: {MODEL_NAME}")
print(f"Periods: {PERIODS}")
print(
    f"CASM: num_slots={CASM_NUM_SLOTS} "
    f"slots_per_period={CASM_SLOTS_PER_PERIOD} "
    f"memory_size=512 injection_layers=4 top_k=1 temp=0.05"
)

run_training(
    cfg,
    model_factory=build_real_model_and_tokenizer,
    resume_model_factory=load_real_model_and_tokenizer,
    dataset_factory=build_synthetic_dataset,
    checkpoint_dir=CHECKPOINT_DIR,
    training_units=PERIODS,
)
