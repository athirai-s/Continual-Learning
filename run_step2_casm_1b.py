# Step 2d: CASM on Periods 1-4 (versioned memory + routing — expect least forgetting)
# Loads from Period 1 pretrained checkpoint, then trains on ALL 4 periods.
#
# CASM story across 4 periods:
#   aug_sep  → slot 0 learns P1 facts  (backbone frozen — P1 also in backbone weights)
#   sep_oct  → slot 1 learns P2 facts
#   oct_nov  → slot 2 learns P3 facts
#   nov_dec  → slot 3 learns P4 facts
#   (extra slots 4-5 absorb contradiction branches)
#
# Backbone FROZEN throughout → P1 backbone knowledge is NEVER overwritten.
# After training: evaluate on aug_sep probes → expect HIGH retention.

from training.train_config import TrainConfig
from training.train_runner import (
    build_real_dataset,
    build_real_model_and_tokenizer,
    load_real_model_and_tokenizer,
    run_training,
)

PRETRAINED_CHECKPOINT = "/scratch1/ramyakri/checkpoints/pretrain_period1_1b/checkpoints/ckpt-000001"

cfg = TrainConfig(
    run_id="step2_casm_1b_v7",
    model_name=PRETRAINED_CHECKPOINT,
    method="casm",
    dataset_name="temporal_wiki",
    precision="bfloat16",
    batch_size=4,
    grad_accum_steps=4,            # effective batch = 16
    learning_rate=2e-3,            # v4 proven lr
    epochs_per_period=5,
    warmup_steps=5,                # v4 proven warmup
    max_passages_per_period=400,   # matches full_ft for fair comparison
    log_every_n_steps=10,
    eval_after_each_period=True,
    seed=42,
    # --- slot bank (now query_dependent=True — per-token gating) ---
    casm_num_slots=6,              # 4 periods + 2 branch buffer
    casm_memory_size=512,          # large capacity for per-token gating
    # --- router ---
    casm_router_hidden_size=512,   # matches memory size
    casm_top_k=1,                  # clean gradient signal — one slot per query
    casm_router_temperature=0.3,   # sharp routing for specialization
    # --- losses ---
    casm_sparsity_weight=0.001,    # v4 proven
    casm_overlap_weight=0.001,     # v4 proven
    casm_branch_on_contradiction=True,
    # Single layer injection (v4 proven — multi-layer hurt performance)
)

cfg.validate()
print("Step 2d: CASM on P1→P4  (versioned memory, frozen backbone)")
print(f"Loading from: {PRETRAINED_CHECKPOINT}")

run_training(
    cfg,
    model_factory=build_real_model_and_tokenizer,
    resume_model_factory=load_real_model_and_tokenizer,
    dataset_factory=build_real_dataset,
    checkpoint_dir="/scratch1/ramyakri/checkpoints",
    training_units=["aug_sep", "sep_oct", "oct_nov", "nov_dec"],
)
