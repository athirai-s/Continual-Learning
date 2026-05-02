# Step 2c: SMF on Periods 1-4 (sparse memory — expect less forgetting than full_ft)
# Loads from Period 1 pretrained checkpoint, then trains on ALL 4 periods.
#
# Backbone is FROZEN — P1 knowledge cannot be overwritten.
# Sparse memory blocks accumulate knowledge across periods.
# Llama-3.2-1B has 16 layers (0-15), using mid-to-late layers 8-14.

from training.train_config import TrainConfig
from training.train_runner import (
    build_real_dataset,
    build_real_model_and_tokenizer,
    load_real_model_and_tokenizer,
    run_training,
)

PRETRAINED_CHECKPOINT = "/scratch1/ramyakri/checkpoints/pretrain_period1_1b/checkpoints/ckpt-000001"

cfg = TrainConfig(
    run_id="step2_smf_1b",
    model_name=PRETRAINED_CHECKPOINT,
    method="smf",
    dataset_name="temporal_wiki",
    precision="bfloat16",
    batch_size=4,
    grad_accum_steps=4,           # effective batch = 16
    learning_rate=5e-4,
    epochs_per_period=5,
    warmup_steps=5,
    max_passages_per_period=400,
    log_every_n_steps=10,
    eval_after_each_period=True,
    seed=42,
    smf_memory_size=64,
    smf_sparsity_ratio=0.1,
    smf_update_layers=[8, 10, 12, 14],
    smf_regularization_weight=0.01,
    smf_freeze_backbone=True,
    smf_learning_rate=1e-3,
)

cfg.validate()
print("Step 2c: SMF on P1→P4  (sparse memory, frozen backbone)")
print(f"Loading from: {PRETRAINED_CHECKPOINT}")

run_training(
    cfg,
    model_factory=build_real_model_and_tokenizer,
    resume_model_factory=load_real_model_and_tokenizer,
    dataset_factory=build_real_dataset,
    checkpoint_dir="/scratch1/ramyakri/checkpoints",
    training_units=["aug_sep", "sep_oct", "oct_nov", "nov_dec"],
)
