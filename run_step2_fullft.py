# Step 2a: Full Fine-Tuning on Periods 2-4 (baseline — expect most forgetting)
# Loads from Period 1 pretrained checkpoint.

from training.train_config import TrainConfig
from training.train_runner import (
    build_real_dataset,
    build_real_model_and_tokenizer,
    load_real_model_and_tokenizer,
    run_training,
)

PRETRAINED_CHECKPOINT = "/scratch1/ashanmug/checkpoints/pretrain_period1_3b/checkpoints/ckpt-000001"

cfg = TrainConfig(
    run_id="step2_fullft_3b",
    model_name=PRETRAINED_CHECKPOINT,
    method="full_ft",
    dataset_name="temporal_wiki",
    batch_size=1,
    grad_accum_steps=16,
    learning_rate=2e-4,
    epochs_per_period=3,
    warmup_steps=50,
    max_passages_per_period=200,
    log_every_n_steps=10,
    eval_after_each_period=True,
    seed=42,
)

cfg.validate()
print(f"Step 2a: Full FT on P2-P4")
print(f"Loading from: {PRETRAINED_CHECKPOINT}")

run_training(
    cfg,
    model_factory=build_real_model_and_tokenizer,
    resume_model_factory=load_real_model_and_tokenizer,
    dataset_factory=build_real_dataset,
    checkpoint_dir="/scratch1/ashanmug/checkpoints",
    training_units=["sep_oct", "oct_nov", "nov_dec"],
)
