# Step 1: Full fine-tune on TemporalWiki Period 1 (aug_sep) only.
# This teaches the 1B model the facts so we have a meaningful baseline.
# After this, run SMF/CASM/Full-FT on periods 2-4 starting from this checkpoint.

from training.train_config import TrainConfig
from training.train_runner import (
    build_real_dataset,
    build_real_model_and_tokenizer,
    load_real_model_and_tokenizer,
    run_training,
)

cfg = TrainConfig(
    run_id="pretrain_period1_1b",
    model_name="/scratch1/ramyakri/models/Llama-3.2-1B-Instruct",
    method="full_ft",
    dataset_name="temporal_wiki",
    precision="bfloat16",
    batch_size=4,
    grad_accum_steps=4,
    learning_rate=3e-4,
    epochs_per_period=3,
    warmup_steps=5,
    max_passages_per_period=200,
    log_every_n_steps=10,
    eval_after_each_period=True,
    seed=42,
)

cfg.validate()
print(f"Method: {cfg.method}")
print(f"Model: {cfg.model_name}")
print(f"Training on Period 1 (aug_sep) ONLY — 1B model")
print(f"Epochs: {cfg.epochs_per_period}")

# Only train on aug_sep — the first period
run_training(
    cfg,
    model_factory=build_real_model_and_tokenizer,
    resume_model_factory=load_real_model_and_tokenizer,
    dataset_factory=build_real_dataset,
    checkpoint_dir="/scratch1/ramyakri/checkpoints",
    training_units=["aug_sep"],
)
