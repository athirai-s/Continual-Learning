# Step 1: Full fine-tune on TemporalWiki Period 1 (aug_sep) only.
#
# This is the FOUNDATION of the entire experiment.
# The model must memorise P1 (aug_sep) facts well here — otherwise
# there is nothing to forget and nothing to compare across methods.
#
# Target: high F1 on aug_sep changed + unchanged probes before step2 begins.
# Run step3_eval on this checkpoint to get the upper-bound retention score.

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
    grad_accum_steps=4,           # effective batch = 16
    learning_rate=2e-4,           # conservative — stable fact memorisation
    epochs_per_period=8,          # 8 epochs (was 3) — model must memorise P1 facts
    warmup_steps=10,
    max_passages_per_period=400,  # 400 passages (was 200) — more P1 coverage
    log_every_n_steps=10,
    eval_after_each_period=True,  # eval score here = upper bound for step3
    seed=42,
)

cfg.validate()
print(f"Method: {cfg.method}")
print(f"Model: {cfg.model_name}")
print(f"Training on Period 1 (aug_sep) ONLY — 1B model")
print(f"Epochs: {cfg.epochs_per_period}  Passages: {cfg.max_passages_per_period}")
print("Target: high F1 on aug_sep probes — this is the forgetting baseline")

run_training(
    cfg,
    model_factory=build_real_model_and_tokenizer,
    resume_model_factory=load_real_model_and_tokenizer,
    dataset_factory=build_real_dataset,
    checkpoint_dir="/scratch1/ramyakri/checkpoints",
    training_units=["aug_sep"],
)
