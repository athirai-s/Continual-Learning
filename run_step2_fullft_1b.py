# Step 2a: Full Fine-Tuning 1B on Periods 1-4 (baseline — expect most forgetting)
# Loads base 1B model, then continues on ALL 4 periods with raw passages.
#
# Catastrophic forgetting story:
#   aug_sep  → model learns P1 facts
#   sep_oct  → model learns P2, starts to overwrite P1
#   oct_nov  → model learns P3, overwrites more P1
#   nov_dec  → model learns P4, P1 largely forgotten
#
# After training: evaluate on aug_sep probes → expect LOW retention (forgetting).

from training.train_config import TrainConfig
from training.train_runner import (
    build_real_dataset,
    build_real_model_and_tokenizer,
    load_real_model_and_tokenizer,
    run_training,
)

PRETRAINED_CHECKPOINT = "/scratch1/ashanmug/models/Llama-3.2-1B-Instruct"

cfg = TrainConfig(
    run_id="step2_fullft_1b",
    model_name=PRETRAINED_CHECKPOINT,
    method="full_ft",
    dataset_name="temporal_wiki",
    precision="bfloat16",
    batch_size=4,
    grad_accum_steps=4,           # effective batch = 16
    learning_rate=5e-4,           # aggressive LR → maximises weight overwriting
    epochs_per_period=5,          # 5 epochs per period → more overwriting of old facts
    warmup_steps=5,
    max_passages_per_period=400,  # more passages → stronger interference signal
    log_every_n_steps=10,
    eval_after_each_period=True,
    seed=42,
)

cfg.validate()
print("Step 2a: Full FT 1B on P1→P4  (catastrophic forgetting baseline)")
print(f"Loading from: {PRETRAINED_CHECKPOINT}")

run_training(
    cfg,
    model_factory=build_real_model_and_tokenizer,
    resume_model_factory=load_real_model_and_tokenizer,
    dataset_factory=build_real_dataset,
    checkpoint_dir="/scratch1/ashanmug/checkpoints",
    training_units=["aug_sep", "sep_oct", "oct_nov", "nov_dec"],
)