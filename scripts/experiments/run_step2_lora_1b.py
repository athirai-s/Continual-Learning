# Step 2b: LoRA on Periods 1-4 (parameter-efficient baseline)
# Loads from Period 1 pretrained checkpoint, then trains on ALL 4 periods.
#
# LoRA keeps the backbone frozen (only adapter weights updated).
# Expected: moderate retention — backbone holds P1 knowledge,
# but adapters may shift toward later periods.

from training.train_config import TrainConfig
from training.train_runner import (
    build_real_dataset,
    build_real_model_and_tokenizer,
    load_real_model_and_tokenizer,
    run_training,
)

PRETRAINED_CHECKPOINT = "/scratch1/ramyakri/checkpoints/pretrain_period1_1b/checkpoints/ckpt-000001"

cfg = TrainConfig(
    run_id="step2_lora_1b",
    model_name=PRETRAINED_CHECKPOINT,
    method="lora",
    dataset_name="temporal_wiki",
    precision="bfloat16",
    batch_size=4,
    grad_accum_steps=4,           # effective batch = 16
    learning_rate=2e-4,           # standard LoRA lr — adapters are small, don't need aggressive lr
    epochs_per_period=5,
    warmup_steps=5,
    max_passages_per_period=400,
    log_every_n_steps=10,
    eval_after_each_period=True,
    seed=42,
    lora_r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

cfg.validate()
print("Step 2b: LoRA on P1→P4")
print(f"Loading from: {PRETRAINED_CHECKPOINT}")

run_training(
    cfg,
    model_factory=build_real_model_and_tokenizer,
    resume_model_factory=load_real_model_and_tokenizer,
    dataset_factory=build_real_dataset,
    checkpoint_dir="/scratch1/ramyakri/checkpoints",
    training_units=["aug_sep", "sep_oct", "oct_nov", "nov_dec"],
)
