# Step 2b: LoRA on Periods 2-4 (parameter-efficient baseline)
# Loads from Period 1 pretrained checkpoint.

from training.train_config import TrainConfig
from training.train_runner import (
    build_real_dataset,
    build_real_model_and_tokenizer,
    load_real_model_and_tokenizer,
    run_training,
)

PRETRAINED_CHECKPOINT = "/content/drive/MyDrive/checkpoints/pretrain_period1_1b/checkpoints/ckpt-000001"

cfg = TrainConfig(
    run_id="step2_lora_1b",
    model_name=PRETRAINED_CHECKPOINT,
    method="lora",
    dataset_name="temporal_wiki",
    precision="bfloat16",
    batch_size=4,
    grad_accum_steps=4,         # effective batch = 16
    learning_rate=5e-4,         # LoRA trains far fewer params — needs higher lr than full FT
    epochs_per_period=3,
    warmup_steps=5,             # ~13% of 38 total optimizer steps
    max_passages_per_period=200,
    log_every_n_steps=10,
    eval_after_each_period=True,
    seed=42,
    lora_r=8,                   # 1B hidden dim is 2048 (vs 3072 for 3B) — r=8 is sufficient
    lora_alpha=16,              # 2x rank — standard convention
    lora_dropout=0.05,
    lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

cfg.validate()
print(f"Step 2b: LoRA on P2-P4")
print(f"Loading from: {PRETRAINED_CHECKPOINT}")

run_training(
    cfg,
    model_factory=build_real_model_and_tokenizer,
    resume_model_factory=load_real_model_and_tokenizer,
    dataset_factory=build_real_dataset,
    checkpoint_dir="/content/drive/MyDrive/checkpoints",
    training_units=["sep_oct", "oct_nov", "nov_dec"],
)
