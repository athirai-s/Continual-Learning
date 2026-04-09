# Run LoRA fine-tuning experiment on Llama 3.2 3B with temporal_wiki
# Parameter-efficient baseline — only low-rank adapter weights updated

from training.train_config import TrainConfig
from training.train_runner import run_mode

cfg = TrainConfig(
    run_id="lora_3b_experiment",
    model_name="/scratch1/ashanmug/models/Llama-3.2-3B-Instruct",
    method="lora",
    dataset_name="temporal_wiki",
    precision="bfloat16",
    batch_size=1,
    grad_accum_steps=16,
    learning_rate=2e-4,
    epochs_per_period=3,
    max_passages_per_period=200,
    log_every_n_steps=10,
    eval_after_each_period=True,
    seed=42,

    # LoRA-specific settings
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    lora_target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
)

cfg.validate()
print(f"Method: {cfg.method}")
print(f"Model: {cfg.model_name}")
print(f"Precision: {cfg.precision}")
print(f"LoRA rank: {cfg.lora_r}")
print(f"LoRA alpha: {cfg.lora_alpha}")
print(f"LoRA dropout: {cfg.lora_dropout}")
print(f"LoRA target modules: {cfg.lora_target_modules}")

run_mode("real", cfg, checkpoint_dir="/scratch1/ashanmug/checkpoints")
