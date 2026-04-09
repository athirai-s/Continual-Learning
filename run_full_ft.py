# Run full fine-tuning experiment on Llama 3.2 3B with temporal_wiki
# Baseline for catastrophic forgetting — all parameters updated each period

from training.train_config import TrainConfig
from training.train_runner import run_mode

cfg = TrainConfig(
    run_id="full_ft_3b_experiment",
    model_name="/scratch1/ashanmug/models/Llama-3.2-3B-Instruct",
    method="full_ft",
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
)

cfg.validate()
print(f"Method: {cfg.method}")
print(f"Model: {cfg.model_name}")
print(f"Precision: {cfg.precision}")
print(f"Batch size: {cfg.batch_size}")
print(f"Grad accum steps: {cfg.grad_accum_steps}")
print(f"Effective batch size: {cfg.batch_size * cfg.grad_accum_steps}")
print(f"Learning rate: {cfg.learning_rate}")
print(f"Epochs per period: {cfg.epochs_per_period}")
print(f"Max passages per period: {cfg.max_passages_per_period}")

run_mode("real", cfg, checkpoint_dir="/scratch1/ashanmug/checkpoints")
