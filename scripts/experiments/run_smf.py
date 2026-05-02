# Run SMF (Sparse Memory Finetuning) experiment on Llama 3.2 3B

from training.train_config import TrainConfig
from training.train_runner import run_mode

cfg = TrainConfig(
    run_id="smf_3b_experiment",
    model_name="/scratch1/ashanmug/models/Llama-3.2-3B-Instruct",
    method="smf",
    dataset_name="temporal_wiki",
    batch_size=1,
    grad_accum_steps=16,
    learning_rate=2e-4,
    epochs_per_period=1,       # reduced from 3 — memory overfits fast with frozen backbone
    warmup_steps=10,           # reduced from 100 — ~12 optimizer steps per period at these settings
    max_passages_per_period=200,
    log_every_n_steps=10,
    eval_after_each_period=True,
    seed=42,

    # SMF-specific settings
    smf_memory_size=64,
    smf_sparsity_ratio=0.1,
    smf_update_layers=[8, 12, 16, 20, 24],  # mid-to-late layers only; layer 0 encodes syntax not facts
    smf_regularization_weight=0.01,
    smf_freeze_backbone=True,
    smf_learning_rate=1e-3,    # higher than backbone LR — memory params train from scratch
)

cfg.validate()
print(f"Method: {cfg.method}")
print(f"Model: {cfg.model_name}")
print(f"SMF memory size: {cfg.smf_memory_size}")
print(f"SMF update layers: {cfg.smf_update_layers}")
print(f"SMF sparsity ratio: {cfg.smf_sparsity_ratio}")

run_mode("real", cfg, checkpoint_dir="/scratch1/ashanmug/checkpoints")