# Step 2: Run all 4 methods (Full FT, LoRA, SMF, CASM) on Periods 2-4
# starting from the Period 1 pretrained checkpoint.
# Each method starts from the SAME checkpoint so comparison is fair.

from training.train_config import TrainConfig
from training.train_runner import (
    build_real_dataset,
    build_real_model_and_tokenizer,
    load_real_model_and_tokenizer,
    run_training,
)

PRETRAINED_CHECKPOINT = "/scratch1/ashanmug/checkpoints/pretrain_period1_3b/checkpoints/ckpt-000001"
CHECKPOINT_DIR = "/scratch1/ashanmug/checkpoints"
PERIODS_2_4 = ["sep_oct", "oct_nov", "nov_dec"]

# Shared settings
SHARED = dict(
    model_name=PRETRAINED_CHECKPOINT,
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


def run_full_ft():
    print("\n" + "=" * 60)
    print("  EXPERIMENT 1: Full Fine-Tuning on P2-P4")
    print("=" * 60)
    cfg = TrainConfig(
        run_id="step2_fullft_3b",
        method="full_ft",
        **SHARED,
    )
    cfg.validate()
    run_training(
        cfg,
        model_factory=build_real_model_and_tokenizer,
        resume_model_factory=load_real_model_and_tokenizer,
        dataset_factory=build_real_dataset,
        checkpoint_dir=CHECKPOINT_DIR,
        training_units=PERIODS_2_4,
    )


def run_lora():
    print("\n" + "=" * 60)
    print("  EXPERIMENT 2: LoRA on P2-P4")
    print("=" * 60)
    cfg = TrainConfig(
        run_id="step2_lora_3b",
        method="lora",
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        **SHARED,
    )
    cfg.validate()
    run_training(
        cfg,
        model_factory=build_real_model_and_tokenizer,
        resume_model_factory=load_real_model_and_tokenizer,
        dataset_factory=build_real_dataset,
        checkpoint_dir=CHECKPOINT_DIR,
        training_units=PERIODS_2_4,
    )


def run_smf():
    print("\n" + "=" * 60)
    print("  EXPERIMENT 3: SMF on P2-P4")
    print("=" * 60)
    cfg = TrainConfig(
        run_id="step2_smf_3b",
        method="smf",
        smf_memory_size=64,
        smf_sparsity_ratio=0.1,
        smf_update_layers=[8, 12, 16, 20, 24],
        smf_regularization_weight=0.01,
        smf_freeze_backbone=True,
        smf_learning_rate=1e-3,
        **SHARED,
    )
    cfg.validate()
    run_training(
        cfg,
        model_factory=build_real_model_and_tokenizer,
        resume_model_factory=load_real_model_and_tokenizer,
        dataset_factory=build_real_dataset,
        checkpoint_dir=CHECKPOINT_DIR,
        training_units=PERIODS_2_4,
    )


def run_casm():
    print("\n" + "=" * 60)
    print("  EXPERIMENT 4: CASM on P2-P4")
    print("=" * 60)
    cfg = TrainConfig(
        run_id="step2_casm_3b",
        method="casm",
        casm_num_slots=8,
        casm_router_hidden_size=256,
        casm_top_k=2,
        casm_router_temperature=1.0,
        casm_sparsity_weight=0.01,
        casm_overlap_weight=0.01,
        casm_branch_on_contradiction=True,
        casm_memory_size=64,
        **SHARED,
    )
    cfg.validate()
    run_training(
        cfg,
        model_factory=build_real_model_and_tokenizer,
        resume_model_factory=load_real_model_and_tokenizer,
        dataset_factory=build_real_dataset,
        checkpoint_dir=CHECKPOINT_DIR,
        training_units=PERIODS_2_4,
    )


if __name__ == "__main__":
    run_full_ft()
    run_lora()
    run_smf()
    run_casm()
    print("\n\nAll 4 experiments complete!")
