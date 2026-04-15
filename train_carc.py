"""
train_carc.py — CARC training script converted from train_colab.ipynb.

Usage:
    python train_carc.py --method full_ft
    python train_carc.py --method lora
    python train_carc.py --method smf
    python train_carc.py --method casm

Trains on all 4 periods (aug_sep, sep_oct, oct_nov, nov_dec) using
meta-llama/Llama-3.2-3B by default.  Checkpoints are saved to
/scratch1/ramyakri/checkpoints/<run_id>/.

Set --augmented to use augmented CSVs (default), or --no-augmented for
TWiki_Diffsets.zip passages.
"""

import argparse
import json
import os
import sys
import torch

# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Continual-Learning CARC trainer")
parser.add_argument(
    "--method",
    required=True,
    choices=["full_ft", "lora", "smf", "casm"],
    help="Training method",
)
parser.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct", help="HuggingFace model name or checkpoint path")
parser.add_argument("--run-id", default=None, help="Experiment name (default: auto-generated from method + model)")
parser.add_argument("--checkpoint-dir", default="/scratch1/ramyakri/checkpoints", help="Root checkpoint directory")
parser.add_argument("--periods", nargs="+", default=["aug_sep", "sep_oct", "oct_nov", "nov_dec"], help="Periods to train")
parser.add_argument("--augmented", action="store_true", default=True, help="Use augmented CSV passages (default)")
parser.add_argument("--no-augmented", action="store_true", help="Use TWiki_Diffsets.zip passages instead")
parser.add_argument("--dataset-fraction", type=float, default=None, help="Fraction of dataset to use (e.g. 0.1 for 10%%)")
parser.add_argument("--epochs", type=int, default=5, help="Epochs per period")
parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
parser.add_argument("--lr", type=float, default=None, help="Learning rate (default: method-dependent)")
parser.add_argument("--max-passages", type=int, default=400, help="Max passages per period (None for all)")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--capture-activations", action="store_true", default=False, help="Capture per-layer activations")
args = parser.parse_args()

USE_AUGMENTED_DATA = not args.no_augmented

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
METHOD = args.method
MODEL_NAME = args.model
PERIODS = args.periods
CHECKPOINT_DIR = args.checkpoint_dir
SEED = args.seed

# Auto-generate run_id from method + model size if not provided
if args.run_id is not None:
    RUN_ID = args.run_id
else:
    model_tag = "3b" if "3B" in MODEL_NAME or "3b" in MODEL_NAME else "1b"
    RUN_ID = f"step2_{METHOD}_{model_tag}"

# Default learning rates per method (match existing CARC scripts)
DEFAULT_LR = {
    "full_ft": 2e-5,
    "lora": 2e-4,
    "smf": 1e-3,
    "casm": 2e-3,
}
LEARNING_RATE = args.lr if args.lr is not None else DEFAULT_LR[METHOD]

BATCH_SIZE = args.batch_size
GRAD_ACCUM_STEPS = args.grad_accum
EPOCHS_PER_PERIOD = args.epochs
MAX_PASSAGES_PER_PERIOD = args.max_passages
DATASET_FRACTION = args.dataset_fraction
PRECISION = "bfloat16"
CAPTURE_ACTIVATIONS = args.capture_activations

# ---- LoRA settings ----
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]

# ---- SMF settings ----
# Llama-3.2-3B has 28 transformer layers (indices 0-27)
# Llama-3.2-1B has 16 transformer layers (indices 0-15)
if "3B" in MODEL_NAME or "3b" in MODEL_NAME:
    SMF_UPDATE_LAYERS = [7, 14, 21]
else:
    SMF_UPDATE_LAYERS = [4, 8, 12]
SMF_MEMORY_SIZE = 64
SMF_SPARSITY_RATIO = 0.1
SMF_REGULARIZATION_WEIGHT = 0.01
SMF_LEARNING_RATE = 1e-3

# ---- CASM settings ----
CASM_NUM_SLOTS = 6          # 4 periods + 2 branch buffer
CASM_MEMORY_SIZE = 512
CASM_ROUTER_HIDDEN_SIZE = 512
CASM_TOP_K = 1
CASM_ROUTER_TEMPERATURE = 0.3
CASM_SPARSITY_WEIGHT = 0.001
CASM_OVERLAP_WEIGHT = 0.001
CASM_BRANCH_ON_CONTRADICTION = True

# ---------------------------------------------------------------------------
# Print configuration
# ---------------------------------------------------------------------------
print("=" * 60)
print("CARC TRAINING — Continual Learning")
print("=" * 60)
print(f"Method:               {METHOD}")
print(f"Model:                {MODEL_NAME}")
print(f"Run ID:               {RUN_ID}")
print(f"Periods:              {PERIODS}")
print(f"Batch size:           {BATCH_SIZE}")
print(f"Grad accum steps:     {GRAD_ACCUM_STEPS}  (eff. batch = {BATCH_SIZE * GRAD_ACCUM_STEPS})")
print(f"Learning rate:        {LEARNING_RATE}")
print(f"Epochs per period:    {EPOCHS_PER_PERIOD}")
print(f"Max passages:         {MAX_PASSAGES_PER_PERIOD}")
print(f"Dataset fraction:     {DATASET_FRACTION if DATASET_FRACTION is not None else 'all (None)'}")
print(f"Augmented data:       {USE_AUGMENTED_DATA}")
print(f"Capture activations:  {CAPTURE_ACTIVATIONS}")
print(f"Checkpoint dir:       {CHECKPOINT_DIR}")
print(f"Seed:                 {SEED}")
print(f"CUDA available:       {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU:                  {torch.cuda.get_device_name(0)}")
print()

# ---------------------------------------------------------------------------
# Dataset verification
# ---------------------------------------------------------------------------
REPO_DIR = os.path.dirname(os.path.abspath(__file__))

if USE_AUGMENTED_DATA:
    bundled_probes = os.path.join(
        REPO_DIR, "casf_dataset_api", "download_dataset_scripts", "data", "TWiki_Probes.zip"
    )
    assert os.path.exists(bundled_probes), (
        f"TWiki_Probes.zip not found at {bundled_probes}. "
        "Make sure the repo was cloned correctly."
    )
    print(f"TWiki_Probes.zip: OK ({os.path.getsize(bundled_probes):,} bytes)")

    aug_dir = os.path.join(REPO_DIR, "data", "augmented", "TWiki_Diffsets")
    for period in PERIODS:
        csv_path = os.path.join(aug_dir, f"{period}.csv")
        assert os.path.exists(csv_path), f"Augmented CSV missing: {csv_path}"
        print(f"  {period}.csv: OK ({os.path.getsize(csv_path):,} bytes)")

    print("\nAll dataset files present — no download needed.")
else:
    diffsets_zip = os.path.join(REPO_DIR, "data", "TWiki_Diffsets.zip")
    probes_zip = os.path.join(REPO_DIR, "data", "TWiki_Probes.zip")
    assert os.path.exists(diffsets_zip), f"TWiki_Diffsets.zip not found at {diffsets_zip}"
    assert os.path.exists(probes_zip), f"TWiki_Probes.zip not found at {probes_zip}"
    print(f"TWiki_Diffsets.zip: OK ({os.path.getsize(diffsets_zip):,} bytes)")
    print(f"TWiki_Probes.zip:   OK ({os.path.getsize(probes_zip):,} bytes)")

print()

# ---------------------------------------------------------------------------
# Build TrainConfig
# ---------------------------------------------------------------------------
from training.train_config import TrainConfig

config_kwargs = dict(
    run_id=RUN_ID,
    model_name=MODEL_NAME,
    method=METHOD,
    dataset_name="temporal_wiki",
    precision=PRECISION,
    batch_size=BATCH_SIZE,
    grad_accum_steps=GRAD_ACCUM_STEPS,
    learning_rate=LEARNING_RATE,
    epochs_per_period=EPOCHS_PER_PERIOD,
    max_passages_per_period=MAX_PASSAGES_PER_PERIOD,
    dataset_fraction=DATASET_FRACTION,
    log_every_n_steps=10,
    eval_after_each_period=True,
    capture_activations=CAPTURE_ACTIVATIONS,
    seed=SEED,
)

if METHOD == "lora":
    config_kwargs.update(
        lora_r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        lora_target_modules=LORA_TARGET_MODULES,
    )
elif METHOD == "smf":
    config_kwargs.update(
        smf_memory_size=SMF_MEMORY_SIZE,
        smf_sparsity_ratio=SMF_SPARSITY_RATIO,
        smf_update_layers=SMF_UPDATE_LAYERS,
        smf_regularization_weight=SMF_REGULARIZATION_WEIGHT,
        smf_learning_rate=SMF_LEARNING_RATE,
        smf_freeze_backbone=True,
    )
elif METHOD == "casm":
    config_kwargs.update(
        casm_num_slots=CASM_NUM_SLOTS,
        casm_router_hidden_size=CASM_ROUTER_HIDDEN_SIZE,
        casm_top_k=CASM_TOP_K,
        casm_router_temperature=CASM_ROUTER_TEMPERATURE,
        casm_sparsity_weight=CASM_SPARSITY_WEIGHT,
        casm_overlap_weight=CASM_OVERLAP_WEIGHT,
        casm_branch_on_contradiction=CASM_BRANCH_ON_CONTRADICTION,
        casm_memory_size=CASM_MEMORY_SIZE,
    )

cfg = TrainConfig(**config_kwargs)
cfg.validate()

print(f"Config validated for method={cfg.method}")
print()

# ---------------------------------------------------------------------------
# Detect existing checkpoint for automatic resume
# ---------------------------------------------------------------------------
RESUME_FROM = None
run_root = os.path.join(CHECKPOINT_DIR, RUN_ID)
latest_json = os.path.join(run_root, "latest.json")

if os.path.exists(latest_json):
    with open(latest_json) as f:
        pointer = json.load(f)
    last_period = pointer.get("last_period", "unknown")
    print(f"Existing checkpoint found (last period: {last_period}).")
    print(f"Training will resume from: {run_root}")
    RESUME_FROM = run_root
else:
    print("No existing checkpoint — starting fresh.")

print()

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
from training.train_runner import (
    run_training,
    build_real_model_and_tokenizer,
    load_real_model_and_tokenizer,
    build_augmented_dataset,
    build_real_dataset,
)

dataset_factory = build_augmented_dataset if USE_AUGMENTED_DATA else build_real_dataset
print(f"Dataset factory: {'augmented CSVs' if USE_AUGMENTED_DATA else 'TWiki_Diffsets.zip'}")
print()

results = run_training(
    cfg,
    model_factory=build_real_model_and_tokenizer,
    resume_model_factory=load_real_model_and_tokenizer,
    dataset_factory=dataset_factory,
    checkpoint_dir=CHECKPOINT_DIR,
    training_units=PERIODS,
    resume_from=RESUME_FROM,
)

# ---------------------------------------------------------------------------
# Training summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 50)
print("TRAINING SUMMARY")
print("=" * 50)
for i, r in enumerate(results):
    period = PERIODS[i] if i < len(PERIODS) else f"period_{i}"
    print(f"\n{period}:")
    loss = r.get("train_loss_final")
    print(f"  Final loss:        {loss:.4f}" if loss is not None else "  Final loss:        N/A")
    print(f"  Passages trained:  {r.get('n_passages_trained', 'N/A')}")
    print(f"  Train time (s):    {r.get('train_duration_sec', 0):.1f}")
    if "evaluation" in r:
        for split, eval_result in r["evaluation"].items():
            if isinstance(eval_result, dict):
                plasticity = eval_result.get("plasticity")
                stability = eval_result.get("stability")
                token_f1 = eval_result.get("token_f1")
            else:
                plasticity = eval_result.plasticity
                stability = eval_result.stability
                token_f1 = eval_result.token_f1
            p = f"{plasticity:.3f}" if plasticity is not None else "N/A"
            s = f"{stability:.3f}" if stability is not None else "N/A"
            f = f"{token_f1:.3f}" if token_f1 is not None else "N/A"
            print(f"  Eval [{split:9s}]: plasticity={p}  stability={s}  token_f1={f}")
    print(f"  Checkpoint:        {r.get('checkpoint_path', 'N/A')}")

# ---------------------------------------------------------------------------
# Eval summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 50)
print("EVAL SUMMARY")
print("=" * 50)
for i, r in enumerate(results):
    period = PERIODS[i] if i < len(PERIODS) else f"period_{i}"
    print(f"\n{period}:")
    if "evaluation" in r:
        for split, eval_result in r["evaluation"].items():
            if isinstance(eval_result, dict):
                plasticity = eval_result.get("plasticity")
                stability = eval_result.get("stability")
                token_f1 = eval_result.get("token_f1")
            else:
                plasticity = eval_result.plasticity
                stability = eval_result.stability
                token_f1 = eval_result.token_f1
            p = f"{plasticity:.3f}" if plasticity is not None else "N/A"
            s = f"{stability:.3f}" if stability is not None else "N/A"
            f = f"{token_f1:.3f}" if token_f1 is not None else "N/A"
            print(f"  [{split:9s}]: plasticity={p}  stability={s}  token_f1={f}")
    else:
        print("  (no evaluation results)")

# ---------------------------------------------------------------------------
# Sample generation
# ---------------------------------------------------------------------------
print("\n" + "=" * 50)
print("SAMPLE GENERATIONS (sep_oct changed probes)")
print("=" * 50)

checkpoint_path = results[0]["checkpoint_path"]
model, tokenizer = load_real_model_and_tokenizer(cfg, checkpoint_path)
model.eval()

device = next(model.parameters()).device

sample_dataset = dataset_factory("sep_oct", cfg)
sample_dataset.load("changed")
probes = sample_dataset.get_probes("changed")[:5]

for probe in probes:
    inputs = tokenizer(probe.prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,
        )
    generated = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()

    print(f"Prompt:   {probe.prompt}")
    print(f"Expected: {probe.ground_truth}")
    print(f"Got:      {generated!r}")
    print()

# ---------------------------------------------------------------------------
# Parameter summary
# ---------------------------------------------------------------------------
from collections import defaultdict

print("=" * 50)
print("PARAMETER SUMMARY")
print("=" * 50)

total_params = 0
trainable_params = 0
groups = defaultdict(lambda: {"total": 0, "trainable": 0})

for name, param in model.named_parameters():
    n = param.numel()
    top = name.split(".")[0]
    total_params += n
    groups[top]["total"] += n
    if param.requires_grad:
        trainable_params += n
        groups[top]["trainable"] += n

col = 42
print(f"\n{'Module':<{col}} {'Total params':>14} {'Trainable':>12} {'%':>7}")
print("-" * (col + 36))
for group, counts in sorted(groups.items()):
    t, tr = counts["total"], counts["trainable"]
    pct = 100 * tr / t if t else 0
    marker = "  <-- updated" if tr > 0 else ""
    print(f"  {group:<{col-2}} {t:>14,} {tr:>12,} {pct:>6.1f}%{marker}")
print("-" * (col + 36))
pct = 100 * trainable_params / total_params if total_params else 0
print(f"  {'TOTAL':<{col-2}} {total_params:>14,} {trainable_params:>12,} {pct:>6.1f}%")
print(f"\nTrainable: {trainable_params:,} / {total_params:,}  ({pct:.4f}% of model)")

print("\nDone.")
