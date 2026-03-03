"""
Download WikiFactDiff dataset from HuggingFace and save to data/ directory.
https://huggingface.co/datasets/Orange/WikiFactDiff

Usage:
    uv run download_wikifactdiff.py
    # or
    python download_wikifactdiff.py

Requirements:
    uv pip install datasets huggingface_hub pyarrow
"""

import os
from datasets import load_dataset

# ── Config ────────────────────────────────────────────────────────────────────
OUTPUT_DIR = "data/wikifactdiff"

# ── Download ──────────────────────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading WikiFactDiff from HuggingFace...")
print("  Source : Orange/WikiFactDiff")
print("  T_old  : 4 January 2021")
print("  T_new  : 27 February 2023")
print()

ds = load_dataset("Orange/WikiFactDiff")

print(f"Dataset splits: {list(ds.keys())}")
for split, data in ds.items():
    print(f"  {split}: {len(data):,} records")
print()

# ── Save to parquet ───────────────────────────────────────────────────────────
for split, data in ds.items():
    out_path = os.path.join(OUTPUT_DIR, f"{split}.parquet")
    print(f"Saving {split} -> {out_path} ...")
    data.to_parquet(out_path)
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"  Done. ({size_mb:.1f} MB)")

print()

# ── Quick sanity check ────────────────────────────────────────────────────────
print("Sample record fields:")
sample = next(iter(ds.values()))[0]
for key, val in sample.items():
    print(f"  {key}: {repr(val)[:80]}")

print()
print("Done. Load later with:")
print("  import pandas as pd")
print(f"  df = pd.read_parquet('{OUTPUT_DIR}/train.parquet')")