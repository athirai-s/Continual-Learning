#!/usr/bin/env python3
"""Convert synthetic_facts_raw.json into prompt files for generate_dataset.py.

generate_dataset.py matches prompt-file names against PERIODS:
    ["aug_sep", "sep_oct", "oct_nov", "nov_dec"]

Because synthetic data uses year-based period names ("2018", "2020", …)
but generate_dataset.py cannot be modified, this script maps years to the
aug_sep naming convention for FILE NAMES ONLY.  The mapping is:

    2018  →  aug_sep    (written to aug_sep_synthetic.txt)
    2020  →  sep_oct    (written to sep_oct_synthetic.txt)
    2022  →  oct_nov    (written to oct_nov_synthetic.txt)
    2024  →  nov_dec    (written to nov_dec_synthetic.txt)

Output directory:  dataset_utils/prompts/synthetic/

After running this script, generate augmented passages with:

    uv run python dataset_utils/generate_dataset.py \\
        --prompts-dir dataset_utils/prompts/synthetic \\
        --outdir data/augmented/synthetic

generate_dataset.py writes CSVs named aug_sep.csv, sep_oct.csv, etc.
Rename them to year names immediately after generation:

    mv data/augmented/synthetic/aug_sep.csv data/augmented/synthetic/2018.csv
    mv data/augmented/synthetic/sep_oct.csv data/augmented/synthetic/2020.csv
    mv data/augmented/synthetic/oct_nov.csv data/augmented/synthetic/2022.csv
    mv data/augmented/synthetic/nov_dec.csv data/augmented/synthetic/2024.csv

SyntheticDataset expects year-named files (2018.csv, 2020.csv, etc.).

Input format per line (tab-separated):
    The harbour master of Veldris Corp is\tMaren Holt

Usage:
    uv run python data/build_augmentation_prompts.py
    uv run python data/build_augmentation_prompts.py \\
        --facts data/synthetic_facts_raw.json \\
        --outdir dataset_utils/prompts/synthetic
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PERIODS = ["2018", "2020", "2022", "2024"]

# Maps synthetic year names → aug_sep-style names used by generate_dataset.py
YEAR_TO_AUGMENTED_NAME: dict[str, str] = {
    "2018": "aug_sep",
    "2020": "sep_oct",
    "2022": "oct_nov",
    "2024": "nov_dec",
}

DEFAULT_FACTS = Path("data/synthetic_facts_raw.json")
DEFAULT_OUTDIR = Path("dataset_utils/prompts/synthetic")


def build_prompt_line(fact: dict, period: str) -> str:
    """Return a single tab-separated "prompt\tanswer" line."""
    relation_str = fact["relation"].replace("_", " ")
    entity = fact["entity"]
    value = fact[f"value_{period}"]
    prompt = f"The {relation_str} of {entity} is"
    return f"{prompt}\t{value}"


def build_augmentation_prompts(
    facts: list[dict],
    outdir: Path,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    for period in PERIODS:
        aug_name = YEAR_TO_AUGMENTED_NAME[period]
        out_path = outdir / f"{aug_name}_synthetic.txt"
        lines = [build_prompt_line(f, period) for f in facts]
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"  {period} -> {out_path}  ({len(lines)} prompts)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build generate_dataset.py prompt files from raw facts"
    )
    parser.add_argument("--facts", type=Path, default=DEFAULT_FACTS)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()

    facts = json.loads(args.facts.read_text())
    print(f"Loaded {len(facts)} facts from {args.facts}")

    build_augmentation_prompts(facts, args.outdir)

    print(f"\nDone. Run augmentation with:")
    print(
        f"  uv run python dataset_utils/generate_dataset.py"
        f" --prompts-dir {args.outdir}"
        f" --outdir data/augmented/synthetic"
    )


if __name__ == "__main__":
    main()
