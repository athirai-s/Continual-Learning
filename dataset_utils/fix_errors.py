#!/usr/bin/env python3
"""Re-generate any ERROR rows in already-written CSVs.

Reads each <period>.csv from the output dir, finds rows where text == ERROR,
re-calls Gemini for those rows using the original prompt file, and overwrites
the CSV with the fixed rows in place.

Usage:
    uv run python dataset_utils/fix_errors.py
    uv run python dataset_utils/fix_errors.py --outdir data/augmented/TWiki_Diffsets --prompts-dir dataset_utils/prompts
"""
import argparse
import csv
import os
import random
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from dotenv import load_dotenv
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "python-dotenv"])
    from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from google import genai
from google.genai import types

PERIODS     = ["aug_sep", "sep_oct", "oct_nov", "nov_dec"]
MODEL_NAME  = "gemini-2.5-flash-lite"
RETRY_DELAYS = [5, 15, 30, 60]

DEFAULT_OUTDIR      = Path("data/augmented/TWiki_Diffsets")
DEFAULT_PROMPTS_DIR = Path("dataset_utils/prompts")

INSTRUCTION = (
    "You are generating training data for a machine learning model. "
    "Write a passage of 1-3 short sentences (under 20 words each). "
    "Exactly one sentence must state the given fact naturally — use the exact answer provided. "
    "Any remaining sentences should be brief, topically related context (not random filler). "
    "Do not repeat or paraphrase the prompt. Output only the passage, nothing else."
)


def clean_passage(text: str) -> str:
    import re
    text = text.strip().strip('"').strip("'").strip()
    text = re.sub(r'  +', ' ', text)
    text = re.sub(r' +([.,!?])', r'\1', text)
    return text


def call_gemini(client: genai.Client, prompt: str, answer: str) -> str:
    full_prompt = f"{INSTRUCTION}\n\nPrompt: {prompt}\nAnswer: {answer}"
    for attempt, wait in enumerate(RETRY_DELAYS, 1):
        try:
            temperature = round(random.uniform(0.5, 1.2), 2)
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=full_prompt,
                config=types.GenerateContentConfig(temperature=temperature),
            )
            return clean_passage(response.text)
        except Exception as e:
            msg = str(e)
            is_retryable = any(x in msg for x in ["429", "503"]) or \
                           any(x in msg.lower() for x in ["quota", "rate", "unavailable"])
            if is_retryable and attempt < len(RETRY_DELAYS):
                print(f"    [RETRY {attempt}] waiting {wait}s ...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Exhausted retries")


def load_prompts_for_period(prompts_dir: Path, period: str) -> list[tuple[str, str]]:
    prompts = []
    for path in sorted(prompts_dir.glob(f"{period}_*.txt")):
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("###"):
                    continue
                parts = line.split("\t")
                prompt       = parts[0]
                ground_truth = parts[1] if len(parts) > 1 else ""
                if prompt:
                    prompts.append((prompt, ground_truth))
    return prompts


def fix_period(period: str, outdir: Path, prompts_dir: Path, client: genai.Client) -> tuple[int, int]:
    csv_path = outdir / f"{period}.csv"
    if not csv_path.exists():
        print(f"  [{period}] No CSV found, skipping.")
        return 0, 0

    with csv_path.open(encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    error_indices = [i for i, row in enumerate(rows) if row and row[0] == "ERROR"]
    if not error_indices:
        print(f"  [{period}] No errors found.")
        return 0, 0

    print(f"  [{period}] {len(error_indices)} ERROR row(s) to fix ...")

    prompts = load_prompts_for_period(prompts_dir, period)
    if len(prompts) != len(rows):
        print(f"  [{period}] WARNING: prompt count ({len(prompts)}) != CSV row count ({len(rows)}), skipping.")
        return 0, 0

    fixed = 0
    still_errors = 0
    for i, idx in enumerate(error_indices, 1):
        prompt, answer = prompts[idx]
        print(f"    Fixing {i}/{len(error_indices)}: idx={idx} ...")
        try:
            passage = call_gemini(client, prompt, answer)
            rows[idx] = [passage]
            fixed += 1
        except Exception as e:
            print(f"    [ERROR] idx={idx} still failed: {e}")
            still_errors += 1

    tmp_path = csv_path.with_suffix(".tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)
    tmp_path.replace(csv_path)

    print(f"  [{period}] Fixed {fixed}, still errored {still_errors} -> {csv_path}")
    return fixed, still_errors


def main():
    parser = argparse.ArgumentParser(description="Fix ERROR rows in generated CSVs.")
    parser.add_argument("--outdir",      type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--prompts-dir", type=Path, default=DEFAULT_PROMPTS_DIR)
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        sys.exit("Error: GEMINI_API_KEY environment variable is not set.")

    client = genai.Client(api_key=api_key)

    total_fixed = total_errors = 0
    for period in PERIODS:
        fixed, errors = fix_period(period, args.outdir, args.prompts_dir, client)
        total_fixed  += fixed
        total_errors += errors

    print(f"\nDone. Fixed: {total_fixed}  Still errored: {total_errors}")


if __name__ == "__main__":
    main()
