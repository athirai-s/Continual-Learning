#!/usr/bin/env python3
"""Generate augmented training passages using the Gemini API.

Reads cloze eval prompts from dataset_utils/prompts/*.txt, calls
gemini-2.0-flash for each one, and writes one CSV per time period to
data/augmented/TWiki_Diffsets/<period>.csv.

Usage:
    uv run python dataset_utils/generate_dataset.py
    uv run python dataset_utils/generate_dataset.py --prompts-dir dataset_utils/prompts --outdir data/augmented/TWiki_Diffsets
    uv run python dataset_utils/generate_dataset.py --limit 20
"""
import argparse
import csv
import os
import random
import subprocess
import sys
import threading
import time
import concurrent.futures
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---------------------------------------------------------------------------
# Load .env
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "python-dotenv"])
    from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from google import genai
from google.genai import types

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PERIODS     = ["aug_sep", "sep_oct", "oct_nov", "nov_dec"]
MODEL_NAME  = "gemini-2.5-flash-lite"
MAX_WORKERS = 5
LOG_EVERY   = 10
DEFAULT_PROMPTS_DIR = Path("dataset_utils/prompts")
DEFAULT_OUTDIR      = Path("data/augmented/TWiki_Diffsets")

RETRY_DELAYS = [5, 15, 30, 60]  # seconds between successive rate-limit retries

INSTRUCTION = (
    "You are generating training data for a machine learning model. "
    "Write a passage of 1-3 short sentences (under 20 words each). "
    "Exactly one sentence must state the given fact naturally — use the exact answer provided. "
    "Any remaining sentences should be brief, topically related context (not random filler). "
    "Do not repeat or paraphrase the prompt. Output only the passage, nothing else."
)

# ---------------------------------------------------------------------------
# Load prompts grouped by period
# ---------------------------------------------------------------------------

def load_prompts_by_period(prompts_dir: Path) -> dict[str, list[tuple[str, str]]]:
    """Return {period: [(cloze_prompt, ground_truth), ...]} in sorted chunk order."""
    by_period = defaultdict(list)
    for path in sorted(prompts_dir.glob("*.txt")):
        period = None
        for p in PERIODS:
            if path.name.startswith(p):
                period = p
                break
        if period is None:
            print(f"  [WARN] Skipping unrecognised file: {path.name}")
            continue
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("###"):
                    continue
                parts = line.split("\t")
                prompt       = parts[0]
                ground_truth = parts[1] if len(parts) > 1 else ""
                if prompt:
                    by_period[period].append((prompt, ground_truth))
    return dict(by_period)

# ---------------------------------------------------------------------------
# Gemini call with rate-limit retry
# ---------------------------------------------------------------------------

def clean_passage(text: str) -> str:
    """Normalize whitespace and strip leading/trailing quotes the model sometimes adds."""
    import re
    text = text.strip().strip('"').strip("'").strip()
    text = re.sub(r'  +', ' ', text)          # collapse multiple spaces
    text = re.sub(r' +([.,!?])', r'\1', text) # remove space before punctuation
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
            is_rate_limit = "429" in msg or "503" in msg or "quota" in msg.lower() or "rate" in msg.lower() or "unavailable" in msg.lower()
            if is_rate_limit and attempt < len(RETRY_DELAYS):
                print(f"  [RATE LIMIT] waiting {wait}s before retry {attempt} ...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Exhausted retries")


def generate_passage(client: genai.Client, prompt: str, answer: str, idx: int) -> tuple[int, str]:
    try:
        return idx, call_gemini(client, prompt, answer)
    except Exception as e:
        print(f"  [ERROR] idx={idx} failed: {e}")
        return idx, "ERROR"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fmt_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s:02d}s" if m else f"{s}s"

# ---------------------------------------------------------------------------
# Process one period
# ---------------------------------------------------------------------------

def process_period(period: str, prompts: list[str], client: genai.Client, outdir: Path,
                   period_num: int, total_periods: int) -> tuple[int, int]:
    n = len(prompts)
    print(f"\n{'='*60}")
    print(f"  Period {period_num}/{total_periods}: {period}  ({n} prompts)")
    print(f"{'='*60}")

    results = [None] * n
    completed = 0
    lock = threading.Lock()
    period_start = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(generate_passage, client, prompt, answer, i): i
            for i, (prompt, answer) in enumerate(prompts)
        }
        for future in concurrent.futures.as_completed(futures):
            idx, passage = future.result()
            results[idx] = passage
            with lock:
                completed += 1
                if completed % LOG_EVERY == 0 or completed == n:
                    elapsed = time.time() - period_start
                    pct  = completed / n * 100
                    rate = completed / elapsed if elapsed > 0 else 0
                    eta  = (n - completed) / rate if rate > 0 else 0
                    print(
                        f"  {completed:>4}/{n}  ({pct:5.1f}%)  "
                        f"elapsed={fmt_duration(elapsed)}  eta={fmt_duration(eta)}"
                    )

    # Fill any slots that never completed (shouldn't happen, but be safe)
    n_missing = sum(1 for r in results if r is None)
    if n_missing:
        print(f"  [WARN] {n_missing} result(s) were None — filling with ERROR")
        results = [r if r is not None else "ERROR" for r in results]

    if len(results) != n:
        print(f"  [WARN] Row count mismatch: got {len(results)}, expected {n}")

    n_errors = sum(1 for r in results if r == "ERROR")
    n_ok     = len(results) - n_errors

    # Write to a temp file first, then rename — avoids a partial CSV on failure
    out_path  = outdir / f"{period}.csv"
    tmp_path  = out_path.with_suffix(".tmp")
    try:
        with tmp_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            writer.writerow(["text"])
            for passage in results:
                writer.writerow([passage])
        tmp_path.replace(out_path)
    except Exception as e:
        print(f"  [ERROR] Failed to save {out_path}: {e}")
        print(f"  [ERROR] Partial data may be in {tmp_path}")
        raise

    total_time = time.time() - period_start
    print(f"\n  Done {period}: {n_ok} ok, {n_errors} errors in {fmt_duration(total_time)}")
    print(f"  Saved -> {out_path}")
    return n_ok, n_errors

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate training passages via Gemini.")
    parser.add_argument("--prompts-dir", type=Path, default=DEFAULT_PROMPTS_DIR)
    parser.add_argument("--outdir",      type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--limit",       type=int,  default=None,
                        help="Cap total prompts processed (for testing)")
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        sys.exit("Error: GEMINI_API_KEY environment variable is not set.")

    client = genai.Client(api_key=api_key)

    print(f"Loading prompts from {args.prompts_dir} ...")
    by_period = load_prompts_by_period(args.prompts_dir)
    if not by_period:
        sys.exit(f"No prompts found in {args.prompts_dir}")

    for period in PERIODS:
        if period not in by_period:
            print(f"[WARN] No prompts found for period {period}, skipping.")

    args.outdir.mkdir(parents=True, exist_ok=True)

    active_periods = [p for p in PERIODS if p in by_period]

    if args.limit is not None:
        remaining = args.limit
        for p in active_periods:
            take = min(remaining, len(by_period[p]))
            by_period[p] = by_period[p][:take]
            remaining = max(0, remaining - take)
        active_periods = [p for p in active_periods if by_period[p]]
        print(f"[--limit] Capped to {args.limit} prompts total.")

    total_prompts = sum(len(by_period[p]) for p in active_periods)
    print(f"\n{len(active_periods)} periods  |  {total_prompts} total prompts  |  {MAX_WORKERS} workers  |  model={MODEL_NAME}")

    run_start    = time.time()
    total_ok     = total_errors = 0
    for i, period in enumerate(active_periods, 1):
        ok, errors = process_period(period, by_period[period], client, args.outdir, i, len(active_periods))
        total_ok     += ok
        total_errors += errors

    total_time = time.time() - run_start
    print(f"\n{'='*60}")
    print(f"  All periods complete in {fmt_duration(total_time)}")
    print(f"  Succeeded : {total_ok}")
    print(f"  Errored   : {total_errors}")
    print(f"  Output dir: {args.outdir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
