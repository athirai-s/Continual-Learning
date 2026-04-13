#!/usr/bin/env python3
"""Export eval cloze prompts — one file per time period, split into chunks.

Each data line: <cloze_prompt><TAB><ground_truth>
Output files:  <outdir>/<period>_chunk<N>of<total>.txt

Usage:
    uv run python dataset_utils/export_eval_prompts.py
    uv run python dataset_utils/export_eval_prompts.py --chunks 5
    uv run python dataset_utils/export_eval_prompts.py --outdir dataset_utils/prompts --chunks 1
"""
import argparse
import math
import sys
from pathlib import Path

# Ensure the project root is on sys.path when the script is run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from casf_dataset_api import TemporalWikiDataset
import casf_dataset_api.download_dataset_scripts.data.temporal_wiki as _tw

# The module's PROBES_ZIP/DIFFSETS_ZIP constants assume data/ at project root,
# but the actual ZIPs live alongside temporal_wiki.py.  Patch them here.
_DATA_DIR = Path(_tw.__file__).parent
_tw.PROBES_ZIP   = _DATA_DIR / "TWiki_Probes.zip"
_tw.DIFFSETS_ZIP = _DATA_DIR / "TWiki_Diffsets.zip"

PERIODS = ["aug_sep", "sep_oct", "oct_nov", "nov_dec"]
SPLITS  = ["changed", "unchanged"]
DEFAULT_OUTDIR = Path("dataset_utils/eval_prompts")
DEFAULT_CHUNKS = 3


def write_chunk(path: Path, period: str, chunk_idx: int, total_chunks: int,
                probes: list) -> None:
    """Write a single chunk file."""
    with path.open("w", encoding="utf-8") as f:
        f.write(f"### PERIOD={period}  CHUNK={chunk_idx}of{total_chunks}  N={len(probes)}\n")
        for probe in probes:
            f.write(f"{probe.prompt}\t{probe.ground_truth}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Export eval cloze prompts — one file per period, split into chunks."
    )
    parser.add_argument(
        "--outdir", type=Path, default=DEFAULT_OUTDIR,
        help=f"Output directory (default: {DEFAULT_OUTDIR})",
    )
    parser.add_argument(
        "--chunks", type=int, default=DEFAULT_CHUNKS,
        help=f"Number of chunks per period (default: {DEFAULT_CHUNKS})",
    )
    args = parser.parse_args()

    if args.chunks < 1:
        parser.error("--chunks must be >= 1")

    args.outdir.mkdir(parents=True, exist_ok=True)

    total_written = 0
    files_written = []

    for period in PERIODS:
        dataset = TemporalWikiDataset(period=period)

        # Collect all probes for this period (changed then unchanged)
        all_probes = []
        for split in SPLITS:
            all_probes.extend(dataset.get_probes(split))

        n = len(all_probes)
        chunk_size = math.ceil(n / args.chunks)

        for i in range(args.chunks):
            chunk_probes = all_probes[i * chunk_size : (i + 1) * chunk_size]
            if not chunk_probes:
                continue  # fewer probes than chunks requested
            chunk_num = i + 1
            fname = args.outdir / f"{period}_chunk{chunk_num}of{args.chunks}.txt"
            write_chunk(fname, period, chunk_num, args.chunks, chunk_probes)
            files_written.append(fname)
            total_written += len(chunk_probes)

        print(f"  {period}: {n} prompts -> {args.chunks} chunk(s)")

    print(f"\nWrote {total_written} prompts across {len(files_written)} files in {args.outdir}/")


if __name__ == "__main__":
    main()
