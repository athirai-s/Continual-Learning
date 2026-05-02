"""Diagnostic: contradiction structure for TemporalWiki vs synthetic.

For each dataset computes, per period:
    n_changed, n_unchanged
    sequential_contradictions — (subject, relation) keys whose value in the
        current period differs from any value seen in the immediately prior
        period.  This is exactly the signal CASM's ContradictionDetector fires
        on.  A period with zero sequential contradictions gives CASM's
        branch-on-contradiction mechanism nothing to activate.
    contradiction_rate — sequential_contradictions / n_changed
And across all period pairs:
    pairwise_changed_key_overlap — number of (subject, relation) keys that
        appear as "changed" in both periods of the pair.  Zero means every
        period's "changed" set is disjoint from every other's.

Purpose: one-slide mechanistic explanation for why CASM underfires on
TemporalWiki and is the right fit for the synthetic benchmark.

Usage:
    python3 diagnose_contradiction_structure.py
    # prints tables and writes diagnostics/contradiction_structure.json
"""
from __future__ import annotations

import csv
import io
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TW_ZIP = ROOT / "casf_dataset_api" / "download_dataset_scripts" / "data" / "TWiki_Probes.zip"
TW_PERIODS = ["aug_sep", "sep_oct", "oct_nov", "nov_dec"]
TW_PERIOD_TO_PREFIX = {
    "aug_sep": "0801-0901",
    "sep_oct": "0901-1001",
    "oct_nov": "1001-1101",
    "nov_dec": "1101-1201",
}

SYNTH_PROBES = ROOT / "data" / "probes.json"
SYNTH_PERIODS = ["2018", "2020", "2022", "2024"]

OUTPUT_PATH = ROOT / "diagnostics" / "contradiction_structure.json"

Key = tuple[str, str]


def load_tw_period(period: str) -> dict[str, dict[Key, str]]:
    """Return {'changed': {(subj,rel): object}, 'unchanged': {...}} for one TW period."""
    prefix = TW_PERIOD_TO_PREFIX[period]
    out: dict[str, dict[Key, str]] = {"changed": {}, "unchanged": {}}
    with zipfile.ZipFile(TW_ZIP) as z:
        for split in ("changed", "unchanged"):
            filename = f"twiki_probes/{prefix}_{split}.csv"
            with z.open(filename) as f:
                reader = csv.DictReader(io.TextIOWrapper(f, "utf-8"))
                for row in reader:
                    key = (row["subject"], row["relation"])
                    out[split][key] = row["object"]
    return out


def load_synth() -> dict[str, dict[str, dict[Key, str]]]:
    """Return {period: {'changed': {(subj,rel): current_value}, 'unchanged': {...}}}."""
    with open(SYNTH_PROBES) as f:
        data = json.load(f)
    out: dict[str, dict[str, dict[Key, str]]] = {}
    for period, splits in data.items():
        out[period] = {"changed": {}, "unchanged": {}}
        for split_name in ("changed", "unchanged"):
            for probe in splits.get(split_name, []):
                key = (probe["subject"], probe["relation"])
                out[period][split_name][key] = probe.get("current_value", "")
    return out


def compute_stats(
    per_period: dict[str, dict[str, dict[Key, str]]],
    periods: list[str],
    dataset_name: str,
) -> dict:
    """Per-period counts, sequential contradictions, pairwise changed-key overlap."""
    rows = []
    prev_values: dict[Key, str] = {}
    for i, p in enumerate(periods):
        changed = per_period[p]["changed"]
        unchanged = per_period[p]["unchanged"]
        n_changed = len(changed)
        n_unchanged = len(unchanged)

        seq_contra = 0
        if i > 0:
            for key, new_val in changed.items():
                old_val = prev_values.get(key)
                if old_val is not None and old_val != new_val:
                    seq_contra += 1

        rows.append(
            {
                "period": p,
                "n_changed": n_changed,
                "n_unchanged": n_unchanged,
                "sequential_contradictions": seq_contra,
                "contradiction_rate": (seq_contra / n_changed) if n_changed else 0.0,
            }
        )

        # prev_values carries the union of the period's changed + unchanged values
        # forward so the NEXT period can see what values existed before it.
        prev_values = {**unchanged, **changed}

    changed_keys_by_period = {p: set(per_period[p]["changed"].keys()) for p in periods}
    overlap: dict[str, int] = {}
    for i, pi in enumerate(periods):
        for pj in periods[i + 1 :]:
            overlap[f"{pi}<->{pj}"] = len(
                changed_keys_by_period[pi] & changed_keys_by_period[pj]
            )

    return {
        "dataset": dataset_name,
        "per_period": rows,
        "pairwise_changed_key_overlap": overlap,
    }


def print_table(stats: dict) -> None:
    print(f"\n=== {stats['dataset']} ===")
    header = (
        f"{'Period':>10}  {'#Changed':>9}  {'#Unchanged':>11}  "
        f"{'SeqContra':>10}  {'Rate':>6}"
    )
    print(header)
    print("-" * len(header))
    for r in stats["per_period"]:
        rate = f"{r['contradiction_rate']:.2f}"
        print(
            f"{r['period']:>10}  {r['n_changed']:>9}  {r['n_unchanged']:>11}  "
            f"{r['sequential_contradictions']:>10}  {rate:>6}"
        )
    print("\nPairwise changed-key overlap:")
    for pair, n in stats["pairwise_changed_key_overlap"].items():
        print(f"  {pair}: {n}")


def main() -> None:
    tw_per_period = {p: load_tw_period(p) for p in TW_PERIODS}
    tw_stats = compute_stats(tw_per_period, TW_PERIODS, "TemporalWiki")
    print_table(tw_stats)

    synth_per_period = load_synth()
    synth_stats = compute_stats(synth_per_period, SYNTH_PERIODS, "Synthetic")
    print_table(synth_stats)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump({"temporal_wiki": tw_stats, "synthetic": synth_stats}, f, indent=2)
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
