#!/usr/bin/env python3
"""Redistribute change boundaries evenly across the three period transitions.

Problem:  All changed=true facts currently change at the 2020->2022 boundary,
          producing a severely skewed probe distribution.

Fix:      For each changed fact, extract value_a (old value) and value_b (new
          value), assign a boundary evenly across the three transitions, then
          rewrite the four period-value fields so the change lands exactly at
          the assigned boundary.

Boundary semantics:
    2018->2020:  value_a for 2018;              value_b for 2020, 2022, 2024
    2020->2022:  value_a for 2018, 2020;        value_b for 2022, 2024
    2022->2024:  value_a for 2018, 2020, 2022;  value_b for 2024

Unchanged facts (changed=false) are left exactly as-is.

Usage:
    uv run python data/redistribute_changes.py
    uv run python data/redistribute_changes.py --input data/synthetic_facts_raw.json \\
                                               --output data/synthetic_facts_raw.json \\
                                               --seed 42
"""
import argparse
import json
import random
from pathlib import Path

PERIODS = ["2018", "2020", "2022", "2024"]
BOUNDARIES = ["2018->2020", "2020->2022", "2022->2024"]

# For each boundary: which periods get value_a vs value_b
BOUNDARY_SPLIT: dict[str, tuple[list[str], list[str]]] = {
    "2018->2020": (["2018"],                   ["2020", "2022", "2024"]),
    "2020->2022": (["2018", "2020"],            ["2022", "2024"]),
    "2022->2024": (["2018", "2020", "2022"],    ["2024"]),
}


def redistribute(facts: list[dict], seed: int = 42) -> list[dict]:
    rng = random.Random(seed)

    changed_indices = [i for i, f in enumerate(facts) if f.get("changed")]
    n = len(changed_indices)

    # Build an even assignment: repeat BOUNDARIES enough times, trim, then shuffle
    assignment = (BOUNDARIES * (n // 3 + 1))[:n]
    rng.shuffle(assignment)

    result = [dict(f) for f in facts]  # shallow copy every fact

    for idx, boundary in zip(changed_indices, assignment):
        fact = result[idx]

        # Extract the two values from the existing data.
        # value_a is the value in 2018 (always the "old" value before the
        # original 2020->2022 change that was generated).
        # value_b is the value in 2022 (always the "new" value).
        value_a = fact["value_2018"]
        value_b = fact["value_2022"]

        a_periods, b_periods = BOUNDARY_SPLIT[boundary]
        for p in PERIODS:
            key = f"value_{p}"
            if p in a_periods:
                fact[key] = value_a
            else:
                fact[key] = value_b

    return result


def count_distribution(facts: list[dict]) -> None:
    """Print changed/unchanged counts per period boundary for verification."""
    period_pairs = [
        ("2018->2020", "2018", "2020"),
        ("2020->2022", "2020", "2022"),
        ("2022->2024", "2022", "2024"),
    ]
    print("  Boundary     changed  unchanged")
    print("  -----------  -------  ---------")
    for label, pa, pb in period_pairs:
        changed = sum(
            1 for f in facts
            if f.get("changed") and f[f"value_{pb}"] != f[f"value_{pa}"]
        )
        unchanged = sum(
            1 for f in facts
            if f.get("changed") and f[f"value_{pb}"] == f[f"value_{pa}"]
        )
        stable = sum(1 for f in facts if not f.get("changed"))
        print(f"  {label:<13}  {changed:>7}  {unchanged:>9}  (stable facts: {stable})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Redistribute change boundaries evenly across three transitions"
    )
    parser.add_argument(
        "--input", type=Path, default=Path("data/synthetic_facts_raw.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/synthetic_facts_raw.json")
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    facts = json.loads(args.input.read_text())
    n_changed = sum(1 for f in facts if f.get("changed"))
    n_stable = sum(1 for f in facts if not f.get("changed"))
    print(f"Loaded {len(facts)} facts ({n_changed} changed, {n_stable} stable) from {args.input}")

    print("\nBefore redistribution:")
    count_distribution(facts)

    facts = redistribute(facts, seed=args.seed)

    print("\nAfter redistribution:")
    count_distribution(facts)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(facts, indent=2))
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
