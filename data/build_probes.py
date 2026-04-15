#!/usr/bin/env python3
"""Convert synthetic_facts_raw.json into serialised Probe objects.

Reads  data/synthetic_facts_raw.json
Writes data/probes.json

The output JSON is structured as:
    {
      "2018": {"changed": [...], "unchanged": [...]},
      "2020": {"changed": [...], "unchanged": [...]},
      "2022": {"changed": [...], "unchanged": [...]},
      "2024": {"changed": [...], "unchanged": [...]}
    }

Each probe entry is a dataclasses.asdict() dict of a Probe instance and can
be round-tripped back with Probe(**entry).

Period "2018" semantics
-----------------------
Period 2018 is the initial seeding period.  There is no predecessor period,
so no fact can have "changed" at this boundary.

  - "changed" split  →  all 800 facts with is_changed=False.
    The trainer calls get_probes("changed") to populate the registry after
    each period.  For period 2018 the registry is empty, so the detector
    finds zero contradictions, and all facts get written as new slots.

  - "unchanged" split  →  empty list.
    No prior period exists to serve as a stability baseline.

Subsequent period semantics
---------------------------
  - "changed"   →  facts where value_{prev} != value_{period}  (is_changed=True)
  - "unchanged" →  facts where value_{prev} == value_{period}  (is_changed=False)

valid_from computation
----------------------
For each fact/period combination, valid_from is the earliest period at which
the CURRENT value was already in effect.  This enables correct get_at() queries
on the MemoryRegistry.

Usage:
    uv run python data/build_probes.py
    uv run python data/build_probes.py --facts data/synthetic_facts_raw.json \\
                                       --output data/probes.json
"""
import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from casf_dataset_api.casf_types import Probe

PERIODS = ["2018", "2020", "2022", "2024"]

# Prompt templates per relation type (fallback for unknown relations)
PROMPT_TEMPLATES: dict[str, str] = {
    "chief_architect":          "Who is the chief architect of {entity}?",
    "vault_custodian":          "Who is the vault custodian at {entity}?",
    "harbour_master":           "Who is the harbour master of {entity}?",
    "tidal_surveyor":           "Who is the tidal surveyor at {entity}?",
    "dock_registrar":           "Who is the dock registrar of {entity}?",
    "bridge_engineer":          "Who is the bridge engineer for {entity}?",
    "conduit_inspector":        "Who is the conduit inspector at {entity}?",
    "treaty_signatory":         "Who is the treaty signatory representing {entity}?",
    "territorial_adjudicator":  "Who is the territorial adjudicator for {entity}?",
    "border_liaison":           "Who is the border liaison for {entity}?",
    "frequency_director":       "Who is the frequency director at {entity}?",
    "lens_calibrator":          "Who is the lens calibrator at {entity}?",
    "signal_interpreter":       "Who is the signal interpreter for {entity}?",
    "census_commissioner":      "Who is the census commissioner of {entity}?",
    "municipal_liaison":        "Who is the municipal liaison for {entity}?",
    "boundary_surveyor":        "Who is the boundary surveyor at {entity}?",
    "archive_keeper":           "Who is the archive keeper at {entity}?",
    "charter_custodian":        "Who is the charter custodian at {entity}?",
    "records_adjudicator":      "Who is the records adjudicator for {entity}?",
    "fuel_inspector":           "Who is the fuel inspector at {entity}?",
    "thermal_regulator_chief":  "Who is the thermal regulator chief at {entity}?",
    "land_commissioner":        "Who is the land commissioner of {entity}?",
    "water_registrar":          "Who is the water registrar at {entity}?",
    "crop_adjudicator":         "Who is the crop adjudicator for {entity}?",
}


def _make_prompt(entity: str, relation: str) -> str:
    template = PROMPT_TEMPLATES.get(relation)
    if template:
        return template.format(entity=entity)
    # Generic fallback: capitalise and humanise the relation slug
    readable = relation.replace("_", " ")
    return f"Who is the {readable} of {entity}?"


def _valid_from(fact: dict, period: str) -> str:
    """Return the earliest period at which the current value first held."""
    period_idx = PERIODS.index(period)
    value = fact[f"value_{period}"]
    # Scan backwards to find the period when the value first appeared
    for i in range(period_idx - 1, -1, -1):
        if fact[f"value_{PERIODS[i]}"] != value:
            # Value changed between PERIODS[i] and PERIODS[i+1]
            return PERIODS[i + 1]
    # Value was the same all the way back to the first period
    return PERIODS[0]


def build_probes_for_period(facts: list[dict], period: str) -> dict[str, list[dict]]:
    """Return {"changed": [...], "unchanged": [...]} of serialised Probe dicts."""
    period_idx = PERIODS.index(period)

    changed_probes: list[dict] = []
    unchanged_probes: list[dict] = []

    for fact in facts:
        entity = fact["entity"]
        relation = fact["relation"]
        current_value = fact[f"value_{period}"]
        prompt = _make_prompt(entity, relation)
        valid_from = _valid_from(fact, period)

        if period_idx == 0:
            # Initial period — all facts are new; no predecessor
            probe = Probe(
                subject=entity,
                relation=relation,
                current_value=current_value,
                previous_value=None,
                is_changed=False,
                timestamp=period,
                valid_from=valid_from,
                valid_until=None,
                prompt=prompt,
                ground_truth=current_value,
                source="synthetic",
                metadata={"period": period},
            )
            # Initial facts go into "changed" so the trainer writes them to the
            # registry; see module docstring for reasoning.
            changed_probes.append(asdict(probe))
            # "unchanged" is empty for the initial period
        else:
            prev_period = PERIODS[period_idx - 1]
            prev_value = fact[f"value_{prev_period}"]
            is_changed = current_value != prev_value

            probe = Probe(
                subject=entity,
                relation=relation,
                current_value=current_value,
                # previous_value deliberately left None here — ContradictionDetector
                # sets it in-place during detector.check().  We do NOT pre-set it.
                previous_value=None,
                is_changed=is_changed,
                timestamp=period,
                valid_from=valid_from,
                valid_until=None,
                prompt=prompt,
                ground_truth=current_value,
                source="synthetic",
                metadata={"period": period, "prev_period": prev_period},
            )
            if is_changed:
                changed_probes.append(asdict(probe))
            else:
                unchanged_probes.append(asdict(probe))

    return {"changed": changed_probes, "unchanged": unchanged_probes}


def build_all_probes(facts: list[dict]) -> dict[str, dict[str, list[dict]]]:
    """Return probes for all four periods."""
    return {period: build_probes_for_period(facts, period) for period in PERIODS}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Build probe JSON from raw facts")
    parser.add_argument(
        "--facts",
        type=Path,
        default=Path("data/synthetic_facts_raw.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/probes.json"),
    )
    args = parser.parse_args()

    facts = json.loads(args.facts.read_text())
    print(f"Loaded {len(facts)} facts from {args.facts}")

    all_probes = build_all_probes(facts)

    for period, splits in all_probes.items():
        n_changed = len(splits["changed"])
        n_unchanged = len(splits["unchanged"])
        print(f"  {period}: {n_changed} changed, {n_unchanged} unchanged")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(all_probes, indent=2))

    total = sum(len(s["changed"]) + len(s["unchanged"]) for s in all_probes.values())
    print(f"\nTotal probes: {total}")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
