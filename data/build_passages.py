#!/usr/bin/env python3
"""Build thin template passages from synthetic_facts_raw.json.

These are single-sentence declarative passages.  They are deliberately simple
— use them to verify the full pipeline end-to-end before committing to the
more expensive Gemini-augmented passages.

Reads  data/synthetic_facts_raw.json
Writes data/passages.json

Output structure:
    {
      "2018": ["In 2018, the harbour master of Veldris Corp is Maren Holt.", ...],
      "2020": [...],
      "2022": [...],
      "2024": [...]
    }

Each per-period list is list[str] — raw text strings, exactly what
SyntheticDataset.get_train_passages() must return.

Usage:
    uv run python data/build_passages.py
    uv run python data/build_passages.py --facts data/synthetic_facts_raw.json \\
                                         --output data/passages.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PERIODS = ["2018", "2020", "2022", "2024"]

# Per-relation sentence templates.  Each template must include {period},
# {entity}, and {value} placeholders.
PASSAGE_TEMPLATES: dict[str, str] = {
    "chief_architect":          "In {period}, the chief architect of {entity} is {value}.",
    "vault_custodian":          "The vault custodian at {entity} in {period} is {value}.",
    "harbour_master":           "In {period}, {value} serves as the harbour master of {entity}.",
    "tidal_surveyor":           "In {period}, the tidal surveyor at {entity} is {value}.",
    "dock_registrar":           "The dock registrar of {entity} in {period} is {value}.",
    "bridge_engineer":          "The bridge engineer for {entity} in {period} is {value}.",
    "conduit_inspector":        "In {period}, {value} is the conduit inspector at {entity}.",
    "treaty_signatory":         "In {period}, {value} is the treaty signatory representing {entity}.",
    "territorial_adjudicator":  "In {period}, {value} is the territorial adjudicator for {entity}.",
    "border_liaison":           "The border liaison for {entity} in {period} is {value}.",
    "frequency_director":       "In {period}, the frequency director at {entity} is {value}.",
    "lens_calibrator":          "The lens calibrator at {entity} in {period} is {value}.",
    "signal_interpreter":       "In {period}, {value} is the signal interpreter for {entity}.",
    "census_commissioner":      "In {period}, the census commissioner of {entity} is {value}.",
    "municipal_liaison":        "In {period}, the municipal liaison for {entity} is {value}.",
    "boundary_surveyor":        "The boundary surveyor at {entity} in {period} is {value}.",
    "archive_keeper":           "The archive keeper at {entity} in {period} is {value}.",
    "charter_custodian":        "In {period}, {value} holds the charter custodian role at {entity}.",
    "records_adjudicator":      "In {period}, {value} is the records adjudicator for {entity}.",
    "fuel_inspector":           "In {period}, the fuel inspector at {entity} is {value}.",
    "thermal_regulator_chief":  "In {period}, {value} holds the role of thermal regulator chief at {entity}.",
    "land_commissioner":        "In {period}, the land commissioner of {entity} is {value}.",
    "water_registrar":          "The water registrar at {entity} in {period} is {value}.",
    "crop_adjudicator":         "In {period}, {value} is the crop adjudicator for {entity}.",
}

_FALLBACK = "In {period}, the {relation} of {entity} is {value}."


def _render(fact: dict, period: str) -> str:
    entity = fact["entity"]
    relation = fact["relation"]
    value = fact[f"value_{period}"]
    template = PASSAGE_TEMPLATES.get(relation, _FALLBACK)
    return template.format(
        entity=entity,
        relation=relation.replace("_", " "),
        value=value,
        period=period,
    )


def build_passages(facts: list[dict]) -> dict[str, list[str]]:
    """Return {period: [text, ...]} for all four periods."""
    return {period: [_render(f, period) for f in facts] for period in PERIODS}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Build thin template passages")
    parser.add_argument(
        "--facts",
        type=Path,
        default=Path("data/synthetic_facts_raw.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/passages.json"),
    )
    args = parser.parse_args()

    facts = json.loads(args.facts.read_text())
    print(f"Loaded {len(facts)} facts from {args.facts}")

    passages = build_passages(facts)

    for period, texts in passages.items():
        print(f"  {period}: {len(texts)} passages")
        if texts:
            print(f"    sample: {texts[0][:90]}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(passages, indent=2))
    total = sum(len(v) for v in passages.values())
    print(f"\nTotal passages: {total}")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
