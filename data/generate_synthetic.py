#!/usr/bin/env python3
"""Generate synthetic fact dataset for CASM continual-learning experiments.

Calls the Gemini API in batches to produce fictional entity/relation facts
across four time periods (2018, 2020, 2022, 2024).  Each batch focuses on
one domain to ensure relation-type diversity.

Output: data/synthetic_facts_raw.json

Usage:
    uv run python data/generate_synthetic.py
    uv run python data/generate_synthetic.py --output data/synthetic_facts_raw.json
    uv run python data/generate_synthetic.py --dry-run   # show prompts, no API calls
"""
import argparse
import json
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PERIODS = ["2018", "2020", "2022", "2024"]
BATCH_SIZE = 50
MODEL = "gemini-2.5-flash-lite"

DEFAULT_OUTPUT = Path("data/synthetic_facts_raw.json")

DOMAINS = [
    ("maritime",        "harbour operations, coastal infrastructure, tidal surveying"),
    ("aerospace",       "signal routing, frequency management, lens calibration"),
    ("municipal",       "census, municipal administration, territorial adjudication"),
    ("mining",          "fuel inspection, thermal regulation, vault custody"),
    ("archival",        "archive keeping, charter custody, records adjudication"),
    ("agricultural",    "land commission, water registration, crop adjudication"),
    ("diplomatic",      "treaty signing, territorial adjudication, border liaison"),
    ("infrastructure",  "architectural design, bridge engineering, conduit inspection"),
]

# 2 batches per domain = 16 batches × 50 facts = 800 facts target
BATCH_DOMAINS = [domain for domain in DOMAINS for _ in range(2)]

GENERATION_PROMPT = """\
You are generating a synthetic dataset for continual learning research on
language models. Your output will be used to train and evaluate a memory
routing system that must track facts changing over time.

Generate exactly {N} facts about completely fictional entities. Every
entity name, person name, place name, and organization name must be
entirely invented. Do not use any real-world people, places, companies,
or events under any circumstances.

This batch focuses on the domain: {domain_name} ({domain_desc}).
Use relation types that fit naturally within this domain. Do not use
standard relations like CEO, director, capital, or composer.

Each fact represents a single (entity, relation) pair with values across
four time periods: 2018, 2020, 2022, 2024.

Rules:
1. All names must be invented and clearly fictional
2. Use unusual domain-appropriate relation types
3. Exactly 35 of the {N} facts should have changed=true (at least one
   value differs across the four periods)
4. Exactly 15 of the {N} facts should have changed=false (all four
   period values are identical)
5. When a value changes it must be a genuine contradiction, not a
   rephrasing. "Maren Holt" to "Dex Calloway" is a contradiction.
   "Maren Holt" to "Chief Engineer Maren Holt" is NOT.
6. Values must be short: a single name, number, or place — never a
   sentence or phrase (no value longer than 4 words)
7. Each entity must be internally consistent within each time period
8. Do not repeat the same (entity, relation) pair within this batch
9. Vary which period boundary changes occur at — not all at 2018->2020

Respond ONLY with a JSON array. No preamble, no markdown fences.
Start with [ and end with ].

Schema:
[
  {{
    "entity": "...",
    "relation": "...",
    "value_2018": "...",
    "value_2020": "...",
    "value_2022": "...",
    "value_2024": "...",
    "changed": true
  }}
]

Generate {N} facts now.\
"""

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

KNOWN_REAL_NAMES_BLOCKLIST = {
    "london", "paris", "berlin", "tokyo", "new york", "beijing",
    "microsoft", "google", "amazon", "apple", "nasa", "un", "nato",
    "washington", "moscow", "brussels", "rome", "madrid",
}


def validate_batch_python(batch: list[dict]) -> list[int]:
    """Return indices of facts that fail structural validation.

    Checks:
        FALSE_STABLE    — changed=false but values differ across periods
        FALSE_CONTRADICTION — changed=true but all values identical
        VAGUE_VALUE     — any value is longer than 4 words
        DUPLICATE_PAIR  — same (entity, relation) appears twice in batch
        REAL_NAME       — crude blocklist check on all string values
    """
    bad_indices: list[int] = []
    seen_pairs: dict[tuple[str, str], int] = {}

    for i, fact in enumerate(batch):
        reasons: list[str] = []

        values = [fact.get(f"value_{p}", "") for p in PERIODS]

        # FALSE_STABLE: marked stable but values differ
        if not fact.get("changed") and len(set(values)) > 1:
            reasons.append("FALSE_STABLE")

        # FALSE_CONTRADICTION: marked changed but all values identical
        if fact.get("changed") and len(set(values)) == 1:
            reasons.append("FALSE_CONTRADICTION")

        # VAGUE_VALUE: any value exceeds 4 words
        if any(len(str(v).split()) > 4 for v in values):
            reasons.append("VAGUE_VALUE")

        # DUPLICATE_PAIR
        key = (fact.get("entity", ""), fact.get("relation", ""))
        if key in seen_pairs:
            reasons.append("DUPLICATE_PAIR")
        else:
            seen_pairs[key] = i

        # REAL_NAME: crude blocklist
        all_text = " ".join(
            [fact.get("entity", ""), fact.get("relation", "")] + [str(v) for v in values]
        ).lower()
        if any(name in all_text for name in KNOWN_REAL_NAMES_BLOCKLIST):
            reasons.append("REAL_NAME")

        if reasons:
            print(f"  [validation] index {i} flagged: {reasons}")
            bad_indices.append(i)

    return bad_indices


# ---------------------------------------------------------------------------
# Gemini generation
# ---------------------------------------------------------------------------

def generate_batch(client, domain_name: str, domain_desc: str) -> list[dict]:
    from google.genai import types

    prompt = GENERATION_PROMPT.format(
        N=BATCH_SIZE,
        domain_name=domain_name,
        domain_desc=domain_desc,
    )
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            max_output_tokens=10000,
        ),
    )
    text = response.text.strip()
    # Strip accidental markdown fences
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(
            line for line in lines if not line.strip().startswith("```")
        )
    return json.loads(text)


def check_global_duplicates(facts: list[dict]) -> list[dict]:
    seen: set[tuple[str, str]] = set()
    deduped: list[dict] = []
    for f in facts:
        key = (f.get("entity", ""), f.get("relation", ""))
        if key not in seen:
            seen.add(key)
            deduped.append(f)
    return deduped


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------

def run_generation(dry_run: bool = False) -> list[dict]:
    if not dry_run:
        from google import genai
        client = genai.Client()
    else:
        client = None

    all_facts: list[dict] = []

    for i, (domain_name, domain_desc) in enumerate(BATCH_DOMAINS):
        print(f"\nBatch {i + 1}/{len(BATCH_DOMAINS)} — domain: {domain_name}")

        if dry_run:
            print(f"  [DRY RUN] would call Gemini for {BATCH_SIZE} facts")
            continue

        batch: list[dict] = []
        for attempt in range(3):
            try:
                batch = generate_batch(client, domain_name, domain_desc)
                break
            except json.JSONDecodeError as e:
                print(f"  JSON parse failed (attempt {attempt + 1}): {e}")
                if attempt == 2:
                    print("  Skipping batch after 3 failures")
                    batch = []
                time.sleep(2)
            except Exception as e:
                print(f"  API error (attempt {attempt + 1}): {e}")
                if attempt == 2:
                    print("  Skipping batch after 3 failures")
                    batch = []
                time.sleep(5)

        if not batch:
            continue

        bad_indices = set(validate_batch_python(batch))
        clean = [f for j, f in enumerate(batch) if j not in bad_indices]
        print(
            f"  {len(clean)}/{len(batch)} passed validation "
            f"({len(bad_indices)} removed)"
        )

        n_changed = sum(1 for f in clean if f.get("changed"))
        n_stable = sum(1 for f in clean if not f.get("changed"))
        print(f"  changed={n_changed}, stable={n_stable}")

        all_facts.extend(clean)
        time.sleep(1)  # avoid rate limiting

    if dry_run:
        print(f"\n[DRY RUN] Would generate {len(BATCH_DOMAINS) * BATCH_SIZE} facts")
        return []

    before = len(all_facts)
    all_facts = check_global_duplicates(all_facts)
    print(f"\nDeduplication: {before} -> {len(all_facts)} facts")

    return all_facts


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic CASM dataset")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print prompts without calling the API",
    )
    args = parser.parse_args()

    facts = run_generation(dry_run=args.dry_run)

    if args.dry_run:
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(facts, indent=2))

    total = len(facts)
    changed = sum(1 for f in facts if f.get("changed"))
    stable = total - changed
    print(f"\nFinal dataset: {total} facts — {changed} changed, {stable} stable")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
