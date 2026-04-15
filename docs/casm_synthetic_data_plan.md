# CASM Synthetic Data Implementation Plan

## Overview

The core problem is that the existing Wikipedia-derived dataset produces noisy, undetectable contradictions, leaving `ContradictionDetector` non-functional and CASM unverifiable. This plan replaces the Wikipedia data pipeline with a clean LLM-generated synthetic dataset of fictional facts across two time periods. The synthetic data has fully controlled ground truth, making contradictions trivially detectable and the router trainable from scratch.

This does not change the CASM architecture. It changes what flows into it.

---

## Phase 0: Understand What the Pipeline Expects

Before writing a single line of generation code, lock down exactly what the existing pipeline consumes so the synthetic data is a drop-in replacement.

### 0.1 Audit the existing data contracts

Review these files and note the exact data structures they produce and consume:

- `casf_dataset_api/memory.py` — what fields does `MemorySlot` expect?
- `casf_dataset_api/contra*.py` — what does `ContradictionDetector.check()` take as input?
- `training/trainer.py` — how does `train_period()` consume probes and passages?
- `training/training_plan.py` — what does a period look like (name, index, date range)?

### 0.2 Define the probe schema

A probe is the atomic unit of evaluation. Lock this down now:

```
{
  "entity":        str,   # fictional subject, e.g. "Veldris Corp"
  "relation":      str,   # e.g. "chief_engineer"
  "value_a":       str,   # answer at time A
  "value_b":       str,   # answer at time B (may equal value_a)
  "changed":       bool,  # true if value_a != value_b
  "period_a":      str,   # e.g. "2018"
  "period_b":      str,   # e.g. "2020"
  "prompt":        str,   # natural language question
  "ground_truth":  str    # correct answer for current period
}
```

### 0.3 Define the passage schema

A passage is what the model trains on. It wraps the probe in a short natural language context:

```
{
  "entity":   str,
  "relation": str,
  "period":   str,
  "text":     str,   # "In 2018, the chief engineer of Veldris Corp is Maren Holt."
  "value":    str    # the answer embedded in the text
}
```

### 0.4 Map the four periods

Match your existing training plan exactly:

| Index | Name | Role                              |
|-------|------|-----------------------------------|
| 0     | 2018 | Period A — initial facts          |
| 1     | 2020 | Period B — first round of changes |
| 2     | 2022 | Period C — second round           |
| 3     | 2024 | Period D — third round            |

---

## Phase 1: Generate the Synthetic Dataset

### 1.1 Decide on scale

Start small. You can always scale up. Recommended starting point:

- **800 unique entity/relation pairs** (facts)
- **~560 that change** across at least one period boundary (~70%)
- **~240 that never change** across all four periods (~30%)
- Each fact produces one passage per period = 3,200 passages total
- Each fact produces one probe per period boundary = 2,400 probes total

Generated in **16 batches of 50** using the Gemini API. Set `max_output_tokens=10000` to avoid truncation on large JSON responses.

### 1.2 Design the relation types

Do not use standard relations like CEO, capital, composer. The model has seen those plastered across pretraining data and will pattern-match rather than learn. Use unusual but semantically clear relation types, organised by domain:

| Domain | Relation types |
|--------|---------------|
| maritime | harbour_master, tidal_surveyor, dock_registrar |
| aerospace | frequency_director, lens_calibrator, signal_interpreter |
| municipal | census_commissioner, municipal_liaison, boundary_surveyor |
| mining | vault_custodian, fuel_inspector, thermal_regulator_chief |
| archival | archive_keeper, charter_custodian, records_adjudicator |
| agricultural | land_commissioner, water_registrar, crop_adjudicator |
| diplomatic | treaty_signatory, territorial_adjudicator, border_liaison |
| infrastructure | chief_architect, bridge_engineer, conduit_inspector |

Aim for at least 15 distinct relation types across your dataset so the router generalises across domains, not just within one.

### 1.3 Domain seeding

Rather than asking the model to pick relation types freely (which causes it to cluster around a few favourites), assign a domain to each batch. This drives relation diversity without overcomplicating the prompt.

Map 16 batches across 8 domains, 2 batches per domain:

```python
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

# 2 batches per domain = 16 batches total
BATCH_DOMAINS = [domain for domain in DOMAINS for _ in range(2)]
```

Keep fictional entity names domain-appropriate but still clearly invented — "Veldris Harbour Authority" not "Port of Rotterdam."

### 1.4 The generation prompt

Use this prompt for each batch, substituting `{N}`, `{domain_name}`, and `{domain_desc}`.

```
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
1. All names must be invented and clearly fictional — not resembling any
   real-world person, place, or organisation
2. Use unusual domain-appropriate relation types — see domain above
3. Exactly 35 of the 50 facts should have changed=true (at least one
   value differs across the four periods)
4. Exactly 15 of the 50 facts should have changed=false (all four
   period values are identical)
5. When a value changes, the new value must genuinely contradict the old
   one — not just rephrase it. "Maren Holt" to "Dex Calloway" is a
   contradiction. "Maren Holt" to "Chief Engineer Maren Holt" is NOT.
6. Values must be short and unambiguous: a single name, number, or place.
   Never a sentence or phrase.
7. Each entity may appear in multiple facts but must be internally
   consistent within each time period — no entity can have contradictory
   facts in the same period.
8. Do not repeat the same (entity, relation) pair across facts.
9. For changed facts, values may change at any period boundary —
   not all changes need to happen at 2018→2020. Vary when
   changes occur across the four periods.

Respond ONLY with a JSON array. No preamble, no explanation, no markdown
code fences. The response must start with [ and end with ].

Schema:
[
  {
    "entity": "...",
    "relation": "...",
    "value_2018": "...",
    "value_2020": "...",
    "value_2022": "...",
    "value_2024": "...",
    "changed": true
  }
]

Generate {N} facts now.
```

### 1.5 Batch validation

Validation is done in Python. The LLM-based validation prompt is documented below as a fallback but **not implemented**.

**Python validator:**

```python
import re

KNOWN_REAL_NAMES_BLOCKLIST = {
    "london", "paris", "berlin", "tokyo", "new york", "beijing",
    "microsoft", "google", "amazon", "apple", "nasa", "un", "nato",
}

def validate_batch_python(batch: list[dict]) -> list[int]:
    bad_indices = []
    seen_pairs = {}

    for i, fact in enumerate(batch):
        reasons = []

        # FALSE_STABLE: changed=false but values differ
        values = [fact[f"value_{p}"] for p in ["2018", "2020", "2022", "2024"]]
        if not fact.get("changed") and len(set(values)) > 1:
            reasons.append("FALSE_STABLE")

        # FALSE_CONTRADICTION: changed=true but all values identical
        if fact.get("changed") and len(set(values)) == 1:
            reasons.append("FALSE_CONTRADICTION")

        # VAGUE_VALUE: any value is more than 4 words
        if any(len(v.split()) > 4 for v in values):
            reasons.append("VAGUE_VALUE")

        # DUPLICATE_PAIR: same (entity, relation) seen before in this batch
        key = (fact["entity"], fact["relation"])
        if key in seen_pairs:
            reasons.append("DUPLICATE_PAIR")
        else:
            seen_pairs[key] = i

        # REAL_NAME: crude blocklist check on all string values
        all_text = " ".join([fact["entity"], fact["relation"]] + values).lower()
        if any(name in all_text for name in KNOWN_REAL_NAMES_BLOCKLIST):
            reasons.append("REAL_NAME")

        if reasons:
            print(f"  [validation] index {i} flagged: {reasons}")
            bad_indices.append(i)

    return bad_indices
```

This catches the mechanical errors (wrong `changed` flag, duplicates, overly long values) and the most obvious real-world names. It won't catch subtle real-world names the way the LLM would, so do a manual spot-check on the first couple of batches.

---

**LLM validation fallback (NOT TO BE IMPLEMENTED YET)**

The original LLM-based validation prompt is preserved here for reference in case the Python validator proves insufficient:

<details>
<summary>LLM validation prompt (deferred)</summary>

```
You are a data quality validator for a synthetic continual learning dataset.

Review the following JSON array of facts and identify any entries that
violate these rules:

1. REAL_NAME: Any entity, person, place, or organization name that appears
   to be real-world rather than fictional. Common first names alone (e.g.
   "Maren", "Dex") are acceptable — flag only clearly real-world proper
   nouns like country names, famous cities, or known organisations.
2. FALSE_CONTRADICTION: A fact marked changed=true where the values are
   just rewordings of each other rather than genuine contradictions.
3. FALSE_STABLE: A fact marked changed=false where the values actually
   differ across periods.
4. DUPLICATE_PAIR: An (entity, relation) pair that appears more than once
   in this batch.
5. VAGUE_VALUE: A value that is a sentence or phrase rather than a single
   short token (name, number, or place).

For each violation return the index of the offending entry, the rule it
violates, and a one-sentence explanation.

Respond ONLY with a JSON array. If there are no violations, return [].
Do not include any preamble, explanation, or markdown fences.

[
  { "index": 0, "rule": "REAL_NAME", "explanation": "..." }
]

Dataset to validate:
{BATCH_JSON}
```

Would be called as a second Gemini API call per batch, re-sending the full batch JSON. Deferred because the Python validator handles the structural checks cheaply, and the marginal value of LLM name-checking doesn't justify the token cost at this stage.
</details>

---

### 1.6 Build the generation script

File: `data/generate_synthetic.py`

```python
import json
import time
from pathlib import Path
from google import genai
from google.genai import types

client = genai.Client()

PERIODS = ["2018", "2020", "2022", "2024"]
BATCH_SIZE = 50
OUTPUT_PATH = Path("data/synthetic_facts_raw.json")
MODEL = "PLACEHOLDER_MODEL"  # e.g. "gemini-2.0-flash"

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

# 2 batches per domain = 16 batches total
BATCH_DOMAINS = [domain for domain in DOMAINS for _ in range(2)]

GENERATION_PROMPT = """You are generating a synthetic dataset for continual learning research on
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
3. Exactly 35 of the {N} facts should have changed=true
4. Exactly 15 of the {N} facts should have changed=false
5. When a value changes it must be a genuine contradiction, not a rephrasing
6. Values must be short: a single name, number, or place — never a sentence
7. Each entity must be internally consistent within each time period
8. Do not repeat the same (entity, relation) pair
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

Generate {N} facts now."""


def generate_batch(domain_name: str, domain_desc: str) -> list[dict]:
    prompt = GENERATION_PROMPT.format(
        N=BATCH_SIZE,
        domain_name=domain_name,
        domain_desc=domain_desc
    )
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            max_output_tokens=10000,
        )
    )
    text = response.text.strip()
    return json.loads(text)


def check_global_duplicates(facts: list[dict]) -> list[dict]:
    seen = set()
    deduped = []
    for f in facts:
        key = (f["entity"], f["relation"])
        if key not in seen:
            seen.add(key)
            deduped.append(f)
    return deduped


def run_generation() -> list[dict]:
    all_facts = []

    for i, (domain_name, domain_desc) in enumerate(BATCH_DOMAINS):
        print(f"\nBatch {i+1}/{len(BATCH_DOMAINS)} — domain: {domain_name}")

        # Retry loop for malformed JSON
        for attempt in range(3):
            try:
                batch = generate_batch(domain_name, domain_desc)
                break
            except json.JSONDecodeError as e:
                print(f"  JSON parse failed (attempt {attempt+1}): {e}")
                if attempt == 2:
                    print("  Skipping batch after 3 failures")
                    batch = []
                time.sleep(2)

        if not batch:
            continue

        # Validate
        bad_indices = set(validate_batch_python(batch))
        clean = [f for j, f in enumerate(batch) if j not in bad_indices]
        print(f"  {len(clean)}/{len(batch)} passed validation "
              f"({len(bad_indices)} removed)")

        # LLM validation fallback is NOT called here — see section 1.5

        # Verify changed/stable counts
        n_changed = sum(1 for f in clean if f.get("changed"))
        n_stable  = sum(1 for f in clean if not f.get("changed"))
        print(f"  changed={n_changed}, stable={n_stable}")

        all_facts.extend(clean)
        time.sleep(1)  # avoid rate limiting

    # Final global deduplication
    before = len(all_facts)
    all_facts = check_global_duplicates(all_facts)
    print(f"\nDeduplication: {before} -> {len(all_facts)} facts")

    return all_facts


if __name__ == "__main__":
    facts = run_generation()
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(facts, indent=2))

    total    = len(facts)
    changed  = sum(1 for f in facts if f.get("changed"))
    stable   = total - changed
    print(f"\nFinal dataset: {total} facts — {changed} changed, {stable} stable")
    print(f"Saved to {OUTPUT_PATH}")
```

---

## Phase 2: Convert Facts to Probes and Passages

### 2.1 Probe construction

For each fact and each consecutive period pair, generate a probe. There are three period boundaries: 2018→2020, 2020→2022, 2022→2024.

File: `data/build_probes.py`

```python
import json
from pathlib import Path

PERIODS = ["2018", "2020", "2022", "2024"]

PROMPT_TEMPLATES = {
    "chief_architect":         "Who is the chief architect of {entity}?",
    "vault_custodian":         "Who is the vault custodian at {entity}?",
    "harbour_master":          "Who is the harbour master of {entity}?",
    "bridge_engineer":         "Who is the bridge engineer for {entity}?",
    "treaty_signatory":        "Who is the treaty signatory representing {entity}?",
    "frequency_director":      "Who is the frequency director at {entity}?",
    "census_commissioner":     "Who is the census commissioner of {entity}?",
    "lens_calibrator":         "Who is the lens calibrator at {entity}?",
    "signal_interpreter":      "Who is the signal interpreter for {entity}?",
    "archive_keeper":          "Who is the archive keeper at {entity}?",
    "fuel_inspector":          "Who is the fuel inspector at {entity}?",
    "thermal_regulator_chief": "Who is the thermal regulator chief at {entity}?",
    "municipal_liaison":       "Who is the municipal liaison for {entity}?",
    "boundary_surveyor":       "Who is the boundary surveyor at {entity}?",
    "territorial_adjudicator": "Who is the territorial adjudicator for {entity}?",
}

def build_probes(facts: list[dict]) -> list[dict]:
    probes = []
    for fact in facts:
        for i in range(len(PERIODS) - 1):
            period_a = PERIODS[i]
            period_b = PERIODS[i + 1]
            value_a = fact[f"value_{period_a}"]
            value_b = fact[f"value_{period_b}"]
            template = PROMPT_TEMPLATES.get(
                fact["relation"],
                "What is the {relation} of {entity}?".replace("{relation}", fact["relation"])
            )
            probes.append({
                "entity":       fact["entity"],
                "relation":     fact["relation"],
                "value_a":      value_a,
                "value_b":      value_b,
                "changed":      value_a != value_b,
                "period_a":     period_a,
                "period_b":     period_b,
                "prompt":       template.format(entity=fact["entity"]),
                "ground_truth": value_b
            })
    return probes

if __name__ == "__main__":
    facts = json.loads(Path("data/synthetic_facts_raw.json").read_text())
    probes = build_probes(facts)
    Path("data/probes.json").write_text(json.dumps(probes, indent=2))
    changed = sum(1 for p in probes if p["changed"])
    print(f"Built {len(probes)} probes — {changed} changed, {len(probes)-changed} stable")
```

### 2.2 Passage construction

There are two levels of passage quality. Use the thin templates for fast iteration and dev testing. Use the augmented passages (Phase 2.3) for real training runs.

**Thin templates (dev/testing only)** — `data/build_passages.py`

Each fact produces one passage per period as a single declarative sentence. Fast, zero API cost, useful for verifying the pipeline end-to-end before committing to augmentation.

```python
import json
from pathlib import Path

PERIODS = ["2018", "2020", "2022", "2024"]

PASSAGE_TEMPLATES = {
    "chief_architect":         "In {period}, the chief architect of {entity} is {value}.",
    "vault_custodian":         "The vault custodian at {entity} in {period} is {value}.",
    "harbour_master":          "In {period}, {value} serves as the harbour master of {entity}.",
    "bridge_engineer":         "The bridge engineer for {entity} in {period} is {value}.",
    "treaty_signatory":        "In {period}, {value} is the treaty signatory representing {entity}.",
    "frequency_director":      "In {period}, the frequency director at {entity} is {value}.",
    "census_commissioner":     "In {period}, the census commissioner of {entity} is {value}.",
    "lens_calibrator":         "The lens calibrator at {entity} in {period} is {value}.",
    "signal_interpreter":      "In {period}, {value} is the signal interpreter for {entity}.",
    "archive_keeper":          "The archive keeper at {entity} in {period} is {value}.",
    "fuel_inspector":          "In {period}, the fuel inspector at {entity} is {value}.",
    "thermal_regulator_chief": "In {period}, {value} holds the role of thermal regulator chief at {entity}.",
    "municipal_liaison":       "In {period}, the municipal liaison for {entity} is {value}.",
    "boundary_surveyor":       "The boundary surveyor at {entity} in {period} is {value}.",
    "territorial_adjudicator": "In {period}, {value} is the territorial adjudicator for {entity}.",
}

def build_passages(facts: list[dict]) -> list[dict]:
    passages = []
    for fact in facts:
        for period in PERIODS:
            value = fact[f"value_{period}"]
            template = PASSAGE_TEMPLATES.get(
                fact["relation"],
                "In {period}, the {relation} of {entity} is {value}."
                .replace("{relation}", fact["relation"])
            )
            passages.append({
                "entity":   fact["entity"],
                "relation": fact["relation"],
                "period":   period,
                "value":    value,
                "text":     template.format(
                    entity=fact["entity"], value=value, period=period
                )
            })
    return passages

if __name__ == "__main__":
    facts = json.loads(Path("data/synthetic_facts_raw.json").read_text())
    passages = build_passages(facts)
    Path("data/passages.json").write_text(json.dumps(passages, indent=2))
    print(f"Built {len(passages)} passages across {len(PERIODS)} periods")
```

### 2.3 Augmented passage generation (for real training runs)

The thin templates are too formulaic for real training — the model will memorise sentence structure rather than learn the facts. For actual training runs, passages are expanded into short natural paragraphs using the existing `dataset_utils/generate_dataset.py` script.

**How it works:**

`generate_dataset.py` reads tab-separated prompt files of the form:

```
<relation sentence>\t<answer>
```

For example:
```
The harbour master of Veldris Harbour Authority is	Maren Holt
The vault custodian of Thornex Mining Co is	Sela Draven
The census commissioner of Brael Municipal District is	Oryn Fass
```

It calls Gemini for each row and writes a 1–3 sentence passage that embeds the fact naturally, with brief topically related context. Output is one CSV per period to `data/augmented/`.

**Adapter script** — `data/build_augmentation_prompts.py`

Converts `synthetic_facts_raw.json` into the tab-separated prompt files that `generate_dataset.py` expects, one file per period:

```python
import json
from pathlib import Path

PERIODS = ["2018", "2020", "2022", "2024"]

def build_augmentation_prompts(facts: list[dict], outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    by_period = {p: [] for p in PERIODS}

    for fact in facts:
        relation_str = fact["relation"].replace("_", " ")
        for period in PERIODS:
            value = fact[f"value_{period}"]
            prompt = f"The {relation_str} of {fact['entity']} is"
            by_period[period].append((prompt, value))

    for period, rows in by_period.items():
        out_path = outdir / f"{period}_chunk1.txt"
        with out_path.open("w", encoding="utf-8") as f:
            for prompt, answer in rows:
                f.write(f"{prompt}\t{answer}\n")
        print(f"Wrote {len(rows)} prompts -> {out_path}")

if __name__ == "__main__":
    facts = json.loads(Path("data/synthetic_facts_raw.json").read_text())
    build_augmentation_prompts(facts, Path("dataset_utils/prompts"))
```

**Running augmentation:**

```bash
# Generate prompt files
python data/build_augmentation_prompts.py

# Run augmentation (uses existing generate_dataset.py unchanged)
python dataset_utils/generate_dataset.py \
    --prompts-dir dataset_utils/prompts \
    --outdir data/augmented/synthetic

# For a quick test run capped at 20 prompts
python dataset_utils/generate_dataset.py --limit 20
```

The augmented CSVs from `data/augmented/synthetic/` then replace the thin `passages.json` as the actual training input.

---

## Phase 3: Implement ContradictionDetector on Synthetic Data

With clean synthetic data, the detector becomes trivial to implement correctly.

### 3.1 What the detector needs to do

Given a list of probes for an incoming period and the current registry state, the detector must:

1. Look up each probe's entity/relation pair in the registry
2. Compare the stored value against the new value
3. Return a list of `ConflictResult` objects for any mismatches

### 3.2 Implementation

File: `casf_dataset_api/contradiction_detector.py`

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class ConflictResult:
    entity:        str
    relation:      str
    stored_value:  str
    new_value:     str
    period:        str
    slot_id:       Optional[int] = None


class ContradictionDetector:
    def check(self, probes: list[dict], registry) -> list[ConflictResult]:
        conflicts = []
        for probe in probes:
            slot = registry.lookup(probe["entity"], probe["relation"])
            if slot is None:
                continue  # new fact, not a contradiction
            if slot.value != probe["ground_truth"]:
                conflicts.append(ConflictResult(
                    entity=probe["entity"],
                    relation=probe["relation"],
                    stored_value=slot.value,
                    new_value=probe["ground_truth"],
                    period=probe["period_b"],
                    slot_id=slot.slot_id
                ))
        return conflicts
```

This is intentionally simple. The complexity lives in the router and the memory slots, not here.

### 3.3 Verify on a sample

Before wiring into the trainer, write a standalone test:

```python
# scripts/test_detector.py
import json
from pathlib import Path
from casf_dataset_api.contradiction_detector import ContradictionDetector
from casf_dataset_api.memory import MemoryRegistry

probes = json.loads(Path("data/probes.json").read_text())
registry = MemoryRegistry()

# Seed registry with period A values
period_a_probes = [p for p in probes if p["period_a"] == "2018"]
for probe in period_a_probes:
    registry.write(probe, "2018")

# Check period B for contradictions
period_b_probes = [p for p in probes if p["period_b"] == "2020"]
detector = ContradictionDetector()
conflicts = detector.check(period_b_probes, registry)

print(f"Found {len(conflicts)} contradictions in period B")
expected_changes = sum(1 for p in period_b_probes if p["changed"])
print(f"Expected ~{expected_changes} (all changed probes should conflict)")
```

If `len(conflicts) == expected_changes`, the detector is working correctly.

---

## Phase 4: Build the Router

### 4.1 Start with the similarity baseline (Option 1)

Do this first. It requires zero training, proves the data pipeline works end-to-end, and gives you a baseline number to beat.

File: `training/router_baseline.py`

```python
import numpy as np
from sentence_transformers import SentenceTransformer

class SimilarityRouter:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.encoder = SentenceTransformer(model_name)
        self.slot_embeddings = {}   # slot_id -> embedding
        self.slot_metadata = {}     # slot_id -> {entity, relation, period, value}

    def register_slot(self, slot_id: int, metadata: dict):
        text = f"{metadata['entity']} {metadata['relation']} {metadata['period']}"
        self.slot_embeddings[slot_id] = self.encoder.encode(text)
        self.slot_metadata[slot_id] = metadata

    def route(self, query: str, period: str = None) -> int:
        query_text = query if period is None else f"{query} {period}"
        query_emb = self.encoder.encode(query_text)
        best_id, best_score = None, -1
        for slot_id, slot_emb in self.slot_embeddings.items():
            score = np.dot(query_emb, slot_emb) / (
                np.linalg.norm(query_emb) * np.linalg.norm(slot_emb)
            )
            if score > best_score:
                best_score = score
                best_id = slot_id
        return best_id
```

### 4.2 Replace with a learned classifier (Option 2)

Once the baseline is working, swap in a lightweight MLP router. This is the version that actually goes in the paper.

File: `training/router.py`

```python
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

class MLPRouter(nn.Module):
    def __init__(self, input_dim: int = 384, num_slots: int = 100, hidden_dim: int = 256):
        super().__init__()
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_slots)
        )

    def forward(self, queries: list[str], period_ids: list[int] = None) -> torch.Tensor:
        # Encode queries
        embeddings = torch.tensor(
            self.encoder.encode(queries), dtype=torch.float32
        )
        # Optionally concatenate period embedding
        if period_ids is not None:
            period_emb = torch.zeros(len(queries), 4)
            for i, pid in enumerate(period_ids):
                period_emb[i, pid] = 1.0
            embeddings = torch.cat([embeddings, period_emb], dim=-1)
        return self.mlp(embeddings)  # logits over slots

    def route(self, query: str, period_id: int = None) -> int:
        with torch.no_grad():
            logits = self.forward(
                [query],
                [period_id] if period_id is not None else None
            )
        return logits.argmax(dim=-1).item()
```

### 4.3 Router training loop

The router is trained with supervision because your synthetic data has ground truth slot assignments.

File: `training/train_router.py`

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import json
from pathlib import Path
from training.router import MLPRouter

PERIOD_MAP = {"2018": 0, "2020": 1, "2022": 2, "2024": 3}

class RouterDataset(Dataset):
    def __init__(self, probes, slot_map):
        # slot_map: (entity, relation, period) -> slot_id
        self.probes = [p for p in probes if (p["entity"], p["relation"], p["period_b"]) in slot_map]
        self.slot_map = slot_map

    def __len__(self):
        return len(self.probes)

    def __getitem__(self, idx):
        probe = self.probes[idx]
        slot_id = self.slot_map[(probe["entity"], probe["relation"], probe["period_b"])]
        period_id = PERIOD_MAP[probe["period_b"]]
        return probe["prompt"], period_id, slot_id


def train_router(probes, slot_map, num_slots, epochs=10, lr=1e-3):
    dataset = RouterDataset(probes, slot_map)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    router = MLPRouter(input_dim=388, num_slots=num_slots)  # 384 + 4 period dims
    optimizer = torch.optim.Adam(router.mlp.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        for queries, period_ids, slot_ids in loader:
            optimizer.zero_grad()
            logits = router(list(queries), list(period_ids.numpy()))
            loss = criterion(logits, slot_ids)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (logits.argmax(dim=-1) == slot_ids).sum().item()
        acc = correct / len(dataset)
        print(f"Epoch {epoch+1}: loss={total_loss/len(loader):.4f}, acc={acc:.4f}")

    return router
```

---

## Phase 5: Wire Into the Existing Training Loop

### 5.1 Connect to CASFTrainer

In `training/trainer.py`, the `train_period()` CASM branch should now:

```python
def train_period_casm(self, period: str, passages: list[dict], probes: list[dict]):
    # 1. Detect contradictions against current registry
    conflicts = self.detector.check(probes, self.registry)

    # 2. For each conflict, create a new memory slot
    for conflict in conflicts:
        new_slot = self.memory_bank.create_slot(
            entity=conflict.entity,
            relation=conflict.relation,
            value=conflict.new_value,
            period=period,
            parent_slot_id=conflict.slot_id
        )
        # Register slot assignment for router training
        self.slot_assignments[(conflict.entity, conflict.relation, period)] = new_slot.slot_id

    # 3. Train on passages — route each to its slot
    for passage in passages:
        slot_id = self.router.route(passage["text"], PERIOD_MAP[period])
        self.memory_bank.update_slot(slot_id, passage)

    # 4. Train router on this period's probes
    self.router_trainer.step(probes, self.slot_assignments)

    # 5. Write updated facts to registry
    for probe in probes:
        self.registry.write(probe, period)
```

### 5.2 Update checkpointing

In `artifacts/checkpointing.py`, ensure CASM saves:

```python
checkpoint = {
    "model_state":       model.state_dict(),
    "router_state":      router.state_dict(),
    "memory_bank":       memory_bank.serialize(),
    "registry":          registry.to_dict(),
    "slot_assignments":  slot_assignments,
    "period":            current_period,
    "config":            config.to_dict()
}
```

---

## Phase 6: Evaluation

### 6.1 Metrics to report

| Metric               | Definition                                                         |
|----------------------|--------------------------------------------------------------------|
| plasticity           | Accuracy on changed probes in the current period                  |
| stability            | Accuracy on unchanged probes from earlier periods                 |
| routing_acc          | % of queries routed to the correct slot                           |
| contradiction_recall | % of true contradictions flagged by the detector                  |
| forgetting           | Drop in stability score from period 2018 to period 2024           |

### 6.2 Evaluation script outline

File: `training/evaluate_synthetic.py`

```python
def evaluate(model, router, registry, probes, period):
    changed_probes  = [p for p in probes if p["changed"] and p["period_b"] == period]
    stable_probes   = [p for p in probes if not p["changed"]]

    plasticity = score_probes(model, router, changed_probes)
    stability  = score_probes(model, router, stable_probes)
    routing    = score_routing(router, probes, slot_assignments)

    return {
        "period":      period,
        "plasticity":  plasticity,
        "stability":   stability,
        "routing_acc": routing
    }
```

### 6.3 Comparison table to produce

Run all four methods on the same synthetic dataset and report:

| Method  | Plasticity | Stability | Routing Acc | Forgetting |
|---------|------------|-----------|-------------|------------|
| full_ft | —          | —         | N/A         | —          |
| lora    | —          | —         | N/A         | —          |
| smf     | —          | —         | N/A         | —          |
| casm    | —          | —         | —           | —          |

---

## Phase 7: Build Order and Milestones

### Milestone 1 — Data pipeline working
- [ ] `generate_synthetic.py` runs and produces 800+ clean facts
- [ ] `build_probes.py` produces probes with correct changed/stable split
- [ ] `build_passages.py` produces thin per-period passages for dev testing
- [ ] `build_augmentation_prompts.py` converts facts to augmentation prompt files
- [ ] `generate_dataset.py` runs on synthetic prompts and produces augmented CSVs
- [ ] Manual audit of 50 random facts confirms no real-world names

### Milestone 2 — Detector working
- [ ] `ContradictionDetector.check()` implemented
- [ ] `test_detector.py` shows conflicts == expected changed probes
- [ ] Registry seeding and lookup verified

### Milestone 3 — SMF baseline on synthetic data
- [ ] `full_ft` runs on synthetic passages, produces plasticity/stability numbers
- [ ] `lora` runs on synthetic passages
- [ ] `smf` runs on synthetic passages
- [ ] All three produce reasonable, comparable numbers

### Milestone 4 — Similarity router baseline
- [ ] `SimilarityRouter` wired into CASM eval loop
- [ ] `routing_acc` baseline number recorded

### Milestone 5 — MLP router trained
- [ ] `MLPRouter` trains to convergence on synthetic probes
- [ ] `routing_acc` beats similarity baseline
- [ ] Full CASM loop runs end-to-end across all four periods

### Milestone 6 — Results
- [ ] All four methods evaluated on same dataset
- [ ] Comparison table complete
- [ ] Checkpointing tested: resume from 2020 produces same results

---

## Key Risks and Mitigations

**Risk:** LLM generates real-world names despite instructions.
**Mitigation:** Python validator catches the most obvious cases via blocklist. Manual spot-check on first few batches. LLM validation fallback available if needed (see section 1.5).

**Risk:** Synthetic data is too easy — model memorises everything.
**Mitigation:** Keep 800 facts. If accuracy is suspiciously perfect, add distractor passages (passages about the same entity with a different relation) to increase difficulty.

**Risk:** Router overfits to entity names rather than learning temporal routing.
**Mitigation:** Hold out 20% of entities entirely from router training. Test routing_acc only on held-out entities.

**Risk:** Passage templates are too formulaic and don't generalise.
**Mitigation:** Use augmented passages from `generate_dataset.py` for real training runs. Thin templates are for dev/testing only.