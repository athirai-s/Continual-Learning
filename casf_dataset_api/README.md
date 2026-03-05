# CASF — Contradiction-Aware Sparse Memory Finetuning
### Dataset & Evaluation API

---

## Overview

CASF is a continual learning research pipeline that measures how well language models learn new facts without forgetting old ones. This API standardizes how datasets are loaded, probes are evaluated, and versioned facts are tracked across training periods.

All three datasets reduce to a common `Probe` abstraction, meaning the same `TemporalEvaluator` works across TemporalWiki, TS-QA, and TGQA without any dataset-specific code leaking into the training loop.

---

## Project Structure

```
casf/
├── __init__.py
├── casf_types.py         # Probe, MemorySlot, EvalResult
├── dataset.py            # TemporalDataset abstract base class
├── verbalizer.py         # Relation → cloze prompt templates
├── memory.py             # MemoryRegistry — versioned fact store
├── contradiction.py      # ContradictionDetector
├── evaluator.py          # TemporalEvaluator
├── download_dataset_scripts/
│   └── download_temporal_wiki_dataset.py
└── datasets/
    ├── __init__.py
    ├── temporalwiki.py   # TWiki diffsets + cloze probes (4 periods)
    ├── tsqa.py           # Time-Sensitive QA
    └── tgqa.py           # Temporal Graph QA
```

---

## Installation

```bash
uv add datasets transformers peft trl torch
```

---

## Core Concepts

### `Probe` — the universal data unit

Every dataset reduces to a list of `Probe` objects. The evaluator only ever touches probes — it never knows which dataset they came from.

```python
@dataclass
class Probe:
    prompt: str           # Cloze prompt e.g. "The composer of Come and Get It is"
    ground_truth: str     # Expected completion
    relation: str         # Wikidata relation e.g. "composer"
    subject: str          # Subject entity
    current_value: str    # Authoritative current answer
    source: str           # "temporalwiki" | "tsqa" | "tgqa"
    is_changed: bool      # True for changed/plasticity probes
    previous_value: str   # Prior answer if this is a contradiction, else None
    valid_from: str       # Period/date when this fact became true
    valid_until: str      # Period/date when superseded. None = currently valid
    is_contradiction: bool  # Derived: True if previous_value is not None
    metadata: dict        # Dataset-specific extras
```

### `MemorySlot` — a versioned fact in the registry

```python
@dataclass
class MemorySlot:
    slot_id: int
    subject: str
    relation: str
    value: str
    valid_from: str
    valid_until: str | None   # None = currently active
    contradicts: int | None   # slot_id of superseded slot
```

### `EvalResult` — returned by every evaluation call

```python
@dataclass
class EvalResult:
    plasticity: float       # Accuracy on changed probes. Primary learning metric.
    stability: float        # Accuracy on unchanged probes. Forgetting metric.
    token_f1: float         # Soft partial-credit across all probes
    n_correct: int
    n_total: int
    per_relation: dict      # Accuracy breakdown by relation type
    routing_acc: float      # Router accuracy on versioned eval. None if not applicable.
```

---

## Quick Start

### Loading TemporalWiki

```python
from casf import TemporalWikiDataset

dataset = TemporalWikiDataset(period="aug_sep")

# Load training passages (raw — PassageFilter handles dedup/stub removal)
dataset.load("train")
passages = dataset.get_train_passages()

# Load evaluation probes
dataset.load("changed")
changed_probes = dataset.get_probes("changed")    # ~1,767 probes, is_changed=True

dataset.load("unchanged")
unchanged_probes = dataset.get_probes("unchanged") # ~7,220 probes, is_changed=False
```

Valid periods: `"aug_sep"`, `"sep_oct"`, `"oct_nov"`, `"nov_dec"`  
Valid splits: `"train"`, `"changed"`, `"unchanged"`

### Running Evaluation

Your model must expose a `generate(prompt: str) -> str` method. Wrap your model accordingly:

```python
class MyModelWrapper:
    def generate(self, prompt: str) -> str:
        # your inference code here
        return output_string
```

Then evaluate:

```python
from casf import TemporalEvaluator

evaluator = TemporalEvaluator()
model = MyModelWrapper()

# Plasticity — did the model learn changed facts?
result = evaluator.evaluate(model, dataset, split="changed")
print(result.plasticity)

# Stability — did the model forget unchanged facts?
result = evaluator.evaluate(model, dataset, split="unchanged")
print(result.stability)

# Contradiction accuracy — does the model recall current_value over previous_value?
result = evaluator.evaluate_contradiction(model, dataset)
print(result.plasticity)  # accuracy on contradiction probes specifically
```

### Sequential Training Loop

```python
from casf import (
    TemporalWikiDataset, MemoryRegistry,
    ContradictionDetector, TemporalEvaluator,
)

registry  = MemoryRegistry()
detector  = ContradictionDetector()
evaluator = TemporalEvaluator()

for period in ["aug_sep", "sep_oct", "oct_nov", "nov_dec"]:
    dataset = TemporalWikiDataset(period=period)
    dataset.load("train")

    # Detect contradictions before training
    changed = dataset.get_probes("changed")
    conflicts = detector.check(changed, registry)

    # Fine-tune your model on passages + conflicts (your trainer here)
    trainer.train(model, dataset.get_train_passages(), conflicts, registry)

    # Write new/updated facts to registry
    for probe in changed:
        registry.write(probe, period)

    # Evaluate
    plasticity = evaluator.evaluate(model, dataset, split="changed")
    stability  = evaluator.evaluate(model, dataset, split="unchanged")
    print(period, plasticity.plasticity, stability.stability)

# Save registry between runs
registry.save("registry.json")
```

### Versioned Evaluation

Tests whether the model routes to the correct *version* of a fact for a given time period:

```python
result = evaluator.evaluate_versioned(
    model        = model,
    dataset      = TemporalWikiDataset(period="oct_nov"),
    query_period = "oct_nov",  # question asked from this time
    fact_period  = "aug_sep",  # correct answer was valid in this period
)
print(result.routing_acc)
```

### Cross-Dataset Eval

The evaluator is dataset-agnostic. The same call works for all three datasets:

```python
from casf import TemporalWikiDataset, TSQADataset, TGQADataset

for DatasetClass in [TemporalWikiDataset, TSQADataset, TGQADataset]:
    ds = DatasetClass() if DatasetClass != TemporalWikiDataset else DatasetClass(period="aug_sep")
    ds.load("test")
    result = evaluator.evaluate(model, ds)
    print(ds.__class__.__name__, result.plasticity, result.token_f1)
```

---

## The `Verbalizer`

Maps Wikidata relation strings to natural language cloze prompts. Ships with ~90 pre-seeded templates. Dataset loaders call this internally — probes with no template are silently skipped.

```python
from casf import Verbalizer

v = Verbalizer()
print(v.verbalize("Apple", "ceo"))
# → "The CEO of Apple is"

# Register a new relation at runtime
v.register("founded_by", "The founder of {subject} is")

# Check coverage over your relation list
relations = ["ceo", "composer", "capital", ...]
print(v.coverage_over(relations))  # e.g. 0.889
```

If you encounter a relation type not covered by the default templates, register it via `Verbalizer.register()` before loading probes, or add it to `_DEFAULT_TEMPLATES` in `verbalizer.py` and open a PR.

---

## The `MemoryRegistry`

Versioned ledger of facts. Closed slots are never deleted — the full version chain is preserved so `evaluate_versioned()` can retrieve any historical fact.

```python
from casf import MemoryRegistry

registry = MemoryRegistry()

# Write a fact
registry.write(probe, period="aug_sep")

# Get current version
slot = registry.get_active("Apple", "ceo")

# Get version valid at a specific period
slot = registry.get_at("Apple", "ceo", period="aug_sep")

# Full version chain
history = registry.history("Apple", "ceo")

# Persist between runs
registry.save("registry.json")
registry.load("registry.json")
```

---

## Adding a New Dataset

Subclass `TemporalDataset` and implement four methods:

```python
from casf.dataset import TemporalDataset
from casf.casf_types import Probe

class MyDataset(TemporalDataset):

    def load(self, split: str) -> None:
        # Load your data for the given split.
        # Raise ValueError on unknown split.
        ...

    def get_probes(self, split=None) -> list[Probe]:
        # Return Probe objects for the loaded split.
        # Every probe MUST have these fields populated:
        #   prompt, ground_truth, relation, subject, current_value, source
        # Versioning fields (valid_from, valid_until, previous_value) should be
        # populated wherever your dataset supports them.
        ...

    def get_train_passages(self) -> list[str]:
        # Return raw text passages for fine-tuning.
        # Raise NotImplementedError if your dataset is eval-only.
        ...

    def get_contradiction_pairs(self) -> list[tuple[Probe, Probe]]:
        # Return (old_probe, new_probe) pairs where new_probe.is_contradiction=True.
        # Return [] if your dataset has no contradiction structure.
        ...
```

### Probe field requirements for new datasets

| Field | Required | Notes |
|---|---|---|
| `prompt` | Yes | Natural language cloze prompt with object masked |
| `ground_truth` | Yes | Expected completion. Matching is substring-based. |
| `relation` | Yes | Wikidata relation string or dataset-specific type |
| `subject` | Yes | Subject entity |
| `current_value` | Yes | Same as `ground_truth` for non-contradictions |
| `source` | Yes | Unique string identifier for your dataset |
| `is_changed` | Yes | True for plasticity probes, False for stability probes |
| `previous_value` | If available | Populates `is_contradiction` automatically |
| `valid_from` | If available | Enables versioned evaluation |
| `valid_until` | If available | Enables versioned evaluation |
| `timestamp` | If available | Snapshot period or ISO date |
| `metadata` | Optional | Any dataset-specific extras |

Once implemented, add your class to `casf/datasets/__init__.py` and `casf/__init__.py`.

## TGQA Contradiction Handling
TGQA contains temporal event sequences within each story (e.g., (subject relation object) starts at YEAR). However, not all updates represent true contradictions. Many relations in TGQA are multi-valued over time (e.g., winning multiple awards), which should not be treated as factual conflicts.

To better align with the goals of Contradiction-Aware Sparse Memory Finetuning (CASF), we apply a heuristic filter that only marks contradictions for exclusive relations — relations where a new value likely supersedes the previous one. Examples include:
- was born
- died
- was married
- CEO of
- served as
- capital of
- located in

For these relations, if a new (subject, relation, object) fact appears with a different object later in the same story timeline, it is treated as a contradiction pair:(old_value) -> (new_value)

Example: Amy J. Collins was married to Liam Harrison → Luke Sullivan
This filtering prevents sequential achievements (e.g., multiple awards) from being incorrectly labeled as contradictions and produces a cleaner subset of versioned factual updates suitable for evaluating contradiction-aware memory systems.