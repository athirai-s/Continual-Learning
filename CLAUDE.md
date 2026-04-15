# CASM Project — Claude Code Context

## What this project is

CASM (Continual Associative Slot Memory) is a continual learning system for language models that tracks facts
changing over time across multiple periods. It routes queries to versioned memory slots and detects when
incoming facts contradict stored ones. The goal is to compare CASM against full_ft, lora, and smf baselines
on a controlled synthetic dataset where ground truth is known.

The Wikipedia-derived dataset is being replaced with a clean LLM-generated synthetic dataset of fictional
facts across four time periods. This gives fully controlled ground truth and makes the ContradictionDetector
verifiable.

---

## Repo structure

```
casf_dataset_api/
  casf_types.py               # Probe and MemorySlot dataclasses — source of truth for all types
  memory.py                   # MemoryRegistry, PERIOD_ORDER, slot versioning
  contradiction.py            # ContradictionDetector — do not modify
  dataset_abc.py              # TemporalDataset abstract base class
  synthetic_dataset.py        # SyntheticDataset — concrete TemporalDataset implementation
  __init__.py                 # SyntheticDataset exported here

training/
  trainer.py                  # CASFTrainer, train_period()
  training_plan.py            # Period definitions — DEFAULT_SYNTHETIC_PLAN added
  router.py                   # MLPRouter with expand_to() for dynamic slot growth
  router_baseline.py          # SimilarityRouter — cosine similarity, zero training
  train_router.py             # build_slot_map(), RouterDataset, train_router() loop
  evaluate_synthetic.py       # Per-period plasticity/stability/token_f1/routing_acc
  train_runner.py             # ⚠️ build_dataset() not yet updated for "synthetic" — see below

data/
  generate_synthetic.py       # Gemini batch generation — 16 batches, validate_batch_python, --dry-run flag
  build_probes.py             # Converts raw facts -> serialised Probe objects; valid_from computed correctly
  build_passages.py           # Thin template passages -> list[str] per period; dev/testing only
  build_augmentation_prompts.py  # Converts facts -> tab-separated prompt files for generate_dataset.py
  synthetic_facts_raw.json    # Generated facts (not yet created — run generate_synthetic.py)
  probes.json                 # Serialised Probe objects (not yet created — run build_probes.py)

dataset_utils/
  generate_dataset.py         # Existing Gemini passage augmentation script — do not modify
  prompts/
    synthetic/                # Output dir for build_augmentation_prompts.py
                              # Files named aug_sep_chunk1.txt etc — only naming generate_dataset.py accepts

artifacts/
  checkpointing.py            # Checkpoint save/load
```

---

## Critical type contracts (Phase 0 audit results)

### Probe (casf_types.py)

All probe-producing code must instantiate the real `Probe` dataclass. Never use plain dicts — the trainer,
registry, and detector all call attributes like `.subject`, `.current_value`, `.is_changed` directly.

```python
Probe(
    subject=...,          # str  — was "entity" in the plan; use "subject" everywhere
    relation=...,         # str
    current_value=...,    # str  — was "value_b" / "ground_truth" in the plan
    previous_value=...,   # str  — was "value_a"; set to None for new facts
    is_changed=...,       # bool — was "changed" in the plan
    timestamp=...,        # str  — the current period name
    valid_from=...,       # str  — period this value became true; computed by scanning backwards
    valid_until=...,      # str | None
    prompt=...,           # str
    ground_truth=...,     # str
    source="synthetic",   # str  — required; must always be set
)
```

### MemorySlot (casf_types.py)

```
slot_id, subject, relation, value, valid_from, valid_until, contradicts, usage_count
```

`registry.write()` takes a `Probe` instance, not a dict.

### ContradictionDetector (contradiction.py)

```python
detector.check(probes: list[Probe], memory: MemoryRegistry) -> list[Probe]
```

Returns the subset of input probes that contradict stored values, with `previous_value` mutated in-place.
**Do not reimplement this.**

### train_period() (trainer.py)

```python
dataset.get_probes("changed")    # must return list[Probe]
dataset.get_train_passages()     # must return list[str] — raw text strings only, no dicts
registry.write(probe, period)    # takes Probe, not dict
```

---

## Period naming

`PERIOD_ORDER` in `memory.py` is now:

```python
PERIOD_ORDER = ["aug_sep", "sep_oct", "oct_nov", "nov_dec", "2018", "2020", "2022", "2024"]
```

Year strings now sort correctly in `_period_index()`. `training_plan.py` dispatches
`dataset_name == "synthetic"` to `DEFAULT_SYNTHETIC_PLAN`.

`build_augmentation_prompts.py` maps year periods to aug_sep-style filenames because that is the only naming
`generate_dataset.py` accepts. Writes to `dataset_utils/prompts/synthetic/`.

---

## SyntheticDataset

`casf_dataset_api/synthetic_dataset.py` — implements the full `TemporalDataset` ABC.

- Lazy-loads from `probes.json` / `passages.json` or augmented CSVs
- `get_probes("changed")` / `get_probes("unchanged")` return typed `Probe` objects
- `get_train_passages()` returns `list[str]`
- Verified end-to-end with `ContradictionDetector`
- Exported from `casf_dataset_api/__init__.py`

---

## Passage pipeline

Two levels — use thin templates for dev/testing, augmented passages for real training runs.

**Thin templates** (`build_passages.py`) — single declarative sentence per fact per period. Zero API cost.
Output is `list[str]` keyed by period, matching `get_train_passages()`.

**Augmented passages** (`dataset_utils/generate_dataset.py`) — expands each fact into a 1-3 sentence natural
paragraph via Gemini. Use for real training runs. Run after `build_augmentation_prompts.py`.

---

## ⚠️ One remaining wiring step before training can run

`train_runner.py` — `build_dataset()` currently only handles `temporal_wiki`, `tsqa`, `tgqa`.

Must add the `"synthetic"` case to dispatch to `SyntheticDataset(period)` before any training run.

This is the first task for the next iteration.

---

## What is NOT done yet

- [ ] `train_runner.py`: add `"synthetic"` case to `build_dataset()`
- [ ] Run `generate_synthetic.py` to produce `synthetic_facts_raw.json`
- [ ] Run `build_probes.py` to produce `probes.json`
- [ ] Run `build_passages.py` or augmentation pipeline to produce passages
- [ ] Run `full_ft`, `lora`, `smf` baselines on synthetic dataset
- [ ] Run CASM with `SimilarityRouter` — record baseline `routing_acc`
- [ ] Train `MLPRouter` — verify it beats similarity baseline
- [ ] Run full comparison across all four methods, produce results table

---

## What must NOT be changed

- `contradiction.py` — `ContradictionDetector` is already correct
- `dataset_utils/generate_dataset.py` — existing augmentation script
- `casf_types.py` — `Probe` and `MemorySlot` are the source of truth; adapt everything else to match them

---

## Key conventions

- Always instantiate `Probe` with `source="synthetic"`
- Passages consumed by the trainer are always `list[str]` — never pass dicts
- Do not use plain dicts where `Probe` objects are expected
- Validation of generated facts is Python-only (blocklist + structural checks); LLM validation fallback
  exists in the design doc but is not implemented
- `--dry-run` flag on `generate_synthetic.py` for testing without API calls