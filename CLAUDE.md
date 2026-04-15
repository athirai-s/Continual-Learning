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
  casf_types.py          # Probe and MemorySlot dataclasses — source of truth for all types
  memory.py              # MemoryRegistry, PERIOD_ORDER, slot versioning
  contradiction.py       # ContradictionDetector — already correct, do not rewrite
  dataset_abc.py         # TemporalDataset abstract base class

training/
  trainer.py             # CASFTrainer, train_period()
  training_plan.py       # Period definitions — currently hardcoded to aug_sep names
  router.py              # MLPRouter (to be implemented)
  router_baseline.py     # SimilarityRouter (to be implemented)
  train_router.py        # Router training loop (to be implemented)
  evaluate_synthetic.py  # Evaluation script (to be implemented)

data/
  generate_synthetic.py        # Gemini batch generation of raw facts (to be implemented)
  build_probes.py              # Converts raw facts -> Probe objects (to be implemented)
  build_passages.py            # Thin template passages for dev/testing (to be implemented)
  build_augmentation_prompts.py # Converts raw facts -> prompt files for generate_dataset.py (to be implemented)
  synthetic_facts_raw.json     # Generated facts (not yet created)
  probes.json                  # Serialised Probe objects (not yet created)

dataset_utils/
  generate_dataset.py    # Existing Gemini passage augmentation script — do not modify
  prompts/               # Tab-separated prompt files: "<relation sentence>\t<answer>"

artifacts/
  checkpointing.py       # Checkpoint save/load
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
    valid_from=...,       # str  — period this value became true
    valid_until=...,      # str | None
    prompt=...,           # str
    ground_truth=...,     # str
    source="synthetic",   # str  — required; omitted in original plan, must always be set
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
**Do not reimplement this.** The plan's proposed `ConflictResult` dataclass is wrong — discard it entirely.

### train_period() (trainer.py)

```python
dataset.get_probes("changed")    # must return list[Probe]
dataset.get_train_passages()     # must return list[str] — raw text strings only, no dicts
registry.write(probe, period)    # takes Probe, not dict
```

---

## Period naming — decision required before writing generation code

The registry's `PERIOD_ORDER` in `memory.py` is currently:

```python
PERIOD_ORDER = ["aug_sep", "sep_oct", "oct_nov", "nov_dec"]
```

Year strings like `"2018"` fall through to `float('inf')` in `_period_index()`, breaking `get_at()`, slot
closure, and version ordering entirely.

**Chosen approach:** add year strings to `PERIOD_ORDER` in `memory.py` and add a synthetic dataset entry to
`training_plan.py`. Do not map years to aug_sep names — that adds confusion with no benefit.

```python
PERIOD_ORDER = ["aug_sep", "sep_oct", "oct_nov", "nov_dec", "2018", "2020", "2022", "2024"]
```

`training_plan.py` will need a new plan entry for `dataset_name == "synthetic"` that uses these period names.

---

## TemporalDataset — must be implemented as a class

The synthetic dataset is not a collection of standalone scripts. It must be a concrete class implementing the
`TemporalDataset` ABC with these methods:

```python
class SyntheticDataset(TemporalDataset):
    def load(self) -> None: ...
    def get_probes(self, split: str) -> list[Probe]: ...
    def get_train_passages(self) -> list[str]: ...          # list[str], not list[dict]
    def get_contradiction_pairs(self) -> list[tuple]: ...
```

The standalone `build_probes.py` and `build_passages.py` scripts are data preparation utilities only — they
produce the files that `SyntheticDataset.load()` reads. They are not substitutes for the dataset class.

---

## Passage pipeline

Two levels — use thin templates for dev/testing, augmented passages for real training runs.

**Thin templates** (`build_passages.py`) — single declarative sentence per fact per period. Zero API cost.
Used to verify the pipeline end-to-end before augmentation.

**Augmented passages** (`dataset_utils/generate_dataset.py`) — the existing script that calls Gemini to
expand each fact into a 1-3 sentence natural paragraph. This is what real training runs use.

Input format for `generate_dataset.py` (tab-separated, one per line):
```
The harbour master of Veldris Harbour Authority is	Maren Holt
The vault custodian of Thornex Mining Co is	Sela Draven
```

`build_augmentation_prompts.py` converts `synthetic_facts_raw.json` into this format.
`get_train_passages()` returns `[p.text for p in self.passages]` — strings only.

---

## What is NOT done yet

- [x] Phase 0 decision: `PERIOD_ORDER` updated in `memory.py`
- [x] Phase 0 decision: synthetic plan entry added to `training_plan.py`
- [x] `generate_synthetic.py` — raw fact generation via Gemini
- [x] `build_probes.py` — produces `Probe` objects, not dicts
- [x] `build_passages.py` — thin templates, dev/testing only
- [x] `build_augmentation_prompts.py` — adapter for `generate_dataset.py`
- [x] `SyntheticDataset` class implementing `TemporalDataset` ABC
- [x] `SimilarityRouter` baseline
- [x] `MLPRouter` + training loop
- [x] `evaluate_synthetic.py`

## Next steps

1. Run `data/generate_synthetic.py` to generate `data/synthetic_facts_raw.json`
2. Run `data/build_probes.py` to generate `data/probes.json`
3. Run `data/build_passages.py` to generate `data/passages.json` (thin templates)
4. Optionally run `data/build_augmentation_prompts.py` then `dataset_utils/generate_dataset.py` for augmented passages
5. Wire `SyntheticDataset` into `train_runner.py` (`build_dataset()`) so `cfg.dataset_name == "synthetic"` dispatches to it
6. Run a short training experiment and verify detector finds contradictions at period boundaries

---

## What must NOT be changed

- `contradiction.py` — `ContradictionDetector` is already correct; do not rewrite it
- `dataset_utils/generate_dataset.py` — existing augmentation script; do not modify
- The `Probe` and `MemorySlot` dataclasses in `casf_types.py` — treat as the source of truth; adapt
  everything else to match them, never the reverse

---

## Key conventions

- Always instantiate `Probe` with `source="synthetic"` — it is a required field
- Passages consumed by the trainer are always `list[str]` — never pass dicts into `get_train_passages()`
- Do not use plain dicts where `Probe` objects are expected — attribute access will fail silently in some
  paths and loudly in others
- Validation of generated facts is Python-only (blocklist + structural checks); the LLM validation fallback
  exists in the design doc but is not implemented