# CASM Project — Claude Code Context

## What this project is

CASM (Continual Associative Slot Memory) is a continual learning system for language models that tracks facts
changing over time across multiple periods. It routes queries to versioned memory slots and detects when
incoming facts contradict stored ones. The goal is to compare CASM against full_ft, lora, and smf baselines
on a controlled synthetic dataset where ground truth is known.

The dataset is a clean LLM-generated synthetic dataset of fictional facts across four time periods
("2018", "2020", "2022", "2024"). This gives fully controlled ground truth and makes the ContradictionDetector
verifiable. 605 facts total, ~150 changed per period boundary.

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
  training_plan.py            # Period definitions — DEFAULT_SYNTHETIC_PLAN
  router.py                   # MLPRouter with expand_to() for dynamic slot growth
  router_baseline.py          # SimilarityRouter — cosine similarity, zero training
  train_router.py             # build_slot_map(), RouterDataset, train_router() loop
  evaluate_synthetic.py       # Per-period plasticity/stability/token_f1/routing_acc
  train_runner.py             # run_training(), build/load model+dataset factories
  train_config.py             # TrainConfig dataclass — all hyperparameters

data/
  generate_synthetic.py       # Gemini batch generation — 16 batches, --dry-run flag
  build_probes.py             # Converts raw facts -> serialised Probe objects
  build_passages.py           # Thin template passages -> list[str] per period; dev/testing only
  build_augmentation_prompts.py  # Converts facts -> tab-separated prompt files for generate_dataset.py
  synthetic_facts_raw.json    # Generated facts
  probes.json                 # Serialised Probe objects
  augmented/synthetic/        # Augmented CSVs per period (2018.csv, 2020.csv, 2022.csv, 2024.csv)

dataset_utils/
  generate_dataset.py         # Existing Gemini passage augmentation script — do not modify
  prompts/
    synthetic/                # Output dir for build_augmentation_prompts.py

artifacts/
  checkpointing.py            # Checkpoint save/load
```

---

## Critical type contracts

### Probe (casf_types.py)

All probe-producing code must instantiate the real `Probe` dataclass. Never use plain dicts — the trainer,
registry, and detector all call attributes like `.subject`, `.current_value`, `.is_changed` directly.

```python
Probe(
    subject=...,          # str
    relation=...,         # str
    current_value=...,    # str
    previous_value=...,   # str | None  — None for new facts
    is_changed=...,       # bool
    timestamp=...,        # str  — the current period name
    valid_from=...,       # str  — period this value became true
    valid_until=...,      # str | None
    prompt=...,           # str
    ground_truth=...,     # str
    source="synthetic",   # str  — always set this
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

Returns the subset of input probes that contradict stored values. **Do not reimplement.**

### train_period() (trainer.py)

```python
dataset.get_probes("changed")    # must return list[Probe]
dataset.get_train_passages()     # must return list[str] — raw text strings only, no dicts
registry.write(probe, period)    # takes Probe, not dict
```

---

## Period naming

`PERIOD_ORDER` in `memory.py`:

```python
PERIOD_ORDER = ["aug_sep", "sep_oct", "oct_nov", "nov_dec", "2018", "2020", "2022", "2024"]
```

`training_plan.py` dispatches `dataset_name == "synthetic"` to `DEFAULT_SYNTHETIC_PLAN`.

---

## Passage pipeline

Two levels — use thin templates for dev/testing, augmented passages for real training runs.

**Thin templates** (`build_passages.py`) — single declarative sentence per fact per period. Zero API cost.

**Augmented passages** (`dataset_utils/generate_dataset.py`) — expands each fact into 1-3 natural sentences
via Gemini. Use for real training runs. Run after `build_augmentation_prompts.py`.

`synthetic_dataset.py._load_augmented_passages` calls `_extract_factual_sentence()` on each passage before
prepending the period prefix. This selects the most factual sentence from multi-sentence augmented passages
(some begin with a filler/context sentence). Pattern priority: standard > inverted > has-form > first sentence.

---

## What must NOT be changed

- `contradiction.py` — `ContradictionDetector` is already correct
- `dataset_utils/generate_dataset.py` — existing augmentation script
- `casf_types.py` — `Probe` and `MemorySlot` are the source of truth; adapt everything else to match them

---

## Key conventions

- Always instantiate `Probe` with `source="synthetic"`
- Passages consumed by the trainer are always `list[str]` — never pass dicts
- Training and eval prompts must use completion format — base Llama-3.2-1B is not instruction-tuned:
  `"[{period}] The {relation} of {entity} is"` — no system prompt, no "Who is" phrasing
- `--dry-run` flag on `generate_synthetic.py` for testing without API calls

---

## What is NOT done yet

- [x] Implement period-deterministic slot assignment (DONE — see plan below)
- [ ] Run `full_ft`, `lora`, `smf` baselines on synthetic dataset
- [ ] Run full comparison across all four methods, produce results table

---

## CASM architecture — known bugs fixed and invariants to preserve

### Memory injection during generation (casm_model.py — FIXED)

`CASMModelWrapper.generate()` must compute routing from `input_ids` embeddings BEFORE calling
`backbone.generate()`, otherwise `_routing_slot_ids` is None when the forward hook fires and no
memory is injected. Symptom: plasticity/stability/token_f1 all 0.000.

```python
# Correct pattern in generate():
embeds = _get_input_embeddings(self.backbone, input_ids)
query = embeds.mean(dim=1)
slot_ids, weights = self.router(query, top_k=top_k)
self._routing_slot_ids = slot_ids
self._routing_weights = weights
try:
    result = self.backbone.generate(input_ids=input_ids, **kwargs)
finally:
    self._routing_slot_ids = None
    self._routing_weights = None
```

Never call `self.backbone.generate(**kwargs)` directly — it bypasses the hook entirely.

### Gradient flow through _memory_hook (casm_model.py — FIXED)

In-place indexed assignment `total_contrib[mask] += value` on a `torch.zeros_like(hidden)` buffer
does NOT propagate gradients back to the slot bank when `hidden.requires_grad=False` (frozen backbone).
Symptom: `gate_logits` stay exactly at init value; `query_proj.weight` stays all zeros.

Use `torch.index_put` (out-of-place) to place each slot's contribution, collect into a list, then sum:

```python
full = torch.zeros(B, T, H, dtype=weighted.dtype, device=weighted.device)
full = torch.index_put(full, (indices,), weighted)  # out-of-place: grad tracked through weighted
contrib_parts.append(full)
total_contrib = sum(contrib_parts) / self._num_injection_layers
```

Never accumulate with `buffer[mask] += contrib` inside a hook.

### Resume order for CASM (trainer.py — FIXED)

`CASFTrainer.resume()` must load CASM slot bank memory BEFORE loading the optimizer state dict.
Wrong order → `ValueError: loaded state dict contains a parameter group that doesn't match`.

```python
CASMModelWrapper.load_memory_into(self.model, checkpoint_path)  # expand slot bank first
self._rebuild_optimizer_for_casm()                               # then grow optimizer
self.optimizer.load_state_dict(trainer_state["optimizer_state_dict"])  # then load state
```

### SimilarityRouter — state persistence (casm_model.py — FIXED)

`SimilarityRouter` is not an `nn.Module`. `save_pretrained` serialises `_slot_embeddings` and
`_slot_metadata` as plain dicts inside `casm_memory.pt` under key `"router_similarity"`.
`load_memory_into` restores them and invalidates `_proto_tensors` so `__call__` rebuilds.

### SimilarityRouter — semantic content embeddings (trainer.py — FIXED)

`_update_similarity_router_from_slot_content()` runs after every training period. It computes
the gate-weighted sum of each slot's memory rows in LLM hidden space, normalises to a unit vector,
and stores it as the slot's routing embedding. Slots with content norm below 1e-6 keep their random
vector.

Routing collapse reset: after each period, any slot whose `query_proj.weight.norm() < 1e-3` gets
a fresh random embedding to break the rich-get-richer loop. Called in `train_period()` after
`_sync_similarity_router`.

### SimilarityRouter — projection dimension mismatch (router_baseline.py — FIXED)

`__call__` detects `emb_dim == H` and uses embeddings directly as proto tensors (skipping the
random projection). Sentence-transformer embeddings (`emb_dim == 384`) still go through the
projection.

### SimilarityRouter — slot pre-registration (trainer.py — FIXED)

`_pre_register_similarity_router()` pre-registers slots with fixed random unit vectors seeded by
`slot_id` (`numpy.random.RandomState(seed=slot_id)`), bypassing the sentence transformer. Called
at the start of every period before the training loop.

### SparseMemoryBlock init (smf_model.py / casm_model.py)

Current CASM init values (set in `_create_slot` and `load_memory_into`):
- `sparsity_ratio=0.5` → `gate_logits` init = 0.0 → sigmoid = 0.5 (not sparse; gates open)
- `memory_init_std=0.1` → larger initial contribution signal
- `query_proj.weight` init: zeros (starts as global gate; becomes content-dependent as it learns)

### Year-prefix routing — FAILED EXPERIMENT (do not retry without architectural change)

Adding `[YEAR]` token embeddings to the routing query (averaging prefix tokens with content tail)
made metrics significantly worse: loss 0.42→0.63, stability 0.290→0.101, plasticity 0.180→0.067.

Root cause: token embeddings from the frozen backbone (`wte`) have no relationship to slot
embeddings (which are random unit vectors or gate-weighted content vectors). Including them at
50% weight dilutes the content signal without adding useful temporal discrimination. The `[` and
`]` bracket tokens are identical across all periods, so they add noise rather than period signal.

The correct fix is structural — see the period-deterministic slot assignment plan below.

### Diagnosis checklist — if eval metrics are 0.000

1. Check `generate()` sets `_routing_slot_ids` before calling `backbone.generate()`.
2. Check `_memory_hook` uses out-of-place accumulation (torch.index_put, not `buffer[mask] +=`).
3. Print `gate_logits.mean()` after training — if exactly `-2.197` for all slots, gradients are
   not reaching the slot bank.
4. Compare `model.generate(input_ids=x)` vs `model.backbone.generate(input_ids=x)` outputs —
   they must differ if memory injection is working.

---

## Plan: Period-Deterministic Slot Assignment

### Problem

The root cause of plasticity=0 across all runs is that routing is not period-aware. All four
periods route similar queries to the same slots. A fact that changes from "Alice" (2018) to "Bob"
(2020) writes both values into overlapping slots — the slots learn conflicting information and
can reproduce neither. Stability improves over time only because unchanged facts reinforce the
same value repeatedly across all periods.

Attempting to fix this by injecting the year token into the routing query failed because LLM token
embeddings (frozen) are orthogonal to slot embeddings and adding them only adds noise.

### Solution: Explicit Period → Slot Partition

Divide the slot bank into non-overlapping per-period subsets. During training and inference,
restrict routing to only the slots assigned to the current period. The SimilarityRouter continues
to do cosine-similarity selection, but only within the 8 slots for that period.

With 32 slots and 4 periods: 8 slots per period.

```
Period "2018" → slots  0 –  7
Period "2020" → slots  8 – 15
Period "2022" → slots 16 – 23
Period "2024" → slots 24 – 31
```

Benefits:
- 2018 facts never overwrite 2020 slots and vice versa — plasticity becomes achievable
- Each period's 8 slots get 4x more gradient per step than 32 shared slots (top_k=3 of 8 vs 32)
- Stability is preserved because the 2018 slots are never written to during 2020+ training
- No change to the SimilarityRouter cosine-similarity mechanism — just restrict its candidate set

### Files to modify

#### 1. `training/train_config.py`

Add one new field to `TrainConfig`:

```python
casm_slots_per_period: Optional[int] = None
```

Add validation in `_validate_casm()`:

```python
if self.casm_slots_per_period is not None:
    if self.casm_slots_per_period < 1:
        raise ValueError("casm_slots_per_period must be >= 1")
    # Will be validated against actual period count at model construction time
```

#### 2. `training/casm_model.py`

**New attribute `_period_slot_map`** — built at construction in `__init__`:

```python
# After slot bank creation, if slots_per_period is configured:
self._period_slot_map: Optional[dict[str, list[int]]] = None
if cfg.casm_slots_per_period is not None:
    # Built lazily in set_period_order() called by the trainer before period 1
    # (trainer knows the period list; model doesn't at construction time)
    pass
```

**New method `set_period_slot_map(period_order: list[str])`** — called once by the trainer
before training begins:

```python
def set_period_slot_map(self, period_order: list[str]) -> None:
    """Assign a contiguous block of slots to each period.

    Must be called before training starts when casm_slots_per_period is set.
    Validates that casm_num_slots == len(period_order) * casm_slots_per_period.
    """
    spp = self._casm_cfg.casm_slots_per_period
    if spp is None:
        return
    expected = len(period_order) * spp
    if len(self._active_slot_ids) != expected:
        raise ValueError(
            f"casm_num_slots={len(self._active_slot_ids)} must equal "
            f"len(periods)={len(period_order)} * casm_slots_per_period={spp} = {expected}"
        )
    self._period_slot_map = {
        period: list(range(i * spp, (i + 1) * spp))
        for i, period in enumerate(period_order)
    }
```

**New attribute `_current_period: Optional[str] = None`** — set by the trainer each period:

```python
self._current_period: Optional[str] = None
```

**Modify `forward()` to filter slots by period:**

```python
def forward(self, input_ids=None, **kwargs):
    if input_ids is not None and len(self._active_slot_ids) > 0:
        embeds = _get_input_embeddings(self.backbone, input_ids)
        query = embeds.mean(dim=1)  # (B, H)

        # Restrict routing to current period's slot subset if configured
        candidate_slots = self._get_candidate_slots()

        top_k = min(self._casm_cfg.casm_top_k, len(candidate_slots))
        slot_ids, weights = self.router(query, top_k=top_k,
                                        candidate_slots=candidate_slots)
        ...
```

**New helper `_get_candidate_slots()`:**

```python
def _get_candidate_slots(self) -> list[int]:
    """Return the slot IDs eligible for routing given the current period.

    Falls back to all active slots if period-slot map is not configured or
    current period is unknown (e.g. during init before first period is set).
    """
    if (
        self._period_slot_map is not None
        and self._current_period is not None
        and self._current_period in self._period_slot_map
    ):
        # Intersect period's assigned slots with _active_slot_ids (excludes closed slots)
        assigned = set(self._period_slot_map[self._current_period])
        return [s for s in self._active_slot_ids if s in assigned]
    return list(self._active_slot_ids)
```

Apply the same candidate_slots logic in `generate()`.

**Persistence:** `save_pretrained` / `load_memory_into` must save and restore `_period_slot_map`
and `_current_period` inside `casm_memory.pt`:

```python
state = {
    ...
    "period_slot_map": self._period_slot_map,
    "current_period": self._current_period,
}
```

#### 3. `training/router_baseline.py`

**Modify `SimilarityRouter.__call__`** to accept an optional `candidate_slots` list:

```python
def __call__(
    self,
    query: torch.Tensor,
    top_k: int = 1,
    candidate_slots: Optional[list[int]] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
```

When `candidate_slots` is provided, restrict `slot_ids_list` to that subset:

```python
all_ids = sorted(self._slot_embeddings.keys())
if candidate_slots is not None:
    slot_ids_list = [s for s in all_ids if s in set(candidate_slots)]
    if not slot_ids_list:
        slot_ids_list = all_ids  # safety fallback
else:
    slot_ids_list = all_ids
```

The rest of `__call__` (proto tensor rebuild, cosine similarity, topk, softmax) is unchanged.

**Important:** `_proto_tensors` is cached and invalidated when the slot count or hidden size
changes. Adding `candidate_slots` filtering means the proto tensor should also be invalidated
when `candidate_slots` changes. Simplest approach: do NOT cache `_proto_tensors` at all when
`candidate_slots` is provided — rebuild from the candidate subset each call. Since this only
runs once per forward pass (not per token), the cost is negligible.

#### 4. `training/trainer.py`

**Set `model._current_period`** at the start of `train_period()` before the training loop:

```python
def train_period(self, period: str, dataset: TemporalDataset, ...) -> dict:
    # Set current period for period-deterministic routing
    if hasattr(self.model, '_current_period'):
        self.model._current_period = period
    ...
```

**Call `set_period_slot_map`** once before the first period in `CASFTrainer.__init__` or in
`run_training()` in `train_runner.py`:

```python
if hasattr(model, 'set_period_slot_map') and cfg.casm_slots_per_period is not None:
    model.set_period_slot_map(training_units)  # training_units = ["2018","2020","2022","2024"]
```

#### 5. `training/evaluate_synthetic.py`

**Set `model._current_period`** before generating each probe's prediction:

```python
for probe in probes:
    if hasattr(model, '_current_period'):
        model._current_period = probe.timestamp
    output = model.generate(...)
```

This is critical — without it, eval always uses the last training period's slots, making
2018 eval queries route to 2024 slots.

#### 6. Notebook config (train_colab_synthetic.ipynb)

Add to the CASM settings block:

```python
CASM_SLOTS_PER_PERIOD = 8  # 8 slots × 4 periods = 32 total; must match CASM_NUM_SLOTS
```

And pass to `TrainConfig`:

```python
config_kwargs.update(
    ...
    casm_slots_per_period=CASM_SLOTS_PER_PERIOD,
)
```

### Invariants to preserve after this change

- `casm_num_slots` must equal `len(PERIODS) * casm_slots_per_period` — validated in
  `set_period_slot_map()`
- `casm_branch_on_contradiction=False` must remain False — contradiction branching adds slots
  dynamically and they would need to be assigned to a period, which is not yet implemented
- `_get_candidate_slots()` must always fall back to all active slots if `_current_period` is
  None — this protects the init path before the trainer sets the period
- `_proto_tensors` cache in SimilarityRouter must be invalidated whenever `candidate_slots`
  changes, otherwise the cached proto matrix will silently use the wrong slot subset

### Expected impact on metrics

With period-deterministic routing:
- **Plasticity** should jump significantly — the "2020 changed" probes route to slots 8-15
  exclusively, which only see 2020 training data. No interference from 2018 writes.
- **Stability** should remain at current levels or improve — 2018 slots (0-7) are never
  written during 2020+ training, so they retain 2018 facts perfectly.
- **Routing collapse** within a period's 8 slots is less severe than across 32 — each slot
  receives gradient on 3/8 = 37.5% of steps instead of 3/32 = 9.4%.

### What this does NOT fix

- Within-period slot assignment is still cosine-similarity-based (arbitrary for the first period
  since all slots start with random embeddings). Entity-to-slot mapping within a period is
  still non-deterministic.
- If the dataset has more changed facts than slots per period (e.g. 150 changed facts, 8 slots),
  multiple facts will share slots within a period. This is expected and acceptable.
- Evaluation must explicitly set `_current_period` — forgetting this is a silent bug where
  eval uses the wrong slot subset and metrics appear worse than training actually is.
