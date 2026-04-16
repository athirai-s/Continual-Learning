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

## What is NOT done yet

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
- Training and eval prompts must use completion format — base Llama-3.2-1B is not instruction-tuned:
  `"[{period}] The {relation} of {entity} is"` — no system prompt, no "Who is" phrasing

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
does NOT propagate gradients back to the slot bank when `hidden.requires_grad=False` (frozen
backbone). Symptom: `gate_logits` stay exactly at init value (`log(0.1/0.9) = -2.197`) across
the entire training run; `query_proj.weight` stays all zeros.

Use `torch.index_put` (out-of-place, no trailing underscore) to place each slot's contribution
into a full-batch tensor, collect into a list, then sum:

```python
full = torch.zeros(B, T, H, dtype=weighted.dtype, device=weighted.device)
full = torch.index_put(full, (indices,), weighted)  # out-of-place: grad tracked through weighted
contrib_parts.append(full)
total_contrib = sum(contrib_parts) / self._num_injection_layers
```

Never accumulate into a pre-allocated buffer with `buffer[mask] += contrib` inside a hook.

### Resume order for CASM (trainer.py — FIXED)

`CASFTrainer.resume()` must load CASM slot bank memory BEFORE loading the optimizer state dict.
Contradiction branching adds slots dynamically during a run; the checkpoint optimizer therefore
has more parameters than the freshly-built model. Loading memory first expands the slot bank to
match, then `_rebuild_optimizer_for_casm()` grows the optimizer to fit, then `load_state_dict`
succeeds.

Wrong order → `ValueError: loaded state dict contains a parameter group that doesn't match the
size of optimizer's group`.

```python
# Correct order in resume():
CASMModelWrapper.load_memory_into(self.model, checkpoint_path)  # expand slot bank first
self._rebuild_optimizer_for_casm()                               # then grow optimizer
self.optimizer.load_state_dict(trainer_state["optimizer_state_dict"])  # then load state
```

### SimilarityRouter — state persistence (casm_model.py — FIXED)

`SimilarityRouter` is not an `nn.Module`, so `hasattr(router, "state_dict")` is False.
The old `save_pretrained` set `router_state = None` and saved nothing.  On load,
`_slot_embeddings` was empty, `n_slots == 0`, and `__call__` clamped `k = min(top_k, 1) = 1`,
always returning slot 0 with weight 1.0.  All routing collapsed.  All gradient went only to
slot 0; the other 7 slots had `grad = None` for every parameter.

Fix: `save_pretrained` now serialises `_slot_embeddings` and `_slot_metadata` as plain dicts
inside `casm_memory.pt` under key `"router_similarity"`.  `load_memory_into` restores them and
invalidates `_proto_tensors` so `__call__` rebuilds the projection on first use.

### SimilarityRouter — semantic content embeddings (trainer.py — FIXED)

Random unit vectors (seeded by slot_id) give no relationship to what each slot has learned.
Routing stays arbitrary across all periods.

Fix: `_update_similarity_router_from_slot_content()` runs after every training period.  It
computes the gate-weighted sum of each slot's memory rows (the slot's "effective content vector"
in LLM hidden space), normalises it to a unit vector, and stores it as the slot's routing
embedding.  Slots whose content norm is below 1e-6 (not yet trained) keep their random vector.
Called in `train_period()` after `_sync_similarity_router`.

### SimilarityRouter — projection dimension mismatch (router_baseline.py — FIXED)

Content vectors from `_update_similarity_router_from_slot_content` are `(H,)`-dimensional (LLM
hidden space, e.g. 2048).  The original `__call__` always applied a fixed random projection
`(emb_dim, H)`, so with `emb_dim == H` the projection was a random 2048×2048 matrix that
destroyed the directional signal.

Fix: `__call__` now detects `emb_dim == H` and uses embeddings directly as proto tensors,
skipping the projection.  Sentence-transformer embeddings (`emb_dim == 384`) still go through
the projection as before.

### SimilarityRouter — slot pre-registration (trainer.py)

`SimilarityRouter` produces identical cosine-similarity embeddings for slots registered with
generic text labels ("period 2018 slot 0" ≈ "period 2018 slot 7"), collapsing routing to
slot 0. Fix: pre-register initial slots with fixed random unit vectors seeded by `slot_id`
(`numpy.random.RandomState(seed=slot_id)`), bypassing the sentence transformer entirely.

`_pre_register_similarity_router()` in `trainer.py` does this at the start of every period.
Called after contradiction branching but before the training loop.

After each period, `_sync_similarity_router()` updates contradiction-branched slots with full
semantic metadata (entity/relation/period/value from MemoryRegistry) so routing improves over
time as real content is written.

### Diagnosis checklist — if eval metrics are 0.000

1. Check `generate()` sets `_routing_slot_ids` before calling `backbone.generate()`.
2. Check `_memory_hook` uses out-of-place accumulation (torch.index_put, not `buffer[mask] +=`).
3. Print `gate_logits.mean()` after training — if exactly `-2.197` for all slots, gradients
   are not reaching the slot bank. Check the hook accumulation method.
4. Compare `model.generate(input_ids=x)` vs `model.backbone.generate(input_ids=x)` outputs —
   they must differ if memory injection is working.

### SparseMemoryBlock init (smf_model.py)

- `gate_logits` init: `log(0.1/0.9) ≈ -2.197` → sigmoid ≈ 0.1 (sparse by default)
- `query_proj.weight` init: zeros (content-dependent gating starts disabled; gate is purely
  global until query_proj learns)
- `memory` init: `torch.randn * 0.02` (small but nonzero — contributes from step 1)

These are intentional. `query_proj` learning from zero is expected and correct.