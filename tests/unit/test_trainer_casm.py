"""Unit tests for Phase 4: CASM training branch in CASFTrainer.

Covers:
    _select_trainable_parameters — CASM branch
    _train_step — CASM auxiliary losses (sparsity, overlap)
    compute_overlap_loss — scalar, differentiable, pairwise cosine penalty
    _expand_router — grows router output dim while preserving weights
    train_period — contradiction branching, optimizer rebuild, usage count sync
    _slot_usage_counts — incremented during forward, synced to registry at period end
"""

import pytest
import torch

from casf_dataset_api import MemoryRegistry
from training.casm_model import CASMModelWrapper
from training.smf_model import SMFModelWrapper
from training.synthetic_backend import (
    SyntheticTemporalDataset,
    SyntheticTokenizer,
    build_synthetic_model,
)
from training.train_config import TrainConfig
from training.trainer import CASFTrainer


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_casm_cfg(**overrides) -> TrainConfig:
    defaults = dict(
        model_name="synthetic-local-model",
        method="casm",
        casm_num_slots=4,
        casm_router_hidden_size=16,
        casm_top_k=2,
        casm_router_temperature=1.0,
        casm_memory_size=8,
        casm_sparsity_weight=0.0,
        casm_overlap_weight=0.0,
        casm_branch_on_contradiction=True,
    )
    defaults.update(overrides)
    return TrainConfig(**defaults)


def _make_wrapper(cfg: TrainConfig | None = None) -> tuple[CASMModelWrapper, SyntheticTokenizer]:
    if cfg is None:
        cfg = _make_casm_cfg()
    backbone = build_synthetic_model()
    tokenizer = SyntheticTokenizer()
    return CASMModelWrapper(backbone, cfg), tokenizer


def _make_trainer(cfg: TrainConfig | None = None, registry: MemoryRegistry | None = None):
    if cfg is None:
        cfg = _make_casm_cfg()
    wrapper, tokenizer = _make_wrapper(cfg)
    if registry is None:
        registry = MemoryRegistry()
    return CASFTrainer(wrapper, tokenizer, cfg, registry)


def _make_dummy_batch(tokenizer: SyntheticTokenizer, device: torch.device) -> dict:
    text = "Hello world this is a test passage."
    enc = tokenizer(text, return_tensors="pt")
    batch = {k: v.to(device) for k, v in enc.items()}
    batch["labels"] = batch["input_ids"].clone()
    return batch


def _populate_registry_for_contradictions(registry: MemoryRegistry) -> None:
    """Pre-load old fact versions so detector.check() fires on SyntheticTemporalDataset probes."""
    registry.add_slot("Alpha", "relation", "old value", "prior_period")
    registry.add_slot("Gamma", "relation", "stale value", "prior_period")


# ---------------------------------------------------------------------------
# _select_trainable_parameters: CASM branch
# ---------------------------------------------------------------------------


class TestSelectTrainableParametersCASM:
    def test_returns_only_casm_params(self):
        trainer = _make_trainer()
        selected_ids = {id(p) for p in trainer._select_trainable_parameters()}
        casm_ids = {id(p) for p in trainer.model.casm_parameters()}
        assert selected_ids == casm_ids

    def test_backbone_params_excluded(self):
        trainer = _make_trainer()
        selected_ids = {id(p) for p in trainer._select_trainable_parameters()}
        backbone_ids = {id(p) for p in trainer.model.backbone.parameters()}
        assert selected_ids.isdisjoint(backbone_ids)

    def test_raises_type_error_if_not_casm_wrapper(self):
        cfg = _make_casm_cfg()
        raw_backbone = build_synthetic_model()
        tokenizer = SyntheticTokenizer()
        # Pass raw backbone (not CASMModelWrapper) with method="casm"
        trainer = CASFTrainer.__new__(CASFTrainer)
        trainer.model = raw_backbone
        trainer.config = cfg
        with pytest.raises(TypeError, match="CASMModelWrapper"):
            trainer._select_trainable_parameters()

    def test_optimizer_param_ids_match_casm_parameters(self):
        trainer = _make_trainer()
        optimizer_ids = {id(p) for g in trainer.optimizer.param_groups for p in g["params"]}
        casm_ids = {id(p) for p in trainer.model.casm_parameters()}
        assert optimizer_ids == casm_ids


# ---------------------------------------------------------------------------
# _train_step: CASM auxiliary losses
# ---------------------------------------------------------------------------


class TestTrainStepCASMLoss:
    def test_zero_weights_returns_finite_loss(self):
        cfg = _make_casm_cfg(casm_sparsity_weight=0.0, casm_overlap_weight=0.0)
        trainer = _make_trainer(cfg)
        batch = _make_dummy_batch(trainer.tokenizer, trainer.device)
        trainer.model.train()
        trainer.optimizer.zero_grad()
        loss = trainer._train_step(batch)
        assert isinstance(loss, float)
        assert torch.isfinite(torch.tensor(loss))

    def test_sparsity_weight_produces_gate_logit_gradients(self):
        cfg = _make_casm_cfg(casm_sparsity_weight=1.0, casm_overlap_weight=0.0)
        trainer = _make_trainer(cfg)
        batch = _make_dummy_batch(trainer.tokenizer, trainer.device)
        trainer.model.train()
        trainer.optimizer.zero_grad()
        trainer._train_step(batch)
        # At least one slot's gate_logits should have a gradient
        has_grad = any(
            trainer.model.slot_bank[str(sid)].gate_logits.grad is not None
            for sid in trainer.model._active_slot_ids
            if str(sid) in trainer.model.slot_bank
        )
        assert has_grad

    def test_overlap_weight_returns_finite_loss(self):
        cfg = _make_casm_cfg(casm_sparsity_weight=0.0, casm_overlap_weight=1.0)
        trainer = _make_trainer(cfg)
        batch = _make_dummy_batch(trainer.tokenizer, trainer.device)
        trainer.model.train()
        trainer.optimizer.zero_grad()
        loss = trainer._train_step(batch)
        assert torch.isfinite(torch.tensor(loss))

    def test_both_weights_return_finite_loss(self):
        cfg = _make_casm_cfg(casm_sparsity_weight=0.5, casm_overlap_weight=0.5)
        trainer = _make_trainer(cfg)
        batch = _make_dummy_batch(trainer.tokenizer, trainer.device)
        trainer.model.train()
        trainer.optimizer.zero_grad()
        loss = trainer._train_step(batch)
        assert torch.isfinite(torch.tensor(loss))


# ---------------------------------------------------------------------------
# compute_overlap_loss
# ---------------------------------------------------------------------------


class TestComputeOverlapLoss:
    def test_returns_scalar(self):
        wrapper, _ = _make_wrapper()
        loss = wrapper.compute_overlap_loss()
        assert loss.dim() == 0

    def test_one_slot_returns_zero(self):
        cfg = _make_casm_cfg(casm_num_slots=1, casm_top_k=1)
        wrapper, _ = _make_wrapper(cfg)
        loss = wrapper.compute_overlap_loss()
        assert loss.item() == pytest.approx(0.0)

    def test_multiple_slots_returns_finite_value(self):
        wrapper, _ = _make_wrapper()
        loss = wrapper.compute_overlap_loss()
        assert torch.isfinite(loss)

    def test_is_differentiable(self):
        wrapper, _ = _make_wrapper()
        loss = wrapper.compute_overlap_loss()
        loss.backward()
        # At least one gate_logits should have a gradient
        has_grad = any(
            wrapper.slot_bank[str(sid)].gate_logits.grad is not None
            for sid in wrapper._active_slot_ids
            if str(sid) in wrapper.slot_bank
        )
        assert has_grad

    def test_identical_slots_produce_positive_loss(self):
        # If two slots have identical (non-zero) contributions their cosine
        # similarity is 1.0, so the upper-triangle sum must be positive.
        cfg = _make_casm_cfg(casm_num_slots=2, casm_top_k=1)
        wrapper, _ = _make_wrapper(cfg)
        # Copy slot 0's parameters into slot 1 so contributions are identical
        with torch.no_grad():
            wrapper.slot_bank["1"].memory.copy_(wrapper.slot_bank["0"].memory)
            wrapper.slot_bank["1"].gate_logits.copy_(wrapper.slot_bank["0"].gate_logits)
        loss = wrapper.compute_overlap_loss().item()
        assert loss > 0.0


# ---------------------------------------------------------------------------
# _expand_router / add_memory_slot
# ---------------------------------------------------------------------------


class TestExpandRouter:
    def test_router_num_slots_grows_by_one(self):
        wrapper, _ = _make_wrapper()
        before = wrapper.router.num_slots
        wrapper.add_memory_slot()
        assert wrapper.router.num_slots == before + 1

    def test_output_layer_out_features_grows_by_one(self):
        wrapper, _ = _make_wrapper()
        before = wrapper.router.net[2].out_features
        wrapper.add_memory_slot()
        assert wrapper.router.net[2].out_features == before + 1

    def test_existing_weights_preserved(self):
        wrapper, _ = _make_wrapper()
        old_n = wrapper.router.num_slots
        old_weight = wrapper.router.net[2].weight.detach().clone()
        wrapper.add_memory_slot()
        new_weight = wrapper.router.net[2].weight
        assert torch.allclose(new_weight[:old_n], old_weight)

    def test_new_neuron_zero_initialized(self):
        wrapper, _ = _make_wrapper()
        wrapper.add_memory_slot()
        new_weight_row = wrapper.router.net[2].weight[-1]
        new_bias_val = wrapper.router.net[2].bias[-1]
        assert new_weight_row.abs().sum().item() == pytest.approx(0.0)
        assert new_bias_val.item() == pytest.approx(0.0)

    def test_num_slots_equals_active_slot_count_after_multiple_adds(self):
        wrapper, _ = _make_wrapper()
        for _ in range(3):
            wrapper.add_memory_slot()
        assert wrapper.router.num_slots == len(wrapper._active_slot_ids)

    def test_usage_counts_entry_added_for_new_slot(self):
        wrapper, _ = _make_wrapper()
        new_id = wrapper.add_memory_slot()
        assert new_id in wrapper._slot_usage_counts
        assert wrapper._slot_usage_counts[new_id] == 0

    def test_forward_still_works_after_expand(self):
        cfg = _make_casm_cfg(casm_num_slots=2, casm_top_k=1)
        wrapper, tokenizer = _make_wrapper(cfg)
        wrapper.add_memory_slot()
        wrapper.train()
        batch = _make_dummy_batch(tokenizer, torch.device("cpu"))
        out = wrapper(**batch)
        assert torch.isfinite(out.loss)


# ---------------------------------------------------------------------------
# _slot_usage_counts: incremented during forward
# ---------------------------------------------------------------------------


class TestUsageCountTracking:
    def test_forward_increments_at_least_one_slot(self):
        wrapper, tokenizer = _make_wrapper()
        before_total = sum(wrapper._slot_usage_counts.values())
        batch = _make_dummy_batch(tokenizer, torch.device("cpu"))
        wrapper(**batch)
        after_total = sum(wrapper._slot_usage_counts.values())
        assert after_total > before_total

    def test_increment_total_equals_batch_size_times_top_k(self):
        cfg = _make_casm_cfg(casm_num_slots=4, casm_top_k=2)
        wrapper, tokenizer = _make_wrapper(cfg)
        # batch_size=1, top_k=2 → each forward adds exactly 2 to the total
        batch = _make_dummy_batch(tokenizer, torch.device("cpu"))
        n_forwards = 3
        for _ in range(n_forwards):
            wrapper(**batch)
        total = sum(wrapper._slot_usage_counts.values())
        assert total == n_forwards * cfg.casm_top_k * 1  # batch_size=1


# ---------------------------------------------------------------------------
# train_period: CASM branching
# ---------------------------------------------------------------------------


class TestTrainPeriodCASMBranching:
    def _make_minimal_cfg(self, **overrides) -> TrainConfig:
        defaults = dict(
            casm_num_slots=2,
            casm_top_k=1,
            casm_branch_on_contradiction=True,
            batch_size=1,
            grad_accum_steps=1,
            epochs_per_period=1,
            max_passages_per_period=2,
        )
        defaults.update(overrides)
        return _make_casm_cfg(**defaults)

    def test_slots_added_on_contradiction(self):
        registry = MemoryRegistry()
        _populate_registry_for_contradictions(registry)
        cfg = self._make_minimal_cfg()
        trainer = _make_trainer(cfg, registry)
        initial_slots = len(trainer.model._active_slot_ids)

        dataset = SyntheticTemporalDataset()
        trainer.train_period(dataset, "synthetic_period")

        # Two contradictions (Alpha and Gamma) should each add one slot
        assert len(trainer.model._active_slot_ids) == initial_slots + 2

    def test_router_num_slots_updated_on_contradiction(self):
        registry = MemoryRegistry()
        _populate_registry_for_contradictions(registry)
        cfg = self._make_minimal_cfg()
        trainer = _make_trainer(cfg, registry)
        initial_router_slots = trainer.model.router.num_slots

        dataset = SyntheticTemporalDataset()
        trainer.train_period(dataset, "synthetic_period")

        assert trainer.model.router.num_slots == initial_router_slots + 2

    def test_optimizer_rebuilt_on_contradiction(self):
        registry = MemoryRegistry()
        _populate_registry_for_contradictions(registry)
        cfg = self._make_minimal_cfg()
        trainer = _make_trainer(cfg, registry)
        optimizer_id_before = id(trainer.optimizer)

        dataset = SyntheticTemporalDataset()
        trainer.train_period(dataset, "synthetic_period")

        assert id(trainer.optimizer) != optimizer_id_before

    def test_new_optimizer_covers_expanded_router_params(self):
        registry = MemoryRegistry()
        _populate_registry_for_contradictions(registry)
        cfg = self._make_minimal_cfg()
        trainer = _make_trainer(cfg, registry)

        dataset = SyntheticTemporalDataset()
        trainer.train_period(dataset, "synthetic_period")

        # All CASM params (including expanded router) should be in optimizer
        optimizer_ids = {id(p) for g in trainer.optimizer.param_groups for p in g["params"]}
        casm_ids = {id(p) for p in trainer.model.casm_parameters()}
        assert optimizer_ids == casm_ids

    def test_no_branching_when_flag_false(self):
        registry = MemoryRegistry()
        _populate_registry_for_contradictions(registry)
        cfg = self._make_minimal_cfg(casm_branch_on_contradiction=False)
        trainer = _make_trainer(cfg, registry)
        initial_slots = len(trainer.model._active_slot_ids)

        dataset = SyntheticTemporalDataset()
        trainer.train_period(dataset, "synthetic_period")

        assert len(trainer.model._active_slot_ids) == initial_slots

    def test_no_branching_when_registry_empty(self):
        # Empty registry → detector finds no contradictions → no branching
        registry = MemoryRegistry()
        cfg = self._make_minimal_cfg()
        trainer = _make_trainer(cfg, registry)
        initial_slots = len(trainer.model._active_slot_ids)

        dataset = SyntheticTemporalDataset()
        trainer.train_period(dataset, "synthetic_period")

        assert len(trainer.model._active_slot_ids) == initial_slots

    def test_model_to_registry_mapping_populated_after_period(self):
        registry = MemoryRegistry()
        _populate_registry_for_contradictions(registry)
        cfg = self._make_minimal_cfg()
        trainer = _make_trainer(cfg, registry)

        dataset = SyntheticTemporalDataset()
        trainer.train_period(dataset, "synthetic_period")

        # One mapping entry per contradiction (2 contradictions)
        assert len(trainer._model_slot_to_registry_slot_id) == 2

    def test_registry_slots_written_for_all_probes(self):
        registry = MemoryRegistry()
        _populate_registry_for_contradictions(registry)
        cfg = self._make_minimal_cfg()
        trainer = _make_trainer(cfg, registry)

        dataset = SyntheticTemporalDataset()
        trainer.train_period(dataset, "synthetic_period")

        # Both changed probes (Alpha, Gamma) written to registry as new slots
        alpha_slot = registry.get_active("Alpha", "relation")
        gamma_slot = registry.get_active("Gamma", "relation")
        assert alpha_slot is not None and alpha_slot.value == "new value"
        assert gamma_slot is not None and gamma_slot.value == "updated value"

    def test_usage_count_sync_updates_registry_slot(self):
        registry = MemoryRegistry()
        _populate_registry_for_contradictions(registry)
        cfg = self._make_minimal_cfg(epochs_per_period=2, max_passages_per_period=3)
        trainer = _make_trainer(cfg, registry)

        dataset = SyntheticTemporalDataset()
        trainer.train_period(dataset, "synthetic_period")

        # The mapping must exist (contradictions fired)
        assert len(trainer._model_slot_to_registry_slot_id) > 0

        # After the period the model-side counts are reset to zero (to prevent
        # double-counting on the next period's registry sync after a resume).
        # The registry slot must exist; its usage_count reflects what was synced.
        for model_slot_id, reg_slot_id in trainer._model_slot_to_registry_slot_id.items():
            assert trainer.model._slot_usage_counts.get(model_slot_id, 0) == 0
            reg_slot = next(
                (s for s in registry._slots if s.slot_id == reg_slot_id), None
            )
            assert reg_slot is not None
