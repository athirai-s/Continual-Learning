"""Unit tests for CASMRouter and CASMModelWrapper (training/casm_model.py).

Covers:
    CASMRouter: valid slot ids, top-k behaviour, output shapes, gradients
    CASMModelWrapper: backbone freezing, forward pass, slot management,
                      trainable parameter selection, persistence
"""

import os

import pytest
import torch

from training.casm_model import CASMRouter, CASMModelWrapper
from training.smf_model import SparseMemoryBlock
from training.synthetic_backend import build_synthetic_model, SyntheticTokenizer
from training.train_config import TrainConfig


# ---------------------------------------------------------------------------
# Helpers


def _make_casm_cfg(**overrides) -> TrainConfig:
    defaults = dict(
        model_name="synthetic-local-model",
        method="casm",
        casm_num_slots=4,
        casm_router_hidden_size=16,
        casm_top_k=2,
        casm_router_temperature=1.0,
        casm_memory_size=8,
    )
    defaults.update(overrides)
    return TrainConfig(**defaults)


def _make_wrapper(cfg: TrainConfig | None = None) -> tuple[CASMModelWrapper, SyntheticTokenizer]:
    if cfg is None:
        cfg = _make_casm_cfg()
    backbone = build_synthetic_model()
    tokenizer = SyntheticTokenizer()
    wrapper = CASMModelWrapper(backbone, cfg)
    return wrapper, tokenizer


def _make_dummy_batch(tokenizer: SyntheticTokenizer, device: torch.device) -> dict:
    text = "Hello world this is a test passage."
    enc = tokenizer(text, return_tensors="pt")
    batch = {k: v.to(device) for k, v in enc.items()}
    batch["labels"] = batch["input_ids"].clone()
    return batch


# ---------------------------------------------------------------------------
# CASMRouter: basic construction


class TestCASMRouterConstruction:
    def test_router_initialises(self):
        router = CASMRouter(hidden_size=32, num_slots=4, router_hidden_size=16)
        assert router.num_slots == 4

    def test_router_raises_on_zero_slots(self):
        with pytest.raises(ValueError, match="num_slots"):
            CASMRouter(hidden_size=32, num_slots=0, router_hidden_size=16)

    def test_router_raises_on_zero_hidden_size(self):
        with pytest.raises(ValueError, match="router_hidden_size"):
            CASMRouter(hidden_size=32, num_slots=4, router_hidden_size=0)


# ---------------------------------------------------------------------------
# CASMRouter: output validity


class TestCASMRouterOutputValidity:
    def _make_router(self, num_slots: int = 4) -> CASMRouter:
        return CASMRouter(hidden_size=32, num_slots=num_slots, router_hidden_size=16)

    def test_slot_ids_are_in_valid_range(self):
        router = self._make_router(num_slots=4)
        query = torch.randn(2, 32)
        slot_ids, _ = router(query, top_k=2)
        assert (slot_ids >= 0).all(), "slot_ids must be non-negative"
        assert (slot_ids < 4).all(), "slot_ids must be < num_slots"

    def test_top_k_one_returns_single_slot_per_example(self):
        router = self._make_router(num_slots=4)
        query = torch.randn(3, 32)
        slot_ids, weights = router(query, top_k=1)
        assert slot_ids.shape == (3, 1)
        assert weights.shape == (3, 1)

    def test_top_k_equals_num_slots(self):
        router = self._make_router(num_slots=4)
        query = torch.randn(2, 32)
        slot_ids, weights = router(query, top_k=4)
        assert slot_ids.shape == (2, 4)
        assert weights.shape == (2, 4)

    def test_weights_sum_to_one(self):
        router = self._make_router(num_slots=4)
        query = torch.randn(5, 32)
        _, weights = router(query, top_k=3)
        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(5), atol=1e-5), (
            "Routing weights should sum to 1 per example"
        )

    def test_weights_are_non_negative(self):
        router = self._make_router(num_slots=4)
        query = torch.randn(4, 32)
        _, weights = router(query, top_k=2)
        assert (weights >= 0).all()

    def test_top_k_exceeds_num_slots_raises(self):
        router = self._make_router(num_slots=3)
        query = torch.randn(2, 32)
        with pytest.raises(ValueError, match="top_k"):
            router(query, top_k=4)


# ---------------------------------------------------------------------------
# CASMRouter: output shapes for various batch sizes


class TestCASMRouterOutputShape:
    def test_batch_size_1(self):
        router = CASMRouter(hidden_size=32, num_slots=4, router_hidden_size=16)
        query = torch.randn(1, 32)
        slot_ids, weights = router(query, top_k=2)
        assert slot_ids.shape == (1, 2)
        assert weights.shape == (1, 2)

    def test_batch_size_8(self):
        router = CASMRouter(hidden_size=32, num_slots=4, router_hidden_size=16)
        query = torch.randn(8, 32)
        slot_ids, weights = router(query, top_k=2)
        assert slot_ids.shape == (8, 2)
        assert weights.shape == (8, 2)

    def test_different_top_k_changes_last_dim(self):
        router = CASMRouter(hidden_size=32, num_slots=6, router_hidden_size=16)
        query = torch.randn(3, 32)
        for k in (1, 2, 3, 6):
            sids, ws = router(query, top_k=k)
            assert sids.shape[-1] == k
            assert ws.shape[-1] == k

    def test_slot_ids_dtype_is_long(self):
        router = CASMRouter(hidden_size=32, num_slots=4, router_hidden_size=16)
        query = torch.randn(2, 32)
        slot_ids, _ = router(query, top_k=2)
        assert slot_ids.dtype == torch.long

    def test_weights_dtype_is_float(self):
        router = CASMRouter(hidden_size=32, num_slots=4, router_hidden_size=16)
        query = torch.randn(2, 32)
        _, weights = router(query, top_k=2)
        assert weights.is_floating_point()


# ---------------------------------------------------------------------------
# CASMRouter: gradient flow


class TestCASMRouterGradients:
    def test_gradients_flow_through_weights(self):
        router = CASMRouter(hidden_size=32, num_slots=4, router_hidden_size=16)
        query = torch.randn(2, 32, requires_grad=True)
        slot_ids, weights = router(query, top_k=2)
        loss = weights.sum()
        loss.backward()
        # Router net parameters should have gradients
        for name, param in router.net.named_parameters():
            assert param.grad is not None, f"Router param '{name}' has no gradient"

    def test_time_signal_ignored_without_error(self):
        """Passing a time_signal should not raise even though it is unused."""
        router = CASMRouter(hidden_size=32, num_slots=4, router_hidden_size=16)
        query = torch.randn(2, 32)
        time_signal = torch.tensor([0.5, 1.0])
        slot_ids, weights = router(query, top_k=2, time_signal=time_signal)
        assert slot_ids.shape == (2, 2)


# ---------------------------------------------------------------------------
# CASMModelWrapper: backbone freezing


class TestCASMWrapperBackboneFreezing:
    def test_all_backbone_params_frozen(self):
        wrapper, _ = _make_wrapper()
        for name, p in wrapper.backbone.named_parameters():
            assert not p.requires_grad, (
                f"Backbone param '{name}' should be frozen"
            )

    def test_slot_bank_params_require_grad(self):
        wrapper, _ = _make_wrapper()
        params = list(wrapper.slot_bank.parameters())
        assert len(params) > 0
        for p in params:
            assert p.requires_grad

    def test_router_params_require_grad(self):
        wrapper, _ = _make_wrapper()
        for name, p in wrapper.router.named_parameters():
            assert p.requires_grad, f"Router param '{name}' should require grad"

    def test_only_casm_params_trainable(self):
        wrapper, _ = _make_wrapper()
        trainable_ids = {id(p) for p in wrapper.parameters() if p.requires_grad}
        casm_ids = {id(p) for p in wrapper.casm_parameters()}
        assert trainable_ids == casm_ids


# ---------------------------------------------------------------------------
# CASMModelWrapper: forward pass


class TestCASMWrapperForward:
    def test_forward_returns_loss(self):
        wrapper, tokenizer = _make_wrapper()
        device = torch.device("cpu")
        batch = _make_dummy_batch(tokenizer, device)
        wrapper.eval()
        with torch.no_grad():
            out = wrapper(**batch)
        assert hasattr(out, "loss")
        assert out.loss is not None

    def test_forward_loss_is_finite(self):
        wrapper, tokenizer = _make_wrapper()
        device = torch.device("cpu")
        batch = _make_dummy_batch(tokenizer, device)
        wrapper.eval()
        with torch.no_grad():
            out = wrapper(**batch)
        assert torch.isfinite(out.loss)

    def test_config_delegates_to_backbone(self):
        wrapper, _ = _make_wrapper()
        assert wrapper.config is wrapper.backbone.config

    def test_memory_contribution_cleared_after_forward(self):
        wrapper, tokenizer = _make_wrapper()
        batch = _make_dummy_batch(tokenizer, torch.device("cpu"))
        wrapper.eval()
        with torch.no_grad():
            wrapper(**batch)
        assert wrapper._current_memory_contribution is None


# ---------------------------------------------------------------------------
# CASMModelWrapper: gradient flow


class TestCASMWrapperGradients:
    def test_slot_bank_receives_gradients(self):
        wrapper, tokenizer = _make_wrapper()
        device = torch.device("cpu")
        batch = _make_dummy_batch(tokenizer, device)
        wrapper.train()
        out = wrapper(**batch)
        out.loss.backward()
        for name, p in wrapper.slot_bank.named_parameters():
            assert p.grad is not None, f"Slot bank param '{name}' has no gradient"

    def test_backbone_params_have_no_gradients(self):
        wrapper, tokenizer = _make_wrapper()
        device = torch.device("cpu")
        batch = _make_dummy_batch(tokenizer, device)
        wrapper.train()
        out = wrapper(**batch)
        out.loss.backward()
        for name, p in wrapper.backbone.named_parameters():
            assert p.grad is None, (
                f"Backbone param '{name}' should have no gradient"
            )


# ---------------------------------------------------------------------------
# CASMModelWrapper: slot lifecycle


class TestCASMWrapperSlotLifecycle:
    def test_initial_slot_count(self):
        cfg = _make_casm_cfg(casm_num_slots=3)
        wrapper, _ = _make_wrapper(cfg)
        assert len(wrapper._active_slot_ids) == 3
        assert wrapper._next_slot_idx == 3

    def test_add_memory_slot_increases_count(self):
        wrapper, _ = _make_wrapper()
        initial = len(wrapper._active_slot_ids)
        new_id = wrapper.add_memory_slot()
        assert len(wrapper._active_slot_ids) == initial + 1
        assert str(new_id) in wrapper.slot_bank

    def test_close_memory_slot_removes_from_active(self):
        wrapper, _ = _make_wrapper()
        slot_id = wrapper._active_slot_ids[0]
        wrapper.close_memory_slot(slot_id)
        assert slot_id not in wrapper._active_slot_ids
        assert slot_id in wrapper._closed_slot_ids

    def test_closed_slot_weights_retained_in_slot_bank(self):
        wrapper, _ = _make_wrapper()
        slot_id = wrapper._active_slot_ids[0]
        wrapper.close_memory_slot(slot_id)
        assert str(slot_id) in wrapper.slot_bank

    def test_close_nonexistent_slot_is_safe(self):
        """Closing a slot that is already inactive (not in _active_slot_ids)
        should not raise."""
        wrapper, _ = _make_wrapper()
        slot_id = wrapper._active_slot_ids[0]
        wrapper.close_memory_slot(slot_id)
        # Closing again should not raise
        wrapper.close_memory_slot(slot_id)

    def test_router_num_slots_matches_initial_count(self):
        cfg = _make_casm_cfg(casm_num_slots=5)
        wrapper, _ = _make_wrapper(cfg)
        assert wrapper.router.num_slots == 5


# ---------------------------------------------------------------------------
# CASMModelWrapper: sparsity loss


class TestCASMWrapperSparsityLoss:
    def test_sparsity_loss_is_non_negative(self):
        wrapper, _ = _make_wrapper()
        loss = wrapper.compute_sparsity_loss()
        assert loss.item() >= 0.0

    def test_sparsity_loss_is_differentiable(self):
        wrapper, _ = _make_wrapper()
        loss = wrapper.compute_sparsity_loss()
        loss.backward()
        for name, p in wrapper.slot_bank.named_parameters():
            if "gate_logits" in name:
                assert p.grad is not None, f"gate_logits '{name}' has no gradient"


# ---------------------------------------------------------------------------
# CASMModelWrapper: persistence


class TestCASMWrapperPersistence:
    def test_save_and_load_memory_round_trip(self, tmp_path):
        cfg = _make_casm_cfg(casm_num_slots=3)
        wrapper, _ = _make_wrapper(cfg)

        # Modify a slot so we can detect a successful restore
        with torch.no_grad():
            wrapper.slot_bank["0"].gate_logits.fill_(99.0)

        wrapper.save_pretrained(str(tmp_path))

        # Build a fresh wrapper and load into it
        backbone2 = build_synthetic_model()
        wrapper2 = CASMModelWrapper(backbone2, cfg)
        CASMModelWrapper.load_memory_into(wrapper2, str(tmp_path))

        saved_logits = wrapper.slot_bank["0"].gate_logits
        loaded_logits = wrapper2.slot_bank["0"].gate_logits
        assert torch.allclose(saved_logits, loaded_logits)

    def test_save_creates_casm_memory_file(self, tmp_path):
        wrapper, _ = _make_wrapper()
        wrapper.save_pretrained(str(tmp_path))
        assert (tmp_path / "casm_memory.pt").exists()

    def test_load_memory_into_noop_when_no_file(self, tmp_path):
        """load_memory_into should silently succeed if casm_memory.pt is absent."""
        wrapper, _ = _make_wrapper()
        CASMModelWrapper.load_memory_into(wrapper, str(tmp_path))  # should not raise
