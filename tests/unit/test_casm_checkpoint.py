"""Tests for CASM checkpoint save/load (Phase 5).

Covers:
    CASMModelWrapper.save_pretrained — persists slot_usage_counts
    CASMModelWrapper.load_memory_into — recreates extra slots, resizes router,
        restores usage counts and slot metadata
    CASFTrainer.checkpoint / resume — CASM round-trip including
        _model_slot_to_registry_slot_id and registry state
    validate_checkpoint_method_compatibility — rejects method mismatch
    usage-count reset — counts are zeroed after the period-end registry sync
"""

import json

import pytest
import torch

from artifacts.checkpointing import CheckpointError, validate_checkpoint_method_compatibility
from casf_dataset_api import MemoryRegistry
from training.casm_model import CASMModelWrapper
from training.synthetic_backend import (
    SyntheticTemporalDataset,
    SyntheticTokenizer,
    build_synthetic_model,
)
from training.train_config import TrainConfig
from training.trainer import CASFTrainer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cfg(**overrides) -> TrainConfig:
    defaults = dict(
        model_name="synthetic-local-model",
        method="casm",
        casm_num_slots=3,
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


def _make_wrapper(cfg=None) -> tuple[CASMModelWrapper, TrainConfig]:
    if cfg is None:
        cfg = _make_cfg()
    backbone = build_synthetic_model()
    return CASMModelWrapper(backbone, cfg), cfg


def _fresh_wrapper(cfg: TrainConfig) -> CASMModelWrapper:
    """Build a new wrapper from scratch using *cfg* (simulates a fresh resume)."""
    return CASMModelWrapper(build_synthetic_model(), cfg)


def _make_trainer(cfg=None, registry=None) -> CASFTrainer:
    if cfg is None:
        cfg = _make_cfg()
    wrapper, cfg = _make_wrapper(cfg)
    if registry is None:
        registry = MemoryRegistry()
    return CASFTrainer(wrapper, SyntheticTokenizer(), cfg, registry)


_MANIFEST_METADATA = {
    "model_name": "synthetic-local-model",
    "training_plan": ["aug_sep", "sep_oct", "oct_nov", "nov_dec"],
    "resume_compatibility": {"method": "casm"},
    "dataset_identity": {"name": "synthetic"},
}


def _minimal_checkpoint_state(trainer: CASFTrainer, period: str) -> dict:
    return {
        "schema_version": 1,
        "last_period": period,
        "current_unit": period,
        "completed_units": [period],
        "next_batch_index": 0,
        "total_batches": 0,
        "optimizer_steps_total": 0,
        "total_optimizer_steps": 0,
        "unit_snapshot": [],
        "unit_completed": True,
        "optimizer_state_dict": trainer.optimizer.state_dict(),
        "scheduler_state_dict": None,
        "rng_state": trainer._capture_rng_state(),
    }


def _do_checkpoint(trainer: CASFTrainer, period: str, run_root: str) -> None:
    trainer.checkpoint(
        period,
        run_root,
        manifest_metadata=_MANIFEST_METADATA,
        lock_run_root=False,
    )


# ---------------------------------------------------------------------------
# save_pretrained / load_memory_into — slot bank
# ---------------------------------------------------------------------------


class TestSlotBankRoundTrip:
    def test_slot_weights_restored(self, tmp_path):
        wrapper, cfg = _make_wrapper()
        with torch.no_grad():
            wrapper.slot_bank["0"].memory.fill_(99.0)

        wrapper.save_pretrained(str(tmp_path))

        new_wrapper = _fresh_wrapper(cfg)
        CASMModelWrapper.load_memory_into(new_wrapper, str(tmp_path))

        assert torch.allclose(
            new_wrapper.slot_bank["0"].memory,
            torch.full_like(new_wrapper.slot_bank["0"].memory, 99.0),
        )

    def test_all_slots_loaded(self, tmp_path):
        wrapper, cfg = _make_wrapper()
        wrapper.save_pretrained(str(tmp_path))

        new_wrapper = _fresh_wrapper(cfg)
        CASMModelWrapper.load_memory_into(new_wrapper, str(tmp_path))

        assert len(new_wrapper.slot_bank) == cfg.casm_num_slots

    def test_extra_slots_created_on_load(self, tmp_path):
        """Slots added via add_memory_slot() must be re-created during load."""
        wrapper, cfg = _make_wrapper()
        extra_id = wrapper.add_memory_slot()
        extra_key = str(extra_id)
        with torch.no_grad():
            wrapper.slot_bank[extra_key].memory.fill_(77.0)

        wrapper.save_pretrained(str(tmp_path))

        new_wrapper = _fresh_wrapper(cfg)
        assert extra_key not in new_wrapper.slot_bank  # not present before load

        CASMModelWrapper.load_memory_into(new_wrapper, str(tmp_path))

        assert extra_key in new_wrapper.slot_bank
        assert torch.allclose(
            new_wrapper.slot_bank[extra_key].memory,
            torch.full_like(new_wrapper.slot_bank[extra_key].memory, 77.0),
        )


# ---------------------------------------------------------------------------
# save_pretrained / load_memory_into — router
# ---------------------------------------------------------------------------


class TestRouterRoundTrip:
    def test_router_weights_restored(self, tmp_path):
        wrapper, cfg = _make_wrapper()
        with torch.no_grad():
            wrapper.router.net[2].weight.fill_(0.5)

        wrapper.save_pretrained(str(tmp_path))

        new_wrapper = _fresh_wrapper(cfg)
        CASMModelWrapper.load_memory_into(new_wrapper, str(tmp_path))

        assert torch.allclose(
            new_wrapper.router.net[2].weight,
            torch.full_like(new_wrapper.router.net[2].weight, 0.5),
        )

    def test_expanded_router_size_restored(self, tmp_path):
        """Router grown by _expand_router must be resized before state is loaded."""
        wrapper, cfg = _make_wrapper()
        wrapper.add_memory_slot()
        wrapper.add_memory_slot()
        expected_slots = wrapper.router.num_slots

        wrapper.save_pretrained(str(tmp_path))

        new_wrapper = _fresh_wrapper(cfg)
        assert new_wrapper.router.num_slots == cfg.casm_num_slots  # before load

        CASMModelWrapper.load_memory_into(new_wrapper, str(tmp_path))

        assert new_wrapper.router.num_slots == expected_slots

    def test_router_forward_works_after_load(self, tmp_path):
        wrapper, cfg = _make_wrapper()
        wrapper.add_memory_slot()
        wrapper.save_pretrained(str(tmp_path))

        new_wrapper = _fresh_wrapper(cfg)
        CASMModelWrapper.load_memory_into(new_wrapper, str(tmp_path))

        query = torch.randn(1, new_wrapper._hidden_size)
        slot_ids, weights = new_wrapper.router(query, top_k=cfg.casm_top_k)
        assert slot_ids.shape == (1, cfg.casm_top_k)


# ---------------------------------------------------------------------------
# save_pretrained / load_memory_into — slot metadata
# ---------------------------------------------------------------------------


class TestSlotMetadataRoundTrip:
    def test_active_slot_ids_restored(self, tmp_path):
        wrapper, cfg = _make_wrapper()
        wrapper.add_memory_slot()
        wrapper.close_memory_slot(0)
        expected = list(wrapper._active_slot_ids)

        wrapper.save_pretrained(str(tmp_path))

        new_wrapper = _fresh_wrapper(cfg)
        CASMModelWrapper.load_memory_into(new_wrapper, str(tmp_path))

        assert new_wrapper._active_slot_ids == expected

    def test_closed_slot_ids_restored(self, tmp_path):
        wrapper, cfg = _make_wrapper()
        wrapper.close_memory_slot(1)

        wrapper.save_pretrained(str(tmp_path))

        new_wrapper = _fresh_wrapper(cfg)
        CASMModelWrapper.load_memory_into(new_wrapper, str(tmp_path))

        assert 1 in new_wrapper._closed_slot_ids
        assert 1 not in new_wrapper._active_slot_ids

    def test_next_slot_idx_restored(self, tmp_path):
        """After load, add_memory_slot() must not reuse an existing ID."""
        wrapper, cfg = _make_wrapper()
        wrapper.add_memory_slot()
        expected_next = wrapper._next_slot_idx

        wrapper.save_pretrained(str(tmp_path))

        new_wrapper = _fresh_wrapper(cfg)
        CASMModelWrapper.load_memory_into(new_wrapper, str(tmp_path))

        assert new_wrapper._next_slot_idx == expected_next
        new_id = new_wrapper.add_memory_slot()
        assert new_id == expected_next  # no collision


# ---------------------------------------------------------------------------
# save_pretrained / load_memory_into — usage counts
# ---------------------------------------------------------------------------


class TestUsageCountsRoundTrip:
    def test_counts_restored(self, tmp_path):
        wrapper, cfg = _make_wrapper()
        wrapper._slot_usage_counts = {0: 10, 1: 5, 2: 3}

        wrapper.save_pretrained(str(tmp_path))

        new_wrapper = _fresh_wrapper(cfg)
        CASMModelWrapper.load_memory_into(new_wrapper, str(tmp_path))

        assert new_wrapper._slot_usage_counts == {0: 10, 1: 5, 2: 3}

    def test_zero_counts_restored(self, tmp_path):
        """Period-boundary checkpoint saves zeros; they must load as zeros."""
        wrapper, cfg = _make_wrapper()
        wrapper._slot_usage_counts = {0: 0, 1: 0, 2: 0}

        wrapper.save_pretrained(str(tmp_path))

        new_wrapper = _fresh_wrapper(cfg)
        CASMModelWrapper.load_memory_into(new_wrapper, str(tmp_path))

        assert all(v == 0 for v in new_wrapper._slot_usage_counts.values())

    def test_counts_for_extra_slots_restored(self, tmp_path):
        wrapper, cfg = _make_wrapper()
        new_id = wrapper.add_memory_slot()
        wrapper._slot_usage_counts[new_id] = 42

        wrapper.save_pretrained(str(tmp_path))

        new_wrapper = _fresh_wrapper(cfg)
        CASMModelWrapper.load_memory_into(new_wrapper, str(tmp_path))

        assert new_wrapper._slot_usage_counts.get(new_id) == 42

    def test_backward_compat_missing_counts_key(self, tmp_path):
        """Checkpoints written before usage-count persistence get zeroed counts."""
        wrapper, cfg = _make_wrapper()
        wrapper.save_pretrained(str(tmp_path))

        # Strip the key to simulate an old checkpoint.
        state = torch.load(str(tmp_path / "casm_memory.pt"), map_location="cpu", weights_only=True)
        state.pop("slot_usage_counts")
        torch.save(state, str(tmp_path / "casm_memory.pt"))

        new_wrapper = _fresh_wrapper(cfg)
        CASMModelWrapper.load_memory_into(new_wrapper, str(tmp_path))

        assert all(v == 0 for v in new_wrapper._slot_usage_counts.values())


# ---------------------------------------------------------------------------
# CASFTrainer.checkpoint + resume — full round-trip
# ---------------------------------------------------------------------------


class TestTrainerCheckpointResume:
    def test_slot_bank_restored_via_resume(self, tmp_path):
        cfg = _make_cfg()
        trainer = _make_trainer(cfg)
        with torch.no_grad():
            trainer.model.slot_bank["0"].memory.fill_(42.0)
        trainer._checkpoint_state = _minimal_checkpoint_state(trainer, "aug_sep")
        _do_checkpoint(trainer, "aug_sep", str(tmp_path))

        new_wrapper = _fresh_wrapper(cfg)
        new_trainer = CASFTrainer(new_wrapper, SyntheticTokenizer(), cfg, MemoryRegistry())
        new_trainer.resume(str(tmp_path))

        assert torch.allclose(
            new_wrapper.slot_bank["0"].memory,
            torch.full_like(new_wrapper.slot_bank["0"].memory, 42.0),
        )

    def test_router_weights_restored_via_resume(self, tmp_path):
        cfg = _make_cfg()
        trainer = _make_trainer(cfg)
        with torch.no_grad():
            trainer.model.router.net[2].weight.fill_(0.25)
        trainer._checkpoint_state = _minimal_checkpoint_state(trainer, "aug_sep")
        _do_checkpoint(trainer, "aug_sep", str(tmp_path))

        new_wrapper = _fresh_wrapper(cfg)
        new_trainer = CASFTrainer(new_wrapper, SyntheticTokenizer(), cfg, MemoryRegistry())
        new_trainer.resume(str(tmp_path))

        assert torch.allclose(
            new_wrapper.router.net[2].weight,
            torch.full_like(new_wrapper.router.net[2].weight, 0.25),
        )

    def test_slot_mapping_restored_via_resume(self, tmp_path):
        cfg = _make_cfg()
        trainer = _make_trainer(cfg)
        trainer._model_slot_to_registry_slot_id = {0: 100, 1: 200}
        trainer._checkpoint_state = _minimal_checkpoint_state(trainer, "aug_sep")
        _do_checkpoint(trainer, "aug_sep", str(tmp_path))

        new_wrapper = _fresh_wrapper(cfg)
        new_trainer = CASFTrainer(new_wrapper, SyntheticTokenizer(), cfg, MemoryRegistry())
        new_trainer.resume(str(tmp_path))

        assert new_trainer._model_slot_to_registry_slot_id == {0: 100, 1: 200}

    def test_registry_restored_via_resume(self, tmp_path):
        cfg = _make_cfg()
        registry = MemoryRegistry()
        registry.add_slot("Alpha", "capital", "Berlin", "aug_sep")
        trainer = _make_trainer(cfg, registry)
        trainer._checkpoint_state = _minimal_checkpoint_state(trainer, "aug_sep")
        _do_checkpoint(trainer, "aug_sep", str(tmp_path))

        new_wrapper = _fresh_wrapper(cfg)
        new_registry = MemoryRegistry()
        new_trainer = CASFTrainer(new_wrapper, SyntheticTokenizer(), cfg, new_registry)
        new_trainer.resume(str(tmp_path))

        slot = new_registry.get_active("Alpha", "capital")
        assert slot is not None
        assert slot.value == "Berlin"

    def test_expanded_slots_restored_via_resume(self, tmp_path):
        """Contradiction-branched slots survive checkpoint + resume."""
        cfg = _make_cfg()
        trainer = _make_trainer(cfg)
        trainer.model.add_memory_slot()
        trainer.model.add_memory_slot()
        expected_slots = len(trainer.model._active_slot_ids)
        trainer._checkpoint_state = _minimal_checkpoint_state(trainer, "aug_sep")
        _do_checkpoint(trainer, "aug_sep", str(tmp_path))

        new_wrapper = _fresh_wrapper(cfg)
        new_trainer = CASFTrainer(new_wrapper, SyntheticTokenizer(), cfg, MemoryRegistry())
        new_trainer.resume(str(tmp_path))

        assert len(new_wrapper._active_slot_ids) == expected_slots
        assert new_wrapper.router.num_slots == expected_slots


# ---------------------------------------------------------------------------
# Usage-count reset after period-end sync
# ---------------------------------------------------------------------------


class TestUsageCountReset:
    def test_counts_zeroed_after_train_period(self):
        """_slot_usage_counts must be reset to zero at the end of every period."""
        cfg = _make_cfg(
            casm_num_slots=2,
            casm_top_k=1,
            casm_branch_on_contradiction=False,
            batch_size=1,
            grad_accum_steps=1,
            epochs_per_period=1,
            max_passages_per_period=2,
        )
        trainer = _make_trainer(cfg)
        dataset = SyntheticTemporalDataset()
        trainer.train_period(dataset, "aug_sep")

        for slot_id, count in trainer.model._slot_usage_counts.items():
            assert count == 0, f"slot {slot_id} count should be 0 after period sync, got {count}"

    def test_period_boundary_checkpoint_saves_zero_counts(self, tmp_path):
        """Checkpoint written after period completion stores zeroed counts."""
        cfg = _make_cfg(
            casm_num_slots=2,
            casm_top_k=1,
            casm_branch_on_contradiction=False,
            batch_size=1,
            grad_accum_steps=1,
            epochs_per_period=1,
            max_passages_per_period=2,
        )
        trainer = _make_trainer(cfg)
        dataset = SyntheticTemporalDataset()
        trainer.train_period(dataset, "aug_sep")
        _do_checkpoint(trainer, "aug_sep", str(tmp_path))

        new_wrapper = _fresh_wrapper(cfg)
        new_trainer = CASFTrainer(new_wrapper, SyntheticTokenizer(), cfg, MemoryRegistry())
        new_trainer.resume(str(tmp_path))

        assert all(v == 0 for v in new_wrapper._slot_usage_counts.values())


# ---------------------------------------------------------------------------
# validate_checkpoint_method_compatibility
# ---------------------------------------------------------------------------


class TestMethodCompatibility:
    def test_matching_method_passes(self, tmp_path):
        (tmp_path / "train_config.json").write_text(json.dumps({"method": "casm"}))
        validate_checkpoint_method_compatibility(tmp_path, "casm")  # must not raise

    def test_mismatched_method_raises(self, tmp_path):
        (tmp_path / "train_config.json").write_text(json.dumps({"method": "casm"}))
        with pytest.raises(CheckpointError, match="casm"):
            validate_checkpoint_method_compatibility(tmp_path, "full_ft")

    def test_missing_config_skips_check(self, tmp_path):
        # No train_config.json present — should be a no-op.
        validate_checkpoint_method_compatibility(tmp_path, "casm")  # must not raise

    def test_config_without_method_key_skips_check(self, tmp_path):
        (tmp_path / "train_config.json").write_text(json.dumps({"lr": 1e-4}))
        validate_checkpoint_method_compatibility(tmp_path, "casm")  # must not raise

    def test_resume_raises_on_method_mismatch(self, tmp_path):
        """CASFTrainer.resume rejects a checkpoint saved with a different method."""
        casm_cfg = _make_cfg()
        trainer = _make_trainer(casm_cfg)
        trainer._checkpoint_state = _minimal_checkpoint_state(trainer, "aug_sep")
        _do_checkpoint(trainer, "aug_sep", str(tmp_path))

        # Try resuming with full_ft config.
        from training.train_config import TrainConfig
        ft_cfg = TrainConfig(model_name="synthetic-local-model", method="full_ft")
        backbone = build_synthetic_model()
        ft_trainer = CASFTrainer(backbone, SyntheticTokenizer(), ft_cfg, MemoryRegistry())

        with pytest.raises(CheckpointError, match="casm"):
            ft_trainer.resume(str(tmp_path))
