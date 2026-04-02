"""Tests for SMF checkpoint save/load (Phase 5).

Covers:
    SMFModelWrapper.save_pretrained — persists memory block weights
    SMFModelWrapper.load_memory_into — restores memory block weights
    CASFTrainer.resume — calls load_memory_into so SMF memory is not random after resume
    validate_checkpoint_method_compatibility — rejects mismatched methods for SMF checkpoints
"""

import pytest
import torch

from artifacts.checkpointing import CheckpointError
from casf_dataset_api import MemoryRegistry
from training.smf_model import SMFModelWrapper
from training.synthetic_backend import SyntheticTokenizer, build_synthetic_model
from training.train_config import TrainConfig
from training.trainer import CASFTrainer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cfg(**overrides) -> TrainConfig:
    defaults = dict(
        model_name="synthetic-local-model",
        method="smf",
        smf_memory_size=16,
        smf_sparsity_ratio=0.5,
        smf_update_layers=[0, 1],
        smf_regularization_weight=0.01,
        smf_freeze_backbone=True,
    )
    defaults.update(overrides)
    return TrainConfig(**defaults)


def _make_wrapper(cfg=None) -> tuple[SMFModelWrapper, TrainConfig]:
    if cfg is None:
        cfg = _make_cfg()
    return SMFModelWrapper(build_synthetic_model(), cfg), cfg


def _fresh_wrapper(cfg: TrainConfig) -> SMFModelWrapper:
    return SMFModelWrapper(build_synthetic_model(), cfg)


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
    "resume_compatibility": {"method": "smf"},
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
# save_pretrained / load_memory_into
# ---------------------------------------------------------------------------


class TestSMFMemoryRoundTrip:
    def test_memory_weights_restored(self, tmp_path):
        wrapper, cfg = _make_wrapper()
        with torch.no_grad():
            wrapper.memory_blocks["0"].memory.fill_(55.0)

        wrapper.save_pretrained(str(tmp_path))

        new_wrapper = _fresh_wrapper(cfg)
        SMFModelWrapper.load_memory_into(new_wrapper, str(tmp_path))

        assert torch.allclose(
            new_wrapper.memory_blocks["0"].memory,
            torch.full_like(new_wrapper.memory_blocks["0"].memory, 55.0),
        )

    def test_gate_logits_restored(self, tmp_path):
        wrapper, cfg = _make_wrapper()
        with torch.no_grad():
            wrapper.memory_blocks["1"].gate_logits.fill_(3.0)

        wrapper.save_pretrained(str(tmp_path))

        new_wrapper = _fresh_wrapper(cfg)
        SMFModelWrapper.load_memory_into(new_wrapper, str(tmp_path))

        assert torch.allclose(
            new_wrapper.memory_blocks["1"].gate_logits,
            torch.full_like(new_wrapper.memory_blocks["1"].gate_logits, 3.0),
        )

    def test_all_configured_layers_restored(self, tmp_path):
        cfg = _make_cfg(smf_update_layers=[0, 1])
        wrapper, _ = _make_wrapper(cfg)
        with torch.no_grad():
            wrapper.memory_blocks["0"].memory.fill_(1.0)
            wrapper.memory_blocks["1"].memory.fill_(2.0)

        wrapper.save_pretrained(str(tmp_path))

        new_wrapper = _fresh_wrapper(cfg)
        SMFModelWrapper.load_memory_into(new_wrapper, str(tmp_path))

        assert torch.allclose(
            new_wrapper.memory_blocks["0"].memory,
            torch.full_like(new_wrapper.memory_blocks["0"].memory, 1.0),
        )
        assert torch.allclose(
            new_wrapper.memory_blocks["1"].memory,
            torch.full_like(new_wrapper.memory_blocks["1"].memory, 2.0),
        )

    def test_missing_smf_file_is_no_op(self, tmp_path):
        """load_memory_into must not raise when smf_memory.pt is absent."""
        cfg = _make_cfg()
        new_wrapper = _fresh_wrapper(cfg)
        SMFModelWrapper.load_memory_into(new_wrapper, str(tmp_path))  # no file → no-op


# ---------------------------------------------------------------------------
# CASFTrainer.resume restores SMF memory
# ---------------------------------------------------------------------------


class TestSMFTrainerResume:
    def test_resume_restores_memory_weights(self, tmp_path):
        """Memory block weights must not stay random after resume."""
        cfg = _make_cfg()
        trainer = _make_trainer(cfg)
        with torch.no_grad():
            trainer.model.memory_blocks["0"].memory.fill_(88.0)
        trainer._checkpoint_state = _minimal_checkpoint_state(trainer, "aug_sep")
        _do_checkpoint(trainer, "aug_sep", str(tmp_path))

        new_wrapper = _fresh_wrapper(cfg)
        new_trainer = CASFTrainer(new_wrapper, SyntheticTokenizer(), cfg, MemoryRegistry())
        new_trainer.resume(str(tmp_path))

        assert torch.allclose(
            new_wrapper.memory_blocks["0"].memory,
            torch.full_like(new_wrapper.memory_blocks["0"].memory, 88.0),
        )

    def test_resume_restores_registry(self, tmp_path):
        cfg = _make_cfg()
        registry = MemoryRegistry()
        registry.add_slot("London", "capital_of", "England", "aug_sep")
        trainer = _make_trainer(cfg, registry)
        trainer._checkpoint_state = _minimal_checkpoint_state(trainer, "aug_sep")
        _do_checkpoint(trainer, "aug_sep", str(tmp_path))

        new_wrapper = _fresh_wrapper(cfg)
        new_registry = MemoryRegistry()
        new_trainer = CASFTrainer(new_wrapper, SyntheticTokenizer(), cfg, new_registry)
        new_trainer.resume(str(tmp_path))

        slot = new_registry.get_active("London", "capital_of")
        assert slot is not None
        assert slot.value == "England"

    def test_resume_rejects_casm_checkpoint(self, tmp_path):
        """SMF trainer must refuse a CASM checkpoint."""
        from training.casm_model import CASMModelWrapper
        from training.train_config import TrainConfig as TC

        casm_cfg = TC(
            model_name="synthetic-local-model",
            method="casm",
            casm_num_slots=2,
            casm_router_hidden_size=16,
            casm_top_k=1,
            casm_router_temperature=1.0,
            casm_memory_size=8,
            casm_sparsity_weight=0.0,
            casm_overlap_weight=0.0,
            casm_branch_on_contradiction=False,
        )
        casm_wrapper = CASMModelWrapper(build_synthetic_model(), casm_cfg)
        casm_trainer = CASFTrainer(casm_wrapper, SyntheticTokenizer(), casm_cfg, MemoryRegistry())
        casm_trainer._checkpoint_state = _minimal_checkpoint_state(casm_trainer, "aug_sep")
        casm_trainer.checkpoint(
            "aug_sep",
            str(tmp_path),
            manifest_metadata={
                "model_name": "synthetic-local-model",
                "training_plan": ["aug_sep"],
                "resume_compatibility": {"method": "casm"},
                "dataset_identity": {"name": "synthetic"},
            },
            lock_run_root=False,
        )

        smf_wrapper = _fresh_wrapper(_make_cfg())
        smf_trainer = CASFTrainer(smf_wrapper, SyntheticTokenizer(), _make_cfg(), MemoryRegistry())
        with pytest.raises(CheckpointError, match="casm"):
            smf_trainer.resume(str(tmp_path))
