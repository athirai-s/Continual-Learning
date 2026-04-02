"""Unit tests for SMFModelWrapper (training/smf_model.py).

Verifies:
- backbone parameters are frozen after wrapping
- only sparse memory parameters require gradients
- forward pass produces a loss
- memory parameters receive non-zero gradients after backward
- optimizer built by CASFTrainer contains only memory parameters
- regularization loss is non-negative and differentiable
"""

import pytest
import torch

from training.smf_model import SMFModelWrapper, SparseMemoryBlock
from training.synthetic_backend import build_synthetic_model, SyntheticTokenizer
from training.train_config import TrainConfig
from training.trainer import CASFTrainer
from casf_dataset_api import MemoryRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_smf_cfg(**overrides) -> TrainConfig:
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


def _make_wrapper(cfg: TrainConfig | None = None) -> tuple[SMFModelWrapper, SyntheticTokenizer]:
    if cfg is None:
        cfg = _make_smf_cfg()
    backbone = build_synthetic_model()
    tokenizer = SyntheticTokenizer()
    wrapper = SMFModelWrapper(backbone, cfg)
    return wrapper, tokenizer


def _make_dummy_batch(tokenizer: SyntheticTokenizer, device: torch.device) -> dict:
    text = "Hello world this is a test passage."
    enc = tokenizer(text, return_tensors="pt")
    batch = {k: v.to(device) for k, v in enc.items()}
    batch["labels"] = batch["input_ids"].clone()
    return batch


# ---------------------------------------------------------------------------
# SparseMemoryBlock
# ---------------------------------------------------------------------------

class TestSparseMemoryBlock:
    def test_output_shape_matches_input(self):
        block = SparseMemoryBlock(memory_size=8, hidden_size=32)
        x = torch.randn(2, 10, 32)  # (batch, seq, hidden)
        out = block(x)
        assert out.shape == x.shape

    def test_sparsity_loss_is_non_negative(self):
        block = SparseMemoryBlock(memory_size=8, hidden_size=32)
        loss = block.sparsity_loss()
        assert loss.item() >= 0.0

    def test_sparsity_loss_is_differentiable(self):
        block = SparseMemoryBlock(memory_size=8, hidden_size=32)
        loss = block.sparsity_loss()
        loss.backward()
        assert block.gate_logits.grad is not None


# ---------------------------------------------------------------------------
# SMFModelWrapper: parameter freezing
# ---------------------------------------------------------------------------

class TestBackboneFreezing:
    def test_all_backbone_params_are_frozen(self):
        wrapper, _ = _make_wrapper()
        for name, param in wrapper.backbone.named_parameters():
            assert not param.requires_grad, (
                f"Backbone parameter '{name}' should be frozen but requires_grad=True"
            )

    def test_memory_params_require_grad(self):
        wrapper, _ = _make_wrapper()
        memory_params = list(wrapper.smf_parameters())
        assert len(memory_params) > 0, "No trainable memory parameters found"
        for param in memory_params:
            assert param.requires_grad, "Memory parameter should require grad"

    def test_only_memory_params_require_grad(self):
        wrapper, _ = _make_wrapper()
        trainable = [p for p in wrapper.parameters() if p.requires_grad]
        memory = list(wrapper.smf_parameters())
        assert set(id(p) for p in trainable) == set(id(p) for p in memory), (
            "Parameters with requires_grad=True must be exactly the memory params"
        )

    def test_memory_blocks_exist_for_each_update_layer(self):
        cfg = _make_smf_cfg(smf_update_layers=[0, 1])
        wrapper, _ = _make_wrapper(cfg)
        assert set(wrapper.memory_blocks.keys()) == {"0", "1"}

    def test_single_layer_update(self):
        cfg = _make_smf_cfg(smf_update_layers=[0])
        wrapper, _ = _make_wrapper(cfg)
        assert set(wrapper.memory_blocks.keys()) == {"0"}

    def test_out_of_range_layer_raises(self):
        cfg = _make_smf_cfg(smf_update_layers=[99])
        backbone = build_synthetic_model()
        with pytest.raises(ValueError, match="index 99"):
            SMFModelWrapper(backbone, cfg)


# ---------------------------------------------------------------------------
# SMFModelWrapper: forward pass
# ---------------------------------------------------------------------------

class TestForwardPass:
    def test_forward_returns_loss(self):
        wrapper, tokenizer = _make_wrapper()
        device = torch.device("cpu")
        batch = _make_dummy_batch(tokenizer, device)
        wrapper.eval()
        with torch.no_grad():
            outputs = wrapper(**batch)
        assert hasattr(outputs, "loss"), "Forward output must have .loss"
        assert outputs.loss is not None

    def test_forward_loss_is_finite(self):
        wrapper, tokenizer = _make_wrapper()
        device = torch.device("cpu")
        batch = _make_dummy_batch(tokenizer, device)
        wrapper.eval()
        with torch.no_grad():
            outputs = wrapper(**batch)
        assert torch.isfinite(outputs.loss), "Loss should be finite"

    def test_config_property_delegates_to_backbone(self):
        wrapper, _ = _make_wrapper()
        assert wrapper.config is wrapper.backbone.config


# ---------------------------------------------------------------------------
# SMFModelWrapper: gradient flow
# ---------------------------------------------------------------------------

class TestGradientFlow:
    def test_memory_params_receive_gradients_after_backward(self):
        wrapper, tokenizer = _make_wrapper()
        device = torch.device("cpu")
        batch = _make_dummy_batch(tokenizer, device)
        wrapper.train()
        outputs = wrapper(**batch)
        outputs.loss.backward()
        for name, param in wrapper.memory_blocks.named_parameters():
            assert param.grad is not None, (
                f"Memory param '{name}' has no gradient after backward"
            )
            assert param.grad.abs().sum().item() > 0, (
                f"Memory param '{name}' gradient is all zeros"
            )

    def test_backbone_params_have_no_gradients_after_backward(self):
        wrapper, tokenizer = _make_wrapper()
        device = torch.device("cpu")
        batch = _make_dummy_batch(tokenizer, device)
        wrapper.train()
        outputs = wrapper(**batch)
        outputs.loss.backward()
        for name, param in wrapper.backbone.named_parameters():
            assert param.grad is None, (
                f"Backbone parameter '{name}' should have no gradient but grad is not None"
            )

    def test_regularization_loss_is_differentiable(self):
        wrapper, tokenizer = _make_wrapper()
        device = torch.device("cpu")
        batch = _make_dummy_batch(tokenizer, device)
        wrapper.train()
        outputs = wrapper(**batch)
        reg = wrapper.compute_regularization_loss()
        total = outputs.loss + 0.01 * reg
        total.backward()
        # gate_logits should have gradients
        for key, block in wrapper.memory_blocks.items():
            assert block.gate_logits.grad is not None, (
                f"gate_logits in block '{key}' has no gradient after backward"
            )


# ---------------------------------------------------------------------------
# CASFTrainer optimizer integration
# ---------------------------------------------------------------------------

class TestTrainerOptimizerSMF:
    def _make_trainer(self) -> CASFTrainer:
        cfg = _make_smf_cfg()
        backbone = build_synthetic_model()
        tokenizer = SyntheticTokenizer()
        wrapper = SMFModelWrapper(backbone, cfg)
        registry = MemoryRegistry()
        return CASFTrainer(wrapper, tokenizer, cfg, registry)

    def test_optimizer_param_ids_are_subset_of_memory_params(self):
        trainer = self._make_trainer()
        optimizer_param_ids = {
            id(p) for group in trainer.optimizer.param_groups for p in group["params"]
        }
        memory_param_ids = {id(p) for p in trainer.model.smf_parameters()}
        assert optimizer_param_ids == memory_param_ids, (
            "Optimizer should contain exactly the SMF memory parameters"
        )

    def test_optimizer_does_not_contain_backbone_params(self):
        trainer = self._make_trainer()
        optimizer_param_ids = {
            id(p) for group in trainer.optimizer.param_groups for p in group["params"]
        }
        for name, param in trainer.model.backbone.named_parameters():
            assert id(param) not in optimizer_param_ids, (
                f"Backbone parameter '{name}' should not be in the optimizer"
            )

    def test_non_smf_method_uses_all_params(self):
        cfg = TrainConfig(model_name="synthetic-local-model", method="full_ft")
        backbone = build_synthetic_model()
        tokenizer = SyntheticTokenizer()
        registry = MemoryRegistry()
        trainer = CASFTrainer(backbone, tokenizer, cfg, registry)
        optimizer_param_ids = {
            id(p) for group in trainer.optimizer.param_groups for p in group["params"]
        }
        all_param_ids = {id(p) for p in backbone.parameters()}
        assert optimizer_param_ids == all_param_ids
