"""Unit tests for train_runner model factory helpers (Phase 5).

Covers:
    _wrap_model_for_method — returns the right wrapper type for each method
    build_synthetic_model_and_tokenizer — returns a wrapped model when cfg.method
        requires it
"""

import pytest
import torch

from training.casm_model import CASMModelWrapper
from training.smf_model import SMFModelWrapper
from training.synthetic_backend import SyntheticTokenizer, build_synthetic_model
from training.train_config import TrainConfig
from training.train_runner import (
    _wrap_model_for_method,
    build_synthetic_model_and_tokenizer,
    build_real_model_and_tokenizer,
    load_synthetic_model_and_tokenizer,
)


def _smf_cfg(**overrides) -> TrainConfig:
    defaults = dict(
        model_name="synthetic-local-model",
        method="smf",
        smf_memory_size=8,
        smf_sparsity_ratio=0.5,
        smf_update_layers=[0],
        smf_regularization_weight=0.0,
        smf_freeze_backbone=True,
    )
    defaults.update(overrides)
    return TrainConfig(**defaults)


def _casm_cfg(**overrides) -> TrainConfig:
    defaults = dict(
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
    defaults.update(overrides)
    return TrainConfig(**defaults)


def _lora_cfg(**overrides) -> TrainConfig:
    defaults = dict(
        model_name="synthetic-local-model",
        method="lora",
        lora_r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        lora_target_modules=["c_attn", "c_proj"],
    )
    defaults.update(overrides)
    return TrainConfig(**defaults)


def _full_ft_cfg() -> TrainConfig:
    return TrainConfig(model_name="synthetic-local-model", method="full_ft")


# ---------------------------------------------------------------------------
# _wrap_model_for_method
# ---------------------------------------------------------------------------


class TestWrapModelForMethod:
    def test_smf_returns_smf_wrapper(self):
        backbone = build_synthetic_model()
        cfg = _smf_cfg()
        result = _wrap_model_for_method(backbone, cfg)
        assert isinstance(result, SMFModelWrapper)

    def test_casm_returns_casm_wrapper(self):
        backbone = build_synthetic_model()
        cfg = _casm_cfg()
        result = _wrap_model_for_method(backbone, cfg)
        assert isinstance(result, CASMModelWrapper)

    def test_full_ft_returns_bare_backbone(self):
        backbone = build_synthetic_model()
        cfg = _full_ft_cfg()
        result = _wrap_model_for_method(backbone, cfg)
        assert result is backbone

    def test_lora_returns_peft_model(self):
        from peft import PeftModel
        backbone = build_synthetic_model()
        cfg = _lora_cfg()
        result = _wrap_model_for_method(backbone, cfg)
        assert isinstance(result, PeftModel)

    def test_lora_freezes_base_model(self):
        backbone = build_synthetic_model()
        cfg = _lora_cfg()
        result = _wrap_model_for_method(backbone, cfg)
        base_params = [p for n, p in result.named_parameters() if "lora_" not in n]
        for param in base_params:
            assert not param.requires_grad

    def test_lora_has_trainable_adapter_params(self):
        backbone = build_synthetic_model()
        cfg = _lora_cfg()
        result = _wrap_model_for_method(backbone, cfg)
        trainable = [p for p in result.parameters() if p.requires_grad]
        assert len(trainable) > 0

    def test_smf_wrapper_freezes_backbone(self):
        backbone = build_synthetic_model()
        cfg = _smf_cfg()
        wrapper = _wrap_model_for_method(backbone, cfg)
        for param in wrapper.backbone.parameters():
            assert not param.requires_grad

    def test_casm_wrapper_freezes_backbone(self):
        backbone = build_synthetic_model()
        cfg = _casm_cfg()
        wrapper = _wrap_model_for_method(backbone, cfg)
        for param in wrapper.backbone.parameters():
            assert not param.requires_grad

    def test_casm_wrapper_has_correct_num_slots(self):
        backbone = build_synthetic_model()
        cfg = _casm_cfg(casm_num_slots=4)
        wrapper = _wrap_model_for_method(backbone, cfg)
        assert len(wrapper._active_slot_ids) == 4


# ---------------------------------------------------------------------------
# build_synthetic_model_and_tokenizer
# ---------------------------------------------------------------------------


class TestBuildSyntheticModelAndTokenizer:
    def test_lora_method_returns_peft_model(self):
        from peft import PeftModel

        cfg = _lora_cfg()
        model, tokenizer = build_synthetic_model_and_tokenizer(cfg)
        assert isinstance(model, PeftModel)
        assert isinstance(tokenizer, SyntheticTokenizer)

    def test_smf_method_returns_smf_wrapper(self):
        cfg = _smf_cfg()
        model, tokenizer = build_synthetic_model_and_tokenizer(cfg)
        assert isinstance(model, SMFModelWrapper)
        assert isinstance(tokenizer, SyntheticTokenizer)

    def test_casm_method_returns_casm_wrapper(self):
        cfg = _casm_cfg()
        model, tokenizer = build_synthetic_model_and_tokenizer(cfg)
        assert isinstance(model, CASMModelWrapper)
        assert isinstance(tokenizer, SyntheticTokenizer)

    def test_full_ft_returns_bare_model(self):
        cfg = _full_ft_cfg()
        model, tokenizer = build_synthetic_model_and_tokenizer(cfg)
        assert not isinstance(model, (SMFModelWrapper, CASMModelWrapper))


# ---------------------------------------------------------------------------
# load_synthetic_model_and_tokenizer
# ---------------------------------------------------------------------------


class TestLoadSyntheticModelAndTokenizer:
    def test_lora_load_returns_trainable_peft_model(self, tmp_path):
        from peft import PeftModel

        cfg = _lora_cfg()
        tokenizer = SyntheticTokenizer()
        backbone = build_synthetic_model()
        model = _wrap_model_for_method(backbone, cfg)
        model.save_pretrained(str(tmp_path))
        tokenizer.save_pretrained(tmp_path)

        reloaded, tok = load_synthetic_model_and_tokenizer(cfg, str(tmp_path))
        assert isinstance(reloaded, PeftModel)
        assert any(param.requires_grad for param in reloaded.parameters())

    def test_lora_load_requires_adapter_checkpoint(self, tmp_path):
        cfg = _lora_cfg()
        tokenizer = SyntheticTokenizer()
        tokenizer.save_pretrained(tmp_path)

        with pytest.raises(FileNotFoundError, match="adapter_config.json"):
            load_synthetic_model_and_tokenizer(cfg, str(tmp_path))

    def test_smf_load_returns_smf_wrapper(self, tmp_path):
        cfg = _smf_cfg()
        # Save a checkpoint first
        from training.smf_model import SMFModelWrapper as SMF
        tokenizer = SyntheticTokenizer()
        backbone = build_synthetic_model()
        wrapper = SMF(backbone, cfg)
        wrapper.save_pretrained(str(tmp_path))
        tokenizer.save_pretrained(tmp_path)

        model, tok = load_synthetic_model_and_tokenizer(cfg, str(tmp_path))
        assert isinstance(model, SMF)

    def test_casm_load_returns_casm_wrapper(self, tmp_path):
        cfg = _casm_cfg()
        tokenizer = SyntheticTokenizer()
        backbone = build_synthetic_model()
        wrapper = CASMModelWrapper(backbone, cfg)
        wrapper.save_pretrained(str(tmp_path))
        tokenizer.save_pretrained(tmp_path)

        model, tok = load_synthetic_model_and_tokenizer(cfg, str(tmp_path))
        assert isinstance(model, CASMModelWrapper)

    def test_full_ft_load_returns_bare_model(self, tmp_path):
        cfg = _full_ft_cfg()
        tokenizer = SyntheticTokenizer()
        backbone = build_synthetic_model()
        backbone.save_pretrained(str(tmp_path))
        tokenizer.save_pretrained(tmp_path)

        model, tok = load_synthetic_model_and_tokenizer(cfg, str(tmp_path))
        assert not isinstance(model, (SMFModelWrapper, CASMModelWrapper))


def test_build_real_model_uses_requested_precision_and_training_memory_settings(monkeypatch):
    calls = {}

    class DummyConfig:
        def __init__(self):
            self.use_cache = True

    class DummyModel:
        def __init__(self):
            self.config = DummyConfig()
            self.gradient_checkpointing_enabled = False

        def gradient_checkpointing_enable(self):
            self.gradient_checkpointing_enabled = True

    dummy_model = DummyModel()

    def fake_tokenizer_from_pretrained(path):
        calls["tokenizer_path"] = path
        return object()

    def fake_model_from_pretrained(path, **kwargs):
        calls["model_path"] = path
        calls["kwargs"] = kwargs
        return dummy_model

    monkeypatch.setattr(
        "training.train_runner.AutoTokenizer.from_pretrained",
        fake_tokenizer_from_pretrained,
    )
    monkeypatch.setattr(
        "training.train_runner.AutoModelForCausalLM.from_pretrained",
        fake_model_from_pretrained,
    )

    cfg = TrainConfig(
        model_name="real-model-id",
        method="full_ft",
        precision="bfloat16",
    )
    model, tokenizer = build_real_model_and_tokenizer(cfg)

    assert model is dummy_model
    assert calls["tokenizer_path"] == "real-model-id"
    assert calls["model_path"] == "real-model-id"
    assert calls["kwargs"]["torch_dtype"] == torch.bfloat16
    assert calls["kwargs"]["low_cpu_mem_usage"] is True
    assert dummy_model.config.use_cache is False
    assert dummy_model.gradient_checkpointing_enabled is True
