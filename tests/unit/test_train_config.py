import pytest

from training.train_config import TrainConfig


def test_make_config_returns_valid_train_config():
    cfg = TrainConfig.make_config(
        run_id="unit-valid",
        model_name="synthetic-local-model",
        method="full_ft",
        dataset_name="temporal_wiki",
        batch_size=1,
        grad_accum_steps=1,
        max_passages_per_period=2,
        log_every_n_steps=1,
    )

    assert cfg.run_id == "unit-valid"
    assert cfg.model_name == "synthetic-local-model"
    assert cfg.dataset_name == "temporal_wiki"
    assert cfg.checkpoint_every_n_optimizer_steps is None
    assert cfg.seed == 0


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"method": "unknown"}, "Unsupported method"),
        ({"precision": "float32"}, "Unsupported precision"),
        ({"dataset_name": "made_up"}, "Unsupported dataset_name"),
        ({"learning_rate": 0}, "learning_rate must be > 0"),
        ({"batch_size": 0}, "batch_size must be > 0"),
        ({"grad_accum_steps": 0}, "grad_accum_steps must be > 0"),
        ({"epochs_per_period": 0}, "epochs_per_period must be > 0"),
        ({"min_passage_length": -1}, "min_passage_length must be >= 0"),
        (
            {"contradiction_batch_frac": 1.5},
            "contradiction_batch_frac must be between 0 and 1",
        ),
        ({"log_every_n_steps": 0}, "log_every_n_steps must be > 0"),
        (
            {"checkpoint_every_n_optimizer_steps": 0},
            "checkpoint_every_n_optimizer_steps must be > 0",
        ),
        ({"seed": -1}, "seed must be >= 0"),
    ],
)
def test_validate_rejects_invalid_values(updates, message):
    cfg = TrainConfig(model_name="synthetic-local-model", method="full_ft")
    for field_name, value in updates.items():
        setattr(cfg, field_name, value)

    with pytest.raises(ValueError, match=message):
        cfg.validate()


# ---------------------------------------------------------------------------
# SMF config tests
# ---------------------------------------------------------------------------

def _valid_smf_cfg(**overrides) -> TrainConfig:
    defaults = dict(
        model_name="synthetic-local-model",
        method="smf",
        smf_memory_size=64,
        smf_sparsity_ratio=0.1,
        smf_update_layers=[0, 1],
        smf_regularization_weight=0.01,
        smf_freeze_backbone=True,
    )
    defaults.update(overrides)
    return TrainConfig(**defaults)


def test_valid_smf_config_passes():
    cfg = _valid_smf_cfg()
    cfg.validate()  # must not raise


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"smf_memory_size": None}, "smf_memory_size must be > 0"),
        ({"smf_memory_size": 0}, "smf_memory_size must be > 0"),
        ({"smf_memory_size": -1}, "smf_memory_size must be > 0"),
        ({"smf_sparsity_ratio": None}, "smf_sparsity_ratio must be in"),
        ({"smf_sparsity_ratio": 0.0}, "smf_sparsity_ratio must be in"),
        ({"smf_sparsity_ratio": 1.1}, "smf_sparsity_ratio must be in"),
        ({"smf_update_layers": None}, "smf_update_layers must be non-empty"),
        ({"smf_update_layers": []}, "smf_update_layers must be non-empty"),
        ({"smf_regularization_weight": -0.1}, "smf_regularization_weight must be >= 0"),
        ({"smf_freeze_backbone": False}, "smf_freeze_backbone must be True"),
    ],
)
def test_smf_config_rejects_invalid_values(overrides, message):
    cfg = _valid_smf_cfg(**overrides)
    with pytest.raises(ValueError, match=message):
        cfg.validate()


def test_smf_sparsity_ratio_boundary_one_is_valid():
    cfg = _valid_smf_cfg(smf_sparsity_ratio=1.0)
    cfg.validate()


def test_smf_regularization_weight_zero_is_valid():
    cfg = _valid_smf_cfg(smf_regularization_weight=0.0)
    cfg.validate()


# ---------------------------------------------------------------------------
# CASM config tests
# ---------------------------------------------------------------------------

def _valid_casm_cfg(**overrides) -> TrainConfig:
    defaults = dict(
        model_name="synthetic-local-model",
        method="casm",
        casm_num_slots=4,
        casm_router_hidden_size=128,
        casm_top_k=2,
        casm_router_temperature=1.0,
        casm_sparsity_weight=0.01,
        casm_overlap_weight=0.01,
        casm_branch_on_contradiction=True,
    )
    defaults.update(overrides)
    return TrainConfig(**defaults)


def test_valid_casm_config_passes():
    cfg = _valid_casm_cfg()
    cfg.validate()  # must not raise


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"casm_num_slots": None}, "casm_num_slots must be >= 1"),
        ({"casm_num_slots": 0}, "casm_num_slots must be >= 1"),
        ({"casm_top_k": None}, "casm_top_k must be >= 1"),
        ({"casm_top_k": 0}, "casm_top_k must be >= 1"),
        ({"casm_top_k": 5}, "casm_top_k must be <= casm_num_slots"),
        ({"casm_router_hidden_size": None}, "casm_router_hidden_size must be > 0"),
        ({"casm_router_hidden_size": 0}, "casm_router_hidden_size must be > 0"),
        ({"casm_router_hidden_size": -1}, "casm_router_hidden_size must be > 0"),
        ({"casm_sparsity_weight": -0.1}, "casm_sparsity_weight must be >= 0"),
        ({"casm_overlap_weight": -0.1}, "casm_overlap_weight must be >= 0"),
    ],
)
def test_casm_config_rejects_invalid_values(overrides, message):
    cfg = _valid_casm_cfg(**overrides)
    with pytest.raises(ValueError, match=message):
        cfg.validate()


def test_casm_top_k_equals_num_slots_is_valid():
    cfg = _valid_casm_cfg(casm_num_slots=3, casm_top_k=3)
    cfg.validate()


def test_casm_zero_weights_are_valid():
    cfg = _valid_casm_cfg(casm_sparsity_weight=0.0, casm_overlap_weight=0.0)
    cfg.validate()


# ---------------------------------------------------------------------------
# from_dict round-trip tests
# ---------------------------------------------------------------------------

def test_from_dict_round_trips_full_ft():
    cfg = TrainConfig(model_name="m", method="full_ft")
    restored = TrainConfig.from_dict(cfg.to_dict())
    assert restored == cfg


def test_from_dict_round_trips_smf():
    cfg = _valid_smf_cfg()
    restored = TrainConfig.from_dict(cfg.to_dict())
    assert restored == cfg


def test_from_dict_round_trips_casm():
    cfg = _valid_casm_cfg()
    restored = TrainConfig.from_dict(cfg.to_dict())
    assert restored == cfg


def test_from_dict_ignores_unknown_keys():
    cfg = TrainConfig(model_name="m", method="full_ft")
    data = cfg.to_dict()
    data["future_field"] = "some_value"
    restored = TrainConfig.from_dict(data)
    assert restored == cfg
