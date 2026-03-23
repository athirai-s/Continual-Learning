import pytest

from train_config import TrainConfig


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
    ],
)
def test_validate_rejects_invalid_values(updates, message):
    cfg = TrainConfig(model_name="synthetic-local-model", method="full_ft")
    for field_name, value in updates.items():
        setattr(cfg, field_name, value)

    with pytest.raises(ValueError, match=message):
        cfg.validate()
