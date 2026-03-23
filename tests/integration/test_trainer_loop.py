from casf_dataset_api import MemoryRegistry
from synthetic_backend import SyntheticTemporalDataset, SyntheticTokenizer, build_synthetic_model
from train_config import TrainConfig
from trainer import CASFTrainer


EXPECTED_RESULT_KEYS = {
    "period",
    "train_loss_final",
    "train_loss_curve",
    "n_passages_trained",
    "n_contradiction_passages",
    "train_duration_sec",
}


def build_config() -> TrainConfig:
    return TrainConfig.make_config(
        run_id="integration-trainer",
        model_name="synthetic-local-model",
        method="full_ft",
        dataset_name="temporal_wiki",
        batch_size=1,
        grad_accum_steps=1,
        max_passages_per_period=2,
        log_every_n_steps=1,
    )


def load_synthetic_probe_dataset() -> SyntheticTemporalDataset:
    dataset = SyntheticTemporalDataset()
    dataset.load("changed")
    dataset.load("unchanged")
    return dataset


def test_casf_trainer_train_period_updates_registry_and_returns_stable_summary():
    trainer = CASFTrainer(
        build_synthetic_model(),
        SyntheticTokenizer(),
        build_config(),
        MemoryRegistry(),
    )
    dataset = load_synthetic_probe_dataset()

    result = trainer.train_period(dataset, "aug_sep")

    alpha_slot = trainer.registry.get_active("Alpha", "relation")
    gamma_slot = trainer.registry.get_active("Gamma", "relation")

    assert set(result) == EXPECTED_RESULT_KEYS
    assert result["period"] == "aug_sep"
    assert isinstance(result["train_loss_final"], float)
    assert all(
        isinstance(step, int) and isinstance(loss, float)
        for step, loss in result["train_loss_curve"]
    )
    assert result["n_passages_trained"] == 2
    assert result["n_contradiction_passages"] == 2
    assert result["train_duration_sec"] >= 0
    assert len(trainer.registry) == 2
    assert alpha_slot is not None
    assert alpha_slot.value == "new value"
    assert gamma_slot is not None
    assert gamma_slot.value == "updated value"
