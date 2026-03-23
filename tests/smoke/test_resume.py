import pytest

from casf_dataset_api import MemoryRegistry
from synthetic_backend import SyntheticTemporalDataset, SyntheticTokenizer, build_synthetic_model
from train_config import TrainConfig
from trainer import CASFTrainer


def build_config() -> TrainConfig:
    return TrainConfig.make_config(
        run_id="resume-smoke",
        model_name="synthetic-local-model",
        method="full_ft",
        dataset_name="temporal_wiki",
        batch_size=1,
        grad_accum_steps=1,
        max_passages_per_period=2,
        log_every_n_steps=1,
    )


def build_trainer() -> CASFTrainer:
    return CASFTrainer(
        build_synthetic_model(),
        SyntheticTokenizer(),
        build_config(),
        MemoryRegistry(),
    )


def load_synthetic_probe_dataset() -> SyntheticTemporalDataset:
    dataset = SyntheticTemporalDataset()
    dataset.load("changed")
    dataset.load("unchanged")
    return dataset


def test_metadata_resume_restores_registry_and_last_period_and_can_continue(tmp_path):
    first_trainer = build_trainer()
    dataset = load_synthetic_probe_dataset()
    run_root = tmp_path / "resume-smoke"

    first_trainer.train_period(dataset, "aug_sep")
    checkpoint_path = first_trainer.checkpoint("aug_sep", str(run_root))

    resumed_trainer = build_trainer()
    restored_period = resumed_trainer.resume(str(run_root))
    alpha_slot = resumed_trainer.registry.get_active("Alpha", "relation")

    continued_result = resumed_trainer.train_period(load_synthetic_probe_dataset(), "sep_oct")

    assert restored_period == "aug_sep"
    assert len(resumed_trainer.registry) >= 2
    assert alpha_slot is not None
    assert alpha_slot.value == "new value"
    assert continued_result["period"] == "sep_oct"
    assert continued_result["n_passages_trained"] == 2


def test_metadata_resume_fails_when_last_period_metadata_is_missing(tmp_path):
    checkpoint_path = tmp_path / "missing-metadata"
    checkpoint_path.mkdir(parents=True)
    MemoryRegistry().save(checkpoint_path / "memory_registry.json")

    trainer = build_trainer()

    with pytest.raises(FileNotFoundError, match="Missing checkpoint metadata"):
        trainer.resume(checkpoint_path)
