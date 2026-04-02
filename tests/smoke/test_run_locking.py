import pytest

from artifacts.checkpointing import CheckpointLockHeldError, RunRootLock
from training.train_config import TrainConfig
from training.train_runner import (
    build_synthetic_dataset,
    build_synthetic_model_and_tokenizer,
    run_training,
)


def build_config() -> TrainConfig:
    return TrainConfig.make_config(
        run_id="run-lock-smoke",
        model_name="synthetic-local-model",
        method="full_ft",
        dataset_name="temporal_wiki",
        batch_size=1,
        grad_accum_steps=1,
        max_passages_per_period=2,
        log_every_n_steps=1,
    )


def test_run_training_fails_fast_when_run_root_is_already_locked(tmp_path):
    run_root = tmp_path / "run-lock-smoke"
    with RunRootLock(run_root):
        with pytest.raises(CheckpointLockHeldError, match="already locked"):
            run_training(
                build_config(),
                model_factory=build_synthetic_model_and_tokenizer,
                dataset_factory=build_synthetic_dataset,
                checkpoint_dir=str(tmp_path),
            )
