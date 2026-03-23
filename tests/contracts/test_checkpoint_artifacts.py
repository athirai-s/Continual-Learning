from pathlib import Path

from train_config import TrainConfig
from train_runner import build_synthetic_dataset, build_synthetic_model_and_tokenizer, run_training


EXPECTED_CHECKPOINT_FILES = {
    "config.json",
    "synthetic_tokenizer.json",
    "memory_registry.json",
    "train_config.json",
    "last_period.txt",
}
MODEL_WEIGHT_FILES = {"model.safetensors", "pytorch_model.bin"}


def build_config() -> TrainConfig:
    return TrainConfig.make_config(
        run_id="integration-artifacts",
        model_name="synthetic-local-model",
        method="full_ft",
        dataset_name="temporal_wiki",
        batch_size=1,
        grad_accum_steps=1,
        max_passages_per_period=2,
        log_every_n_steps=1,
    )


def test_synthetic_training_writes_expected_checkpoint_artifacts(tmp_path):
    results = run_training(
        build_config(),
        model_factory=build_synthetic_model_and_tokenizer,
        dataset_factory=build_synthetic_dataset,
        checkpoint_dir=str(tmp_path),
    )

    checkpoint_path = Path(results[0]["checkpoint_path"])
    checkpoint_files = {path.name for path in checkpoint_path.iterdir()}

    assert (tmp_path / "integration-artifacts_config.json").exists()
    assert EXPECTED_CHECKPOINT_FILES <= checkpoint_files
    assert checkpoint_files & MODEL_WEIGHT_FILES
    assert (checkpoint_path / "last_period.txt").read_text().strip() == "aug_sep"
