from pathlib import Path

from artifacts.checkpointing import read_latest_pointer
from artifacts.run_artifacts import load_run_manifest
from training.train_config import TrainConfig
from training.train_runner import (
    build_synthetic_dataset,
    build_synthetic_model_and_tokenizer,
    run_training,
)


def build_config() -> TrainConfig:
    return TrainConfig.make_config(
        run_id="integration-run-layout",
        model_name="synthetic-local-model",
        method="full_ft",
        dataset_name="temporal_wiki",
        batch_size=1,
        grad_accum_steps=1,
        max_passages_per_period=2,
        log_every_n_steps=1,
    )


def test_run_root_has_stable_manifest_and_layout(tmp_path):
    run_training(
        build_config(),
        model_factory=build_synthetic_model_and_tokenizer,
        dataset_factory=build_synthetic_dataset,
        checkpoint_dir=str(tmp_path),
        training_units=["aug_sep"],
    )

    run_root = tmp_path / "integration-run-layout"
    manifest = load_run_manifest(run_root)

    assert (run_root / "run_manifest.json").exists()
    assert (run_root / "metrics").is_dir()
    assert (run_root / "periods").is_dir()
    assert (run_root / "periods" / "aug_sep").is_dir()
    assert manifest.run_id == "integration-run-layout"
    assert manifest.training_plan == ["aug_sep"]
    assert manifest.artifact_layout == {
        "checkpoints_dir": "checkpoints",
        "metrics_dir": "metrics",
        "periods_dir": "periods",
    }
    assert manifest.period_artifacts["aug_sep"]["path"] == "periods/aug_sep"


def test_m2_checkpoint_paths_remain_valid_under_m3_run_layout(tmp_path):
    results = run_training(
        build_config(),
        model_factory=build_synthetic_model_and_tokenizer,
        dataset_factory=build_synthetic_dataset,
        checkpoint_dir=str(tmp_path),
        training_units=["aug_sep"],
    )

    run_root = tmp_path / "integration-run-layout"
    checkpoint_path = Path(results[0]["checkpoint_path"])
    latest_pointer = read_latest_pointer(run_root)

    assert checkpoint_path == run_root / latest_pointer.checkpoint_relpath
    assert checkpoint_path.parent == run_root / "checkpoints"
