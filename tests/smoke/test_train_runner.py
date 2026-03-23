import subprocess
import sys
from pathlib import Path

from checkpointing import read_latest_pointer
from train_config import TrainConfig
from train_runner import build_synthetic_dataset, build_synthetic_model_and_tokenizer, run_training


def test_run_training_smoke_with_synthetic_runtime(tmp_path):
    cfg = TrainConfig.make_config(
        run_id="synthetic-runner",
        model_name="synthetic-local-model",
        method="full_ft",
        dataset_name="temporal_wiki",
        batch_size=1,
        grad_accum_steps=1,
        max_passages_per_period=2,
        log_every_n_steps=1,
    )

    results = run_training(
        cfg,
        model_factory=build_synthetic_model_and_tokenizer,
        dataset_factory=build_synthetic_dataset,
        checkpoint_dir=str(tmp_path),
    )

    assert len(results) == 1
    checkpoint_path = Path(results[0]["checkpoint_path"])
    latest_pointer = read_latest_pointer(tmp_path / "synthetic-runner")
    assert (tmp_path / "synthetic-runner_config.json").exists()
    assert (checkpoint_path / "config.json").exists()
    assert (checkpoint_path / "synthetic_tokenizer.json").exists()
    assert (checkpoint_path / "memory_registry.json").exists()
    assert checkpoint_path.parent == tmp_path / "synthetic-runner" / "checkpoints"
    assert latest_pointer.checkpoint_id == checkpoint_path.name


def test_main_cli_synthetic_mode_smoke(repo_root, tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "main.py",
            "--mode",
            "synthetic",
            "--run-id",
            "cli-smoke",
            "--checkpoint-dir",
            str(tmp_path),
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Training config:" in result.stdout
    assert (tmp_path / "cli-smoke_config.json").exists()
    latest_pointer = read_latest_pointer(tmp_path / "cli-smoke")
    checkpoint_path = tmp_path / "cli-smoke" / latest_pointer.checkpoint_relpath
    assert (checkpoint_path / "config.json").exists()


def test_run_training_checkpoint_cadence_uses_optimizer_steps(tmp_path):
    cfg = TrainConfig.make_config(
        run_id="cadence-runner",
        model_name="synthetic-local-model",
        method="full_ft",
        dataset_name="temporal_wiki",
        batch_size=1,
        grad_accum_steps=2,
        max_passages_per_period=3,
        log_every_n_steps=1,
        checkpoint_every_n_optimizer_steps=1,
    )

    results = run_training(
        cfg,
        model_factory=build_synthetic_model_and_tokenizer,
        dataset_factory=build_synthetic_dataset,
        checkpoint_dir=str(tmp_path),
    )

    checkpoint_root = tmp_path / "cadence-runner" / "checkpoints"
    checkpoint_dirs = sorted(path.name for path in checkpoint_root.iterdir() if path.is_dir())

    assert len(results) == 1
    assert results[0]["micro_steps_total"] == 3
    assert results[0]["optimizer_steps_total"] == 2
    assert len(checkpoint_dirs) == 2
