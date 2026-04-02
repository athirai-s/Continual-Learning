import os
import subprocess
import sys

from artifacts.checkpointing import read_latest_pointer


def test_run_job_executes_synthetic_smoke_path(repo_root, tmp_path):
    env = os.environ.copy()
    env.update(
        {
            "CONTINUAL_LEARNING_REPO": str(repo_root),
            "CONTINUAL_LEARNING_PYTHON": sys.executable,
            "CONTINUAL_LEARNING_MODE": "synthetic",
            "CONTINUAL_LEARNING_MODEL_ID": "synthetic-local-model",
            "CONTINUAL_LEARNING_CHECKPOINT_DIR": str(tmp_path),
            "CONTINUAL_LEARNING_RUN_ID": "launcher-smoke",
            "CONTINUAL_LEARNING_SKIP_MODULES": "1",
            "CONTINUAL_LEARNING_SKIP_VENV": "1",
        }
    )

    result = subprocess.run(
        ["bash", "run_job.sh"],
        cwd=repo_root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    latest_pointer = read_latest_pointer(tmp_path / "launcher-smoke")
    checkpoint_path = tmp_path / "launcher-smoke" / latest_pointer.checkpoint_relpath
    assert "Training config:" in result.stdout
    assert (tmp_path / "launcher-smoke_config.json").exists()
    assert (checkpoint_path / "config.json").exists()
    assert (checkpoint_path / "synthetic_tokenizer.json").exists()
    assert (checkpoint_path / "memory_registry.json").exists()


def test_run_job_defaults_to_main_entrypoint(repo_root):
    script = (repo_root / "run_job.sh").read_text()

    assert "main.py" in script
    assert "3B_train.py" not in script
    assert 'CONTINUAL_LEARNING_MODE="${CONTINUAL_LEARNING_MODE:-real}"' in script


def test_legacy_launcher_is_marked_experimental(repo_root):
    legacy_path = repo_root / "experiments" / "legacy" / "3B_train.py"
    legacy_script = legacy_path.read_text()

    assert not (repo_root / "3B_train.py").exists()
    assert legacy_path.exists()
    assert "Experimental legacy script kept for reference only." in legacy_script
    assert "supported training path" in legacy_script
    assert "through `main.py`" in legacy_script
