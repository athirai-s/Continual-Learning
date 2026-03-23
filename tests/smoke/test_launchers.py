import os
import subprocess
import sys


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

    checkpoint_path = tmp_path / "launcher-smoke" / "aug_sep"
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
    legacy_script = (repo_root / "3B_train.py").read_text()

    assert "Experimental legacy script kept for reference only." in legacy_script
    assert "supported training path" in legacy_script
    assert "through `main.py`" in legacy_script
