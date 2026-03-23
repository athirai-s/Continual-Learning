import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import transformers

from checkpoint_manifest import CHECKPOINT_MANIFEST_SCHEMA_VERSION
from train_config import TrainConfig


RUN_MANIFEST_FILENAME = "run_manifest.json"
RUN_MANIFEST_SCHEMA_VERSION = 1
METRICS_DIRNAME = "metrics"
PERIODS_DIRNAME = "periods"


@dataclass(frozen=True)
class RunManifest:
    schema_version: int
    run_id: str
    model_name: str
    dataset_name: str
    training_plan: list[str]
    artifact_layout: dict[str, str]
    period_artifacts: dict[str, dict[str, str]]
    reproducibility: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "model_name": self.model_name,
            "dataset_name": self.dataset_name,
            "training_plan": self.training_plan,
            "artifact_layout": self.artifact_layout,
            "period_artifacts": self.period_artifacts,
            "reproducibility": self.reproducibility,
        }


def metrics_root(run_root: str | Path) -> Path:
    return Path(run_root) / METRICS_DIRNAME


def periods_root(run_root: str | Path) -> Path:
    return Path(run_root) / PERIODS_DIRNAME


def period_root(run_root: str | Path, unit: str) -> Path:
    return periods_root(run_root) / unit


def run_manifest_path(run_root: str | Path) -> Path:
    return Path(run_root) / RUN_MANIFEST_FILENAME


def ensure_run_layout(run_root: str | Path, training_plan: list[str]) -> None:
    run_root = Path(run_root)
    metrics_root(run_root).mkdir(parents=True, exist_ok=True)
    periods_root(run_root).mkdir(parents=True, exist_ok=True)
    for unit in training_plan:
        period_root(run_root, unit).mkdir(parents=True, exist_ok=True)


def write_run_manifest(
    run_root: str | Path,
    cfg: TrainConfig,
    training_plan: list[str],
    reproducibility: dict[str, Any] | None = None,
) -> Path:
    run_root = Path(run_root)
    manifest = RunManifest(
        schema_version=RUN_MANIFEST_SCHEMA_VERSION,
        run_id=cfg.run_id,
        model_name=cfg.model_name,
        dataset_name=cfg.dataset_name,
        training_plan=list(training_plan),
        artifact_layout={
            "checkpoints_dir": "checkpoints",
            "metrics_dir": METRICS_DIRNAME,
            "periods_dir": PERIODS_DIRNAME,
        },
        period_artifacts={
            unit: {"path": str(period_root(run_root, unit).relative_to(run_root))}
            for unit in training_plan
        },
        reproducibility=reproducibility or {},
    )
    path = run_manifest_path(run_root)
    with open(path, "w") as f:
        json.dump(manifest.to_dict(), f, indent=2, sort_keys=True)
    return path


def load_run_manifest(run_root: str | Path) -> RunManifest:
    path = run_manifest_path(run_root)
    with open(path, "r") as f:
        data = json.load(f)

    return RunManifest(
        schema_version=data["schema_version"],
        run_id=data["run_id"],
        model_name=data["model_name"],
        dataset_name=data["dataset_name"],
        training_plan=list(data["training_plan"]),
        artifact_layout=dict(data["artifact_layout"]),
        period_artifacts={
            unit: dict(artifact_info)
            for unit, artifact_info in data["period_artifacts"].items()
        },
        reproducibility=dict(data.get("reproducibility", {})),
    )


def collect_reproducibility_metadata(
    cfg: TrainConfig,
    training_plan: list[str],
    *,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    repo_root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parent
    git_commit = None
    git_dirty = None

    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        git_dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        git_commit = None
        git_dirty = None

    return {
        "seed": cfg.seed,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "dataset_selection": {
            "dataset_name": cfg.dataset_name,
        },
        "model_id": cfg.model_name,
        "training_plan": list(training_plan),
        "checkpoint_manifest_schema_version": CHECKPOINT_MANIFEST_SCHEMA_VERSION,
    }
