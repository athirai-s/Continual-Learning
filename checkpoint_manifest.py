import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CHECKPOINT_MANIFEST_FILENAME = "checkpoint_manifest.json"
CHECKPOINT_MANIFEST_SCHEMA_VERSION = 1


class CheckpointManifestError(RuntimeError):
    pass


@dataclass(frozen=True)
class CheckpointManifest:
    schema_version: int
    model_name: str
    training_plan: list[str]
    resume_compatibility: dict[str, Any]
    dataset_identity: dict[str, Any]
    required_files: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_name": self.model_name,
            "training_plan": self.training_plan,
            "resume_compatibility": self.resume_compatibility,
            "dataset_identity": self.dataset_identity,
            "required_files": self.required_files,
        }


def manifest_path(checkpoint_path: str | Path) -> Path:
    return Path(checkpoint_path) / CHECKPOINT_MANIFEST_FILENAME


def write_checkpoint_manifest(
    checkpoint_path: str | Path,
    *,
    model_name: str,
    training_plan: list[str],
    resume_compatibility: dict[str, Any],
    dataset_identity: dict[str, Any],
) -> Path:
    checkpoint_path = Path(checkpoint_path)
    required_files = sorted(
        path.name for path in checkpoint_path.iterdir() if path.is_file()
    )
    if CHECKPOINT_MANIFEST_FILENAME not in required_files:
        required_files.append(CHECKPOINT_MANIFEST_FILENAME)

    manifest = CheckpointManifest(
        schema_version=CHECKPOINT_MANIFEST_SCHEMA_VERSION,
        model_name=model_name,
        training_plan=training_plan,
        resume_compatibility=resume_compatibility,
        dataset_identity=dataset_identity,
        required_files=required_files,
    )
    path = manifest_path(checkpoint_path)
    with open(path, "w") as f:
        json.dump(manifest.to_dict(), f, indent=2, sort_keys=True)
    return path


def load_checkpoint_manifest(checkpoint_path: str | Path) -> CheckpointManifest:
    path = manifest_path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint manifest: {path}")
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise CheckpointManifestError(f"Invalid checkpoint manifest JSON: {path}") from exc

    missing = {
        "schema_version",
        "model_name",
        "training_plan",
        "resume_compatibility",
        "dataset_identity",
        "required_files",
    } - set(data)
    if missing:
        raise CheckpointManifestError(
            f"Checkpoint manifest is missing required keys: {sorted(missing)}"
        )
    if data["schema_version"] != CHECKPOINT_MANIFEST_SCHEMA_VERSION:
        raise CheckpointManifestError(
            f"Unsupported checkpoint manifest schema_version={data['schema_version']}"
        )

    return CheckpointManifest(
        schema_version=data["schema_version"],
        model_name=data["model_name"],
        training_plan=list(data["training_plan"]),
        resume_compatibility=dict(data["resume_compatibility"]),
        dataset_identity=dict(data["dataset_identity"]),
        required_files=list(data["required_files"]),
    )


def validate_checkpoint_manifest(checkpoint_path: str | Path) -> CheckpointManifest:
    checkpoint_path = Path(checkpoint_path)
    manifest = load_checkpoint_manifest(checkpoint_path)
    missing_files = [
        name for name in manifest.required_files if not (checkpoint_path / name).exists()
    ]
    if missing_files:
        raise CheckpointManifestError(
            f"Checkpoint is missing required files: {sorted(missing_files)}"
        )
    return manifest
