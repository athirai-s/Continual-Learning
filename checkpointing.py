import json
import os
import shutil
import tempfile
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import fcntl
except ImportError:  # pragma: no cover - exercised in tests via monkeypatch
    fcntl = None


CHECKPOINT_DIRNAME = "checkpoints"
LATEST_POINTER = "latest.json"
LOCK_FILENAME = ".checkpoint.lock"
TEMP_PREFIX = ".tmp-ckpt-"


class CheckpointError(RuntimeError):
    pass


class CheckpointLockUnsupportedError(CheckpointError):
    pass


class CheckpointLockHeldError(CheckpointError):
    pass


@dataclass(frozen=True)
class CheckpointPointer:
    schema_version: int
    checkpoint_id: str
    checkpoint_relpath: str
    last_period: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "checkpoint_id": self.checkpoint_id,
            "checkpoint_relpath": self.checkpoint_relpath,
            "last_period": self.last_period,
        }


class RunRootLock(AbstractContextManager["RunRootLock"]):
    def __init__(self, run_root: str | Path):
        self.run_root = Path(run_root)
        self.lock_path = self.run_root / LOCK_FILENAME
        self._fd: Any | None = None

    def __enter__(self) -> "RunRootLock":
        if fcntl is None:
            raise CheckpointLockUnsupportedError(
                "Advisory file locks are unavailable on this platform; "
                "Milestone 2 checkpointing is unsupported."
            )

        self.run_root.mkdir(parents=True, exist_ok=True)
        self.lock_path.touch(exist_ok=True)
        self._fd = open(self.lock_path, "r+")
        try:
            fcntl.flock(self._fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._fd.close()
            self._fd = None
            raise CheckpointLockHeldError(
                f"Checkpoint root is already locked: {self.run_root}"
            ) from exc
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._fd is None:
            return
        try:
            fcntl.flock(self._fd.fileno(), fcntl.LOCK_UN)
        finally:
            self._fd.close()
            self._fd = None


def checkpoint_root(run_root: str | Path) -> Path:
    return Path(run_root) / CHECKPOINT_DIRNAME


def latest_pointer_path(run_root: str | Path) -> Path:
    return Path(run_root) / LATEST_POINTER


def cleanup_stale_tempdirs(run_root: str | Path) -> list[Path]:
    ckpt_root = checkpoint_root(run_root)
    if not ckpt_root.exists():
        return []

    removed: list[Path] = []
    for child in ckpt_root.iterdir():
        if child.is_dir() and child.name.startswith(TEMP_PREFIX):
            shutil.rmtree(child)
            removed.append(child)
    return removed


def prepare_run_root(run_root: str | Path, *, cleanup_tempdirs: bool = False) -> None:
    run_root = Path(run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root(run_root).mkdir(parents=True, exist_ok=True)
    if cleanup_tempdirs:
        cleanup_stale_tempdirs(run_root)


def _next_checkpoint_id(run_root: str | Path) -> str:
    max_index = 0
    for path in checkpoint_root(run_root).iterdir():
        if not path.is_dir() or not path.name.startswith("ckpt-"):
            continue
        try:
            max_index = max(max_index, int(path.name.removeprefix("ckpt-")))
        except ValueError:
            continue
    return f"ckpt-{max_index + 1:06d}"


def create_checkpoint_tempdir(run_root: str | Path) -> Path:
    prepare_run_root(run_root, cleanup_tempdirs=True)
    return Path(tempfile.mkdtemp(prefix=TEMP_PREFIX, dir=checkpoint_root(run_root)))


def finalize_checkpoint(
    run_root: str | Path,
    temp_dir: str | Path,
    *,
    last_period: str,
) -> Path:
    run_root = Path(run_root)
    temp_dir = Path(temp_dir)
    checkpoint_id = _next_checkpoint_id(run_root)
    final_dir = checkpoint_root(run_root) / checkpoint_id
    temp_dir.rename(final_dir)

    pointer = CheckpointPointer(
        schema_version=1,
        checkpoint_id=checkpoint_id,
        checkpoint_relpath=str(final_dir.relative_to(run_root)),
        last_period=last_period,
    )
    latest_tmp = run_root / f".{LATEST_POINTER}.tmp"
    with open(latest_tmp, "w") as f:
        json.dump(pointer.to_dict(), f, indent=2)
    os.replace(latest_tmp, latest_pointer_path(run_root))
    return final_dir


def resolve_checkpoint_path(path: str | Path) -> Path:
    path = Path(path)
    latest_path = path / LATEST_POINTER
    if latest_path.exists():
        with open(latest_path, "r") as f:
            pointer = json.load(f)
        relpath = pointer["checkpoint_relpath"]
        return path / relpath
    return path


def read_latest_pointer(run_root: str | Path) -> CheckpointPointer:
    with open(latest_pointer_path(run_root), "r") as f:
        data = json.load(f)
    return CheckpointPointer(
        schema_version=data["schema_version"],
        checkpoint_id=data["checkpoint_id"],
        checkpoint_relpath=data["checkpoint_relpath"],
        last_period=data["last_period"],
    )
