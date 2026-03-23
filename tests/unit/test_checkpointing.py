import pytest

import checkpointing
from checkpointing import (
    CheckpointLockHeldError,
    CheckpointLockUnsupportedError,
    RunRootLock,
    prepare_run_root,
)


def test_run_root_lock_rejects_second_writer(tmp_path):
    run_root = tmp_path / "run-root"

    with RunRootLock(run_root):
        with pytest.raises(CheckpointLockHeldError, match="already locked"):
            with RunRootLock(run_root):
                pass


def test_prepare_run_root_removes_stale_tempdirs(tmp_path):
    stale_dir = tmp_path / "run-root" / "checkpoints" / ".tmp-ckpt-stale"
    stale_dir.mkdir(parents=True)
    (stale_dir / "partial.txt").write_text("partial")

    prepare_run_root(tmp_path / "run-root", cleanup_tempdirs=True)

    assert not stale_dir.exists()


def test_run_root_lock_fails_when_advisory_lock_is_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(checkpointing, "fcntl", None)

    with pytest.raises(CheckpointLockUnsupportedError, match="unsupported"):
        with RunRootLock(tmp_path / "run-root"):
            pass
