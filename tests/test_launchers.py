def test_run_job_uses_supported_main_entrypoint(repo_root):
    script = (repo_root / "run_job.sh").read_text()

    assert "python main.py" in script
    assert "--mode real" in script
    assert "3B_train.py" not in script


def test_legacy_launcher_is_marked_experimental(repo_root):
    legacy_script = (repo_root / "3B_train.py").read_text()

    assert "Experimental legacy script kept for reference only." in legacy_script
    assert "supported training path" in legacy_script
    assert "through `main.py`" in legacy_script
