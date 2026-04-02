def test_repo_modules_import():
    import casf_dataset_api  # noqa: F401
    import artifacts.checkpointing  # noqa: F401
    import training.passage_filter  # noqa: F401
    import training.train_config  # noqa: F401
    import training.trainer  # noqa: F401


def test_shared_test_fixture_loads(repo_root):
    assert (repo_root / "pyproject.toml").exists()
