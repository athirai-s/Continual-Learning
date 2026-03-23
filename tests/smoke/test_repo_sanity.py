def test_repo_modules_import():
    import casf_dataset_api  # noqa: F401
    import passage_filter  # noqa: F401
    import train_config  # noqa: F401
    import trainer  # noqa: F401


def test_shared_test_fixture_loads(repo_root):
    assert (repo_root / "pyproject.toml").exists()
