from run_artifacts import load_run_manifest
from train_config import TrainConfig
from train_runner import build_synthetic_dataset, build_synthetic_model_and_tokenizer, run_training


def build_config() -> TrainConfig:
    return TrainConfig.make_config(
        run_id="integration-run-metadata",
        model_name="synthetic-local-model",
        method="full_ft",
        dataset_name="temporal_wiki",
        batch_size=1,
        grad_accum_steps=1,
        max_passages_per_period=2,
        log_every_n_steps=1,
        seed=17,
    )


def test_run_manifest_persists_required_reproducibility_metadata(tmp_path):
    run_training(
        build_config(),
        model_factory=build_synthetic_model_and_tokenizer,
        dataset_factory=build_synthetic_dataset,
        checkpoint_dir=str(tmp_path),
        training_units=["aug_sep"],
    )

    manifest = load_run_manifest(tmp_path / "integration-run-metadata")

    assert manifest.reproducibility["seed"] == 17
    assert isinstance(manifest.reproducibility["git_commit"], str)
    assert isinstance(manifest.reproducibility["git_dirty"], bool)
    assert isinstance(manifest.reproducibility["python_version"], str)
    assert isinstance(manifest.reproducibility["torch_version"], str)
    assert isinstance(manifest.reproducibility["transformers_version"], str)
    assert manifest.reproducibility["dataset_selection"] == {"dataset_name": "temporal_wiki"}
    assert manifest.reproducibility["model_id"] == "synthetic-local-model"
    assert manifest.reproducibility["training_plan"] == ["aug_sep"]
    assert manifest.reproducibility["checkpoint_manifest_schema_version"] == 1
