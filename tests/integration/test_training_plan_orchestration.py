from artifacts.checkpointing import read_latest_pointer
from artifacts.run_artifacts import load_run_manifest
from training.train_config import TrainConfig
from training.train_runner import (
    build_synthetic_dataset,
    build_synthetic_model_and_tokenizer,
    run_training,
)
from training.training_plan import DEFAULT_TEMPORAL_WIKI_PLAN


def build_config() -> TrainConfig:
    return TrainConfig.make_config(
        run_id="integration-training-plan",
        model_name="synthetic-local-model",
        method="full_ft",
        dataset_name="temporal_wiki",
        batch_size=1,
        grad_accum_steps=1,
        max_passages_per_period=2,
        log_every_n_steps=1,
    )


def test_default_temporalwiki_training_plan_runs_all_units_in_order(tmp_path):
    results = run_training(
        build_config(),
        model_factory=build_synthetic_model_and_tokenizer,
        dataset_factory=build_synthetic_dataset,
        checkpoint_dir=str(tmp_path),
    )

    run_root = tmp_path / "integration-training-plan"
    manifest = load_run_manifest(run_root)
    latest_pointer = read_latest_pointer(run_root)

    assert [result["period"] for result in results] == DEFAULT_TEMPORAL_WIKI_PLAN
    assert manifest.training_plan == DEFAULT_TEMPORAL_WIKI_PLAN
    assert latest_pointer.last_period == "nov_dec"
    for unit in DEFAULT_TEMPORAL_WIKI_PLAN:
        assert (run_root / "periods" / unit).is_dir()
