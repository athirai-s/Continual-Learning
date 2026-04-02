from training.evaluation_runner import determine_eval_splits
from training.train_config import TrainConfig


def build_config(dataset_name: str) -> TrainConfig:
    return TrainConfig.make_config(
        run_id=f"eval-splits-{dataset_name}",
        model_name="synthetic-local-model",
        method="full_ft",
        dataset_name=dataset_name,
        batch_size=1,
        grad_accum_steps=1,
    )


def test_temporal_wiki_evaluates_changed_and_unchanged_splits():
    assert determine_eval_splits(build_config("temporal_wiki")) == ["changed", "unchanged"]


def test_tsqa_and_tgqa_evaluate_val_split():
    assert determine_eval_splits(build_config("tsqa")) == ["val"]
    assert determine_eval_splits(build_config("tgqa")) == ["val"]
