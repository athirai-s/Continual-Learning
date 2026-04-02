from training.train_config import TrainConfig
from training.training_plan import DEFAULT_TEMPORAL_WIKI_PLAN, build_training_plan


def build_config(dataset_name: str) -> TrainConfig:
    return TrainConfig.make_config(
        run_id=f"training-plan-{dataset_name}",
        model_name="synthetic-local-model",
        method="full_ft",
        dataset_name=dataset_name,
        batch_size=1,
        grad_accum_steps=1,
    )


def test_temporal_wiki_uses_declared_runner_owned_period_order():
    plan = build_training_plan(build_config("temporal_wiki"))

    assert plan.dataset_name == "temporal_wiki"
    assert plan.units == DEFAULT_TEMPORAL_WIKI_PLAN


def test_training_plan_override_is_used_verbatim():
    plan = build_training_plan(build_config("temporal_wiki"), ["sep_oct", "oct_nov"])

    assert plan.units == ["sep_oct", "oct_nov"]


def test_non_temporalwiki_datasets_default_to_single_unit_plan():
    assert build_training_plan(build_config("tsqa")).units == ["tsqa"]
    assert build_training_plan(build_config("tgqa")).units == ["tgqa"]
