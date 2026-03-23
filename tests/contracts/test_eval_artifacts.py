import json

from train_config import TrainConfig
from train_runner import build_synthetic_dataset, build_synthetic_model_and_tokenizer, run_training


def build_config() -> TrainConfig:
    return TrainConfig.make_config(
        run_id="synthetic-eval-contract",
        model_name="synthetic-local-model",
        method="full_ft",
        dataset_name="temporal_wiki",
        batch_size=1,
        grad_accum_steps=1,
        max_passages_per_period=2,
        log_every_n_steps=1,
    )


def test_eval_outputs_follow_stable_artifact_contract(tmp_path):
    run_training(
        build_config(),
        model_factory=build_synthetic_model_and_tokenizer,
        dataset_factory=build_synthetic_dataset,
        checkpoint_dir=str(tmp_path),
        training_units=["aug_sep"],
    )

    eval_summary_path = tmp_path / "synthetic-eval-contract" / "periods" / "aug_sep" / "eval_summary.json"
    eval_metrics_path = tmp_path / "synthetic-eval-contract" / "metrics" / "eval.jsonl"

    summary = json.loads(eval_summary_path.read_text())
    events = [json.loads(line) for line in eval_metrics_path.read_text().splitlines()]

    assert summary["schema_version"] == 1
    assert summary["dataset_name"] == "temporal_wiki"
    assert summary["unit"] == "aug_sep"
    assert set(summary["results"]) == {"changed", "unchanged"}
    for split, result in summary["results"].items():
        assert {"plasticity", "stability", "token_f1", "n_correct", "n_total", "per_relation", "routing_acc"} <= result.keys()
        assert split in {"changed", "unchanged"}

    assert len(events) == 2
    for event in events:
        assert event["schema_version"] == 1
        assert event["event_type"] == "eval"
        assert event["unit"] == "aug_sep"
        assert event["split"] in {"changed", "unchanged"}
