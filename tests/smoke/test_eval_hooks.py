import json

from train_config import TrainConfig
from train_runner import build_synthetic_dataset, build_synthetic_model_and_tokenizer, run_training


def build_config() -> TrainConfig:
    return TrainConfig.make_config(
        run_id="synthetic-eval-smoke",
        model_name="synthetic-local-model",
        method="full_ft",
        dataset_name="temporal_wiki",
        batch_size=1,
        grad_accum_steps=1,
        max_passages_per_period=2,
        log_every_n_steps=1,
    )


def test_synthetic_training_writes_eval_summary_and_eval_metrics(tmp_path):
    results = run_training(
        build_config(),
        model_factory=build_synthetic_model_and_tokenizer,
        dataset_factory=build_synthetic_dataset,
        checkpoint_dir=str(tmp_path),
    )

    eval_summary_path = tmp_path / "synthetic-eval-smoke" / "periods" / "aug_sep" / "eval_summary.json"
    eval_metrics_path = tmp_path / "synthetic-eval-smoke" / "metrics" / "eval.jsonl"

    summary = json.loads(eval_summary_path.read_text())
    eval_events = [json.loads(line) for line in eval_metrics_path.read_text().splitlines()]

    assert results[0]["evaluation"].keys() == {"changed", "unchanged"}
    assert summary["schema_version"] == 1
    assert summary["unit"] == "aug_sep"
    assert set(summary["results"]) == {"changed", "unchanged"}
    assert [event["split"] for event in eval_events] == ["changed", "unchanged"]
    assert all(event["event_type"] == "eval" for event in eval_events)
