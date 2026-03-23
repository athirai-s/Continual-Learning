import json

from train_config import TrainConfig
from train_runner import build_synthetic_dataset, build_synthetic_model_and_tokenizer, run_training


def build_config() -> TrainConfig:
    return TrainConfig.make_config(
        run_id="integration-metrics-schema",
        model_name="synthetic-local-model",
        method="full_ft",
        dataset_name="temporal_wiki",
        batch_size=1,
        grad_accum_steps=1,
        max_passages_per_period=2,
        log_every_n_steps=1,
    )


def test_metrics_events_follow_stable_schema(tmp_path):
    run_training(
        build_config(),
        model_factory=build_synthetic_model_and_tokenizer,
        dataset_factory=build_synthetic_dataset,
        checkpoint_dir=str(tmp_path),
    )

    events_path = tmp_path / "integration-metrics-schema" / "metrics" / "events.jsonl"
    events = [json.loads(line) for line in events_path.read_text().splitlines()]

    assert events
    for event in events:
        assert event["schema_version"] == 1
        assert isinstance(event["timestamp_unix"], float)
        assert event["event_type"] in {"train_step", "period_end", "checkpoint"}

    train_step_event = next(event for event in events if event["event_type"] == "train_step")
    assert {
        "unit",
        "optimizer_step",
        "total_optimizer_steps",
        "micro_step",
        "loss",
    } <= train_step_event.keys()

    period_end_event = next(event for event in events if event["event_type"] == "period_end")
    assert {
        "unit",
        "train_loss_final",
        "n_passages_trained",
        "n_contradiction_passages",
        "train_duration_sec",
        "micro_steps_total",
        "optimizer_steps_total",
    } <= period_end_event.keys()

    checkpoint_event = next(event for event in events if event["event_type"] == "checkpoint")
    assert {"unit", "optimizer_step", "checkpoint_path"} <= checkpoint_event.keys()
