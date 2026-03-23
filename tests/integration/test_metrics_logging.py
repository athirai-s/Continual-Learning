import json

from train_config import TrainConfig
from train_runner import build_synthetic_dataset, build_synthetic_model_and_tokenizer, run_training


def build_config() -> TrainConfig:
    return TrainConfig.make_config(
        run_id="integration-metrics-events",
        model_name="synthetic-local-model",
        method="full_ft",
        dataset_name="temporal_wiki",
        batch_size=1,
        grad_accum_steps=1,
        max_passages_per_period=2,
        log_every_n_steps=1,
    )


def test_synthetic_training_emits_train_step_period_end_and_checkpoint_events(tmp_path):
    results = run_training(
        build_config(),
        model_factory=build_synthetic_model_and_tokenizer,
        dataset_factory=build_synthetic_dataset,
        checkpoint_dir=str(tmp_path),
    )

    events_path = tmp_path / "integration-metrics-events" / "metrics" / "events.jsonl"
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    event_types = [event["event_type"] for event in events]

    assert event_types == ["train_step", "train_step", "period_end", "checkpoint"]
    assert events[-1]["checkpoint_path"] == "checkpoints/ckpt-000001"
    assert events[-1]["optimizer_step"] == results[0]["optimizer_steps_total"]
    assert events[-2]["unit"] == "aug_sep"
