import torch

from training.synthetic_backend import load_synthetic_model
from training.train_config import TrainConfig
from training.train_runner import (
    build_synthetic_dataset,
    build_synthetic_model_and_tokenizer,
    run_training,
)


def build_config(run_id: str) -> TrainConfig:
    return TrainConfig.make_config(
        run_id=run_id,
        model_name="synthetic-local-model",
        method="full_ft",
        dataset_name="temporal_wiki",
        batch_size=1,
        grad_accum_steps=1,
        max_passages_per_period=2,
        log_every_n_steps=1,
        seed=23,
    )


def test_same_seed_produces_identical_synthetic_final_weights(tmp_path):
    first_results = run_training(
        build_config("seed-run-a"),
        model_factory=build_synthetic_model_and_tokenizer,
        dataset_factory=build_synthetic_dataset,
        checkpoint_dir=str(tmp_path / "first"),
    )
    second_results = run_training(
        build_config("seed-run-b"),
        model_factory=build_synthetic_model_and_tokenizer,
        dataset_factory=build_synthetic_dataset,
        checkpoint_dir=str(tmp_path / "second"),
    )

    first_model = load_synthetic_model(first_results[0]["checkpoint_path"])
    second_model = load_synthetic_model(second_results[0]["checkpoint_path"])

    for name, tensor in first_model.state_dict().items():
        assert torch.equal(tensor, second_model.state_dict()[name]), name
