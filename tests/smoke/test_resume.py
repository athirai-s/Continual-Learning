import pytest
import torch

from casf_dataset_api import MemoryRegistry
from synthetic_backend import (
    SyntheticTokenizer,
    build_synthetic_model,
    load_synthetic_model,
)
from train_config import TrainConfig
from train_runner import (
    build_synthetic_dataset,
    build_synthetic_model_and_tokenizer,
    run_training,
)
from trainer import CASFTrainer


def build_config(run_id: str, *, checkpoint_every_n_optimizer_steps: int | None = None) -> TrainConfig:
    return TrainConfig.make_config(
        run_id=run_id,
        model_name="synthetic-local-model",
        method="full_ft",
        dataset_name="temporal_wiki",
        batch_size=1,
        grad_accum_steps=1,
        max_passages_per_period=3,
        log_every_n_steps=1,
        checkpoint_every_n_optimizer_steps=checkpoint_every_n_optimizer_steps,
    )


def build_trainer(run_id: str = "resume-smoke") -> CASFTrainer:
    return CASFTrainer(
        build_synthetic_model(),
        SyntheticTokenizer(),
        build_config(run_id),
        MemoryRegistry(),
    )


def test_resume_restores_registry_and_returns_resume_state(tmp_path):
    cfg = build_config("resume-state")
    results = run_training(
        cfg,
        model_factory=build_synthetic_model_and_tokenizer,
        resume_model_factory=lambda cfg, path: (
            load_synthetic_model(path),
            SyntheticTokenizer.from_pretrained(path),
        ),
        dataset_factory=build_synthetic_dataset,
        checkpoint_dir=str(tmp_path),
        training_units=["aug_sep"],
    )

    trainer = build_trainer("resume-state")
    state = trainer.resume(results[0]["checkpoint_path"])
    alpha_slot = trainer.registry.get_active("Alpha", "relation")

    assert state.current_unit == "aug_sep"
    assert state.unit_completed is True
    assert state.completed_units == ["aug_sep"]
    assert alpha_slot is not None
    assert alpha_slot.value == "new value"


def test_runner_resume_from_completed_unit_skips_to_next_unit(tmp_path):
    initial_cfg = build_config("resume-first-unit")
    initial_results = run_training(
        initial_cfg,
        model_factory=build_synthetic_model_and_tokenizer,
        resume_model_factory=lambda cfg, path: (
            load_synthetic_model(path),
            SyntheticTokenizer.from_pretrained(path),
        ),
        dataset_factory=build_synthetic_dataset,
        checkpoint_dir=str(tmp_path / "initial"),
        training_units=["aug_sep"],
    )

    resumed_cfg = build_config("resume-next-unit")
    resumed_results = run_training(
        resumed_cfg,
        model_factory=build_synthetic_model_and_tokenizer,
        resume_model_factory=lambda cfg, path: (
            load_synthetic_model(path),
            SyntheticTokenizer.from_pretrained(path),
        ),
        dataset_factory=build_synthetic_dataset,
        checkpoint_dir=str(tmp_path / "resumed"),
        training_units=["aug_sep", "sep_oct"],
        resume_from=initial_results[0]["checkpoint_path"],
    )

    assert len(resumed_results) == 1
    assert resumed_results[0]["period"] == "sep_oct"


def test_runner_resume_from_intermediate_checkpoint_matches_uninterrupted_final_model(tmp_path):
    baseline_cfg = build_config(
        "baseline-split",
        checkpoint_every_n_optimizer_steps=1,
    )
    baseline_results = run_training(
        baseline_cfg,
        model_factory=build_synthetic_model_and_tokenizer,
        resume_model_factory=lambda cfg, path: (
            load_synthetic_model(path),
            SyntheticTokenizer.from_pretrained(path),
        ),
        dataset_factory=build_synthetic_dataset,
        checkpoint_dir=str(tmp_path / "baseline"),
        training_units=["aug_sep"],
    )

    midpoint_checkpoint = baseline_results[0]["checkpoint_paths"][0]
    resumed_cfg = build_config(
        "resumed-split",
        checkpoint_every_n_optimizer_steps=1,
    )
    resumed_results = run_training(
        resumed_cfg,
        model_factory=build_synthetic_model_and_tokenizer,
        resume_model_factory=lambda cfg, path: (
            load_synthetic_model(path),
            SyntheticTokenizer.from_pretrained(path),
        ),
        dataset_factory=build_synthetic_dataset,
        checkpoint_dir=str(tmp_path / "resumed"),
        training_units=["aug_sep"],
        resume_from=midpoint_checkpoint,
    )

    baseline_model = load_synthetic_model(baseline_results[0]["checkpoint_path"])
    resumed_model = load_synthetic_model(resumed_results[0]["checkpoint_path"])

    for name, tensor in baseline_model.state_dict().items():
        assert torch.allclose(
            tensor,
            resumed_model.state_dict()[name],
            atol=1e-6,
            rtol=1e-6,
        ), name


def test_resume_fails_when_last_period_metadata_is_missing(tmp_path):
    checkpoint_path = tmp_path / "missing-metadata"
    checkpoint_path.mkdir(parents=True)
    MemoryRegistry().save(checkpoint_path / "memory_registry.json")

    trainer = build_trainer()

    with pytest.raises(FileNotFoundError, match="Missing checkpoint metadata"):
        trainer.resume(checkpoint_path)
