import os
from dataclasses import replace
from typing import Any, Callable

from transformers import AutoModelForCausalLM, AutoTokenizer

from casf_dataset_api import MemoryRegistry, TemporalWikiDataset, TGQADataset, TSQADataset
from checkpointing import prepare_run_root, resolve_checkpoint_path
from synthetic_backend import (
    SyntheticTemporalDataset,
    SyntheticTokenizer,
    build_synthetic_model,
    load_synthetic_model,
)
from train_config import TrainConfig
from trainer import CASFTrainer, ResumeState


PERIODS = ["aug_sep"]

DatasetFactory = Callable[[str, TrainConfig], Any]
ModelFactory = Callable[[TrainConfig], tuple[Any, Any]]
ResumeModelFactory = Callable[[TrainConfig, str], tuple[Any, Any]]


def build_dataset(dataset_name: str, period: str | None = None):
    if dataset_name == "temporal_wiki":
        return TemporalWikiDataset(period=period)
    if dataset_name == "tsqa":
        return TSQADataset()
    if dataset_name == "tgqa":
        return TGQADataset()
    raise ValueError(f"Unsupported dataset_name: {dataset_name}")


def build_real_model_and_tokenizer(cfg: TrainConfig) -> tuple[Any, Any]:
    print(f"Loading model: {cfg.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name)
    return model, tokenizer


def load_real_model_and_tokenizer(cfg: TrainConfig, checkpoint_path: str) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
    return model, tokenizer


def build_real_dataset(unit: str, cfg: TrainConfig):
    if cfg.dataset_name == "temporal_wiki":
        return build_dataset(cfg.dataset_name, period=unit)
    return build_dataset(cfg.dataset_name)


def build_synthetic_model_and_tokenizer(cfg: TrainConfig) -> tuple[Any, Any]:
    tokenizer = SyntheticTokenizer()
    model = build_synthetic_model(vocab_size=tokenizer.vocab_size)
    return model, tokenizer


def load_synthetic_model_and_tokenizer(cfg: TrainConfig, checkpoint_path: str) -> tuple[Any, Any]:
    tokenizer = SyntheticTokenizer.from_pretrained(checkpoint_path)
    model = load_synthetic_model(checkpoint_path)
    return model, tokenizer


def build_synthetic_dataset(unit: str, cfg: TrainConfig):
    return SyntheticTemporalDataset()


def get_training_units(cfg: TrainConfig) -> list[str]:
    if cfg.dataset_name == "temporal_wiki":
        return PERIODS
    return [cfg.dataset_name]


def prepare_dataset(dataset: Any, cfg: TrainConfig, unit: str) -> str:
    if cfg.dataset_name == "temporal_wiki":
        dataset.load("changed")
        dataset.load("unchanged")
        return unit

    dataset.load("train")
    return unit


def run_training(
    cfg: TrainConfig,
    *,
    model_factory: ModelFactory,
    resume_model_factory: ResumeModelFactory | None = None,
    dataset_factory: DatasetFactory,
    checkpoint_dir: str | None = None,
    training_units: list[str] | None = None,
    resume_from: str | None = None,
) -> list[dict[str, Any]]:
    if checkpoint_dir is not None:
        cfg = replace(cfg, checkpoint_dir=checkpoint_dir)

    print("Training config:")
    print(cfg)

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    run_root = os.path.join(cfg.checkpoint_dir, cfg.run_id)
    prepare_run_root(run_root)
    cfg_path = os.path.join(cfg.checkpoint_dir, f"{cfg.run_id}_config.json")
    cfg.save_json(cfg_path)
    print(f"Saved config to {cfg_path}\n")

    resume_path = resolve_checkpoint_path(resume_from) if resume_from is not None else None
    if resume_path is None:
        model, tokenizer = model_factory(cfg)
    else:
        if resume_model_factory is None:
            raise ValueError("resume_model_factory is required when resume_from is set")
        model, tokenizer = resume_model_factory(cfg, str(resume_path))

    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    registry = MemoryRegistry()
    trainer = CASFTrainer(model, tokenizer, cfg, registry)
    resume_state: ResumeState | None = None
    if resume_from is not None:
        resume_state = trainer.resume(resume_from)

    results: list[dict[str, Any]] = []
    units = training_units or get_training_units(cfg)
    if resume_state is not None and resume_state.current_unit not in units:
        raise ValueError(f"Resume unit {resume_state.current_unit!r} is not in the training plan")
    if resume_state is None:
        pending_units = units
    elif resume_state.unit_completed:
        current_index = units.index(resume_state.current_unit)
        pending_units = units[current_index + 1 :]
    else:
        current_index = units.index(resume_state.current_unit)
        pending_units = units[current_index:]

    for unit in pending_units:
        print(f"\n=== Training unit: {unit} ===")
        dataset = dataset_factory(unit, cfg)
        period_name = prepare_dataset(dataset, cfg, unit)
        checkpoint_paths: list[str] = []

        def checkpoint_hook(period: str, optimizer_step: int) -> None:
            checkpoint_path = trainer.checkpoint(str(period), run_root)
            checkpoint_paths.append(checkpoint_path)
            print(
                f"Checkpoint saved at optimizer_step={optimizer_step}: {checkpoint_path}"
            )

        active_resume_state = (
            resume_state
            if resume_state is not None
            and not resume_state.unit_completed
            and unit == resume_state.current_unit
            else None
        )
        result = trainer.train_period(
            dataset,
            period_name,
            checkpoint_hook=checkpoint_hook,
            resume_state=active_resume_state,
        )
        checkpoint_path = trainer.checkpoint(str(period_name), run_root)

        result["checkpoint_path"] = checkpoint_path
        result["checkpoint_paths"] = checkpoint_paths + [checkpoint_path]
        results.append(result)

        print("Training result:")
        print(f"  Final loss: {result['train_loss_final']}")
        print(f"  Passages trained: {result['n_passages_trained']}")
        print(f"  Contradiction passages: {result['n_contradiction_passages']}")
        print(f"  Train duration (sec): {result['train_duration_sec']:.2f}")
        print(f"Checkpoint saved to: {checkpoint_path}")
        resume_state = None

    print("\nDone.")
    return results


def run_mode(
    mode: str,
    cfg: TrainConfig,
    checkpoint_dir: str | None = None,
    resume_from: str | None = None,
) -> list[dict[str, Any]]:
    if mode == "real":
        return run_training(
            cfg,
            model_factory=build_real_model_and_tokenizer,
            resume_model_factory=load_real_model_and_tokenizer,
            dataset_factory=build_real_dataset,
            checkpoint_dir=checkpoint_dir,
            resume_from=resume_from,
        )
    if mode == "synthetic":
        return run_training(
            cfg,
            model_factory=build_synthetic_model_and_tokenizer,
            resume_model_factory=load_synthetic_model_and_tokenizer,
            dataset_factory=build_synthetic_dataset,
            checkpoint_dir=checkpoint_dir,
            resume_from=resume_from,
        )
    raise ValueError(f"Unsupported mode: {mode}")
