import os
from dataclasses import replace
from typing import Any, Callable

from transformers import AutoModelForCausalLM, AutoTokenizer

from casf_dataset_api import MemoryRegistry, TemporalWikiDataset, TGQADataset, TSQADataset
from synthetic_backend import SyntheticTemporalDataset, SyntheticTokenizer, build_synthetic_model
from train_config import TrainConfig
from trainer import CASFTrainer
from checkpointing import prepare_run_root


PERIODS = ["aug_sep"]

DatasetFactory = Callable[[str, TrainConfig], Any]
ModelFactory = Callable[[TrainConfig], tuple[Any, Any]]


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


def build_real_dataset(unit: str, cfg: TrainConfig):
    if cfg.dataset_name == "temporal_wiki":
        return build_dataset(cfg.dataset_name, period=unit)
    return build_dataset(cfg.dataset_name)


def build_synthetic_model_and_tokenizer(cfg: TrainConfig) -> tuple[Any, Any]:
    tokenizer = SyntheticTokenizer()
    model = build_synthetic_model(vocab_size=tokenizer.vocab_size)
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
    dataset_factory: DatasetFactory,
    checkpoint_dir: str | None = None,
    training_units: list[str] | None = None,
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

    model, tokenizer = model_factory(cfg)

    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    registry = MemoryRegistry()
    trainer = CASFTrainer(model, tokenizer, cfg, registry)

    results: list[dict[str, Any]] = []
    for unit in training_units or get_training_units(cfg):
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

        result = trainer.train_period(dataset, period_name, checkpoint_hook=checkpoint_hook)
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

    print("\nDone.")
    return results


def run_mode(mode: str, cfg: TrainConfig, checkpoint_dir: str | None = None) -> list[dict[str, Any]]:
    if mode == "real":
        return run_training(
            cfg,
            model_factory=build_real_model_and_tokenizer,
            dataset_factory=build_real_dataset,
            checkpoint_dir=checkpoint_dir,
        )
    if mode == "synthetic":
        return run_training(
            cfg,
            model_factory=build_synthetic_model_and_tokenizer,
            dataset_factory=build_synthetic_dataset,
            checkpoint_dir=checkpoint_dir,
        )
    raise ValueError(f"Unsupported mode: {mode}")
