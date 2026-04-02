import hashlib
import json
import os
import random
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from casf_dataset_api import MemoryRegistry, TemporalWikiDataset, TGQADataset, TSQADataset

from artifacts.checkpoint_manifest import CheckpointManifestError
from artifacts.checkpointing import RunRootLock, prepare_run_root, resolve_checkpoint_path
from artifacts.run_artifacts import (
    collect_reproducibility_metadata,
    ensure_run_layout,
    write_run_manifest,
)
from .evaluation_runner import run_period_evaluation
from .metrics_logger import MetricsLogger
from .synthetic_backend import (
    SyntheticTemporalDataset,
    SyntheticTokenizer,
    build_synthetic_model,
    load_synthetic_model,
)
from .train_config import TrainConfig
from .training_plan import build_training_plan
from .trainer import CASFTrainer, ResumeState

DatasetFactory = Callable[[str, TrainConfig], Any]
ModelFactory = Callable[[TrainConfig], tuple[Any, Any]]
ResumeModelFactory = Callable[[TrainConfig, str], tuple[Any, Any]]


def apply_global_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def prepare_dataset(dataset: Any, cfg: TrainConfig, unit: str) -> str:
    if cfg.dataset_name == "temporal_wiki":
        dataset.load("changed")
        dataset.load("unchanged")
        return unit

    dataset.load("train")
    return unit


def build_resume_compatibility(cfg: TrainConfig) -> dict[str, Any]:
    return {
        "dataset_name": cfg.dataset_name,
        "precision": cfg.precision,
        "learning_rate": cfg.learning_rate,
        "batch_size": cfg.batch_size,
        "grad_accum_steps": cfg.grad_accum_steps,
        "epochs_per_period": cfg.epochs_per_period,
        "grad_clip": cfg.grad_clip,
        "warmup_steps": cfg.warmup_steps,
        "min_passage_length": cfg.min_passage_length,
        "max_passages_per_period": cfg.max_passages_per_period,
        "log_every_n_steps": cfg.log_every_n_steps,
        "checkpoint_every_n_optimizer_steps": cfg.checkpoint_every_n_optimizer_steps,
    }


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _stable_json_hash(obj: Any) -> str:
    return _sha256_bytes(json.dumps(obj, sort_keys=True).encode("utf-8"))


def build_dataset_identity(dataset: Any, cfg: TrainConfig, unit: str) -> dict[str, Any]:
    if isinstance(dataset, SyntheticTemporalDataset):
        return {
            "kind": "synthetic",
            "unit": unit,
            "snapshot_id": dataset.snapshot_id,
            "content_sha256": _stable_json_hash(
                {
                    "train": dataset.get_train_passages(),
                    "changed": [probe.prompt for probe in dataset.get_probes("changed")],
                    "unchanged": [probe.prompt for probe in dataset.get_probes("unchanged")],
                }
            ),
        }
    if isinstance(dataset, TemporalWikiDataset):
        from casf_dataset_api.download_dataset_scripts.data.temporal_wiki import (
            DIFFSETS_ZIP,
            PROBES_ZIP,
        )

        return {
            "kind": "temporal_wiki",
            "unit": unit,
            "probes_zip_sha256": _sha256_file(PROBES_ZIP),
            "diffsets_zip_sha256": _sha256_file(DIFFSETS_ZIP),
        }
    if isinstance(dataset, TSQADataset):
        split_name = "validation" if dataset._loaded_split == "val" else dataset._loaded_split
        split_ds = dataset._ds[split_name]
        fingerprint = getattr(split_ds, "_fingerprint", None)
        return {
            "kind": "tsqa",
            "unit": unit,
            "split": split_name,
            "config": dataset.source_filter,
            "fingerprint": fingerprint,
            "content_sha256": None if fingerprint else _stable_json_hash(list(split_ds)),
        }
    if isinstance(dataset, TGQADataset):
        split_ds = dataset._ds[dataset._loaded_split]
        fingerprint = getattr(split_ds, "_fingerprint", None)
        return {
            "kind": "tgqa",
            "unit": unit,
            "split": dataset._loaded_split,
            "config": dataset.config,
            "fingerprint": fingerprint,
            "content_sha256": None if fingerprint else _stable_json_hash(list(split_ds)),
        }
    raise ValueError(f"No dataset identity adapter for {type(dataset).__name__}")


def validate_resume_inputs(
    trainer: CASFTrainer,
    cfg: TrainConfig,
    training_plan: list[str],
    dataset_identity: dict[str, Any],
) -> None:
    manifest = trainer._checkpoint_manifest
    if manifest is None:
        return

    if manifest.model_name != cfg.model_name:
        raise CheckpointManifestError(
            f"Resume model_name mismatch: checkpoint={manifest.model_name!r}, current={cfg.model_name!r}"
        )
    if training_plan[: len(manifest.training_plan)] != manifest.training_plan:
        raise CheckpointManifestError(
            f"Resume training_plan mismatch: checkpoint={manifest.training_plan!r}, current={training_plan!r}"
        )
    current_compat = build_resume_compatibility(cfg)
    if manifest.resume_compatibility != current_compat:
        raise CheckpointManifestError("Resume compatibility settings do not match the checkpoint")
    if manifest.dataset_identity != dataset_identity:
        raise CheckpointManifestError("Dataset identity does not match the checkpoint")


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
    with RunRootLock(run_root):
        apply_global_seed(cfg.seed)
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
        training_plan = build_training_plan(cfg, training_units)
        units = training_plan.units
        if resume_state is not None and resume_state.current_unit not in units:
            raise ValueError(f"Resume unit {resume_state.current_unit!r} is not in the training plan")
        if resume_state is not None and not resume_state.metadata_only:
            resume_dataset = dataset_factory(resume_state.current_unit, cfg)
            resume_period = prepare_dataset(resume_dataset, cfg, resume_state.current_unit)
            resume_dataset_identity = build_dataset_identity(resume_dataset, cfg, resume_period)
            validate_resume_inputs(trainer, cfg, units, resume_dataset_identity)

        ensure_run_layout(run_root, units)
        write_run_manifest(
            run_root,
            cfg,
            units,
            reproducibility=collect_reproducibility_metadata(cfg, units),
        )

        cfg_path = os.path.join(cfg.checkpoint_dir, f"{cfg.run_id}_config.json")
        cfg.save_json(cfg_path)
        print(f"Saved config to {cfg_path}\n")
        metrics_logger = MetricsLogger(run_root)

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
            dataset_identity = build_dataset_identity(dataset, cfg, period_name)
            if resume_state is not None and unit == resume_state.current_unit and not resume_state.metadata_only:
                validate_resume_inputs(trainer, cfg, units, dataset_identity)
            checkpoint_paths: list[str] = []

            manifest_metadata = {
                "model_name": cfg.model_name,
                "training_plan": units,
                "resume_compatibility": build_resume_compatibility(cfg),
                "dataset_identity": dataset_identity,
            }

            def checkpoint_hook(period: str, optimizer_step: int) -> None:
                checkpoint_start = time.perf_counter()
                checkpoint_path = trainer.checkpoint(
                    str(period),
                    run_root,
                    manifest_metadata=manifest_metadata,
                    lock_run_root=False,
                )
                checkpoint_time_sec = time.perf_counter() - checkpoint_start
                checkpoint_paths.append(checkpoint_path)
                metrics_logger.emit(
                    "checkpoint",
                    unit=str(period),
                    optimizer_step=optimizer_step,
                    checkpoint_path=str(Path(checkpoint_path).relative_to(run_root)),
                    checkpoint_time_sec=checkpoint_time_sec,
                )
                print(
                    f"Checkpoint saved at optimizer_step={optimizer_step}: {checkpoint_path}"
                )

            def event_hook(event: dict[str, Any]) -> None:
                event_payload = dict(event)
                event_type = event_payload.pop("event_type")
                metrics_logger.emit(event_type, **event_payload)

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
                event_hook=event_hook,
                resume_state=active_resume_state,
            )
            checkpoint_start = time.perf_counter()
            checkpoint_path = trainer.checkpoint(
                str(period_name),
                run_root,
                manifest_metadata=manifest_metadata,
                lock_run_root=False,
            )
            checkpoint_time_sec = time.perf_counter() - checkpoint_start
            metrics_logger.emit(
                "checkpoint",
                unit=str(period_name),
                optimizer_step=result["optimizer_steps_total"],
                checkpoint_path=str(Path(checkpoint_path).relative_to(run_root)),
                checkpoint_time_sec=checkpoint_time_sec,
            )

            result["checkpoint_path"] = checkpoint_path
            result["checkpoint_paths"] = checkpoint_paths + [checkpoint_path]
            if cfg.eval_after_each_period:
                eval_dataset = dataset_factory(unit, cfg)
                result["evaluation"] = run_period_evaluation(
                    model=trainer.model,
                    tokenizer=trainer.tokenizer,
                    dataset=eval_dataset,
                    cfg=cfg,
                    unit=period_name,
                    run_root=run_root,
                )
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
