import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from casf_dataset_api import EvalResult, TemporalEvaluator
from run_artifacts import metrics_root, period_root
from train_config import TrainConfig


EVAL_SUMMARY_FILENAME = "eval_summary.json"
EVAL_METRICS_FILENAME = "eval.jsonl"
EVAL_SCHEMA_VERSION = 1


def determine_eval_splits(cfg: TrainConfig) -> list[str]:
    if cfg.dataset_name == "temporal_wiki":
        return ["changed", "unchanged"]
    if cfg.dataset_name in {"tsqa", "tgqa"}:
        return ["val"]
    raise ValueError(f"Unsupported dataset_name for evaluation: {cfg.dataset_name}")


def eval_metrics_path(run_root: str | Path) -> Path:
    return metrics_root(run_root) / EVAL_METRICS_FILENAME


def eval_summary_path(run_root: str | Path, unit: str) -> Path:
    return period_root(run_root, unit) / EVAL_SUMMARY_FILENAME


class GenerationAdapter:
    def __init__(self, model: Any, tokenizer: Any):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device

    def _resolve_max_length(self) -> int:
        candidates: list[int] = []
        tokenizer_max_length = getattr(self.tokenizer, "model_max_length", None)
        if isinstance(tokenizer_max_length, int) and 0 < tokenizer_max_length < 100_000:
            candidates.append(tokenizer_max_length)

        model_config = getattr(self.model, "config", None)
        for attr in ("n_positions", "max_position_embeddings", "n_ctx"):
            value = getattr(model_config, attr, None)
            if isinstance(value, int) and value > 0:
                candidates.append(value)

        return min(candidates) if candidates else 512

    def generate(self, prompt: str) -> str:
        encoded = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self._resolve_max_length(),
            padding=None,
            return_tensors="pt",
        )
        batch = {key: value.to(self.device) for key, value in encoded.items()}
        prompt_length = batch["input_ids"].shape[1]
        with torch.no_grad():
            output = self.model.generate(
                **batch,
                max_new_tokens=8,
                pad_token_id=getattr(self.tokenizer, "pad_token_id", None),
                eos_token_id=getattr(self.tokenizer, "eos_token_id", None),
            )
        generated_tokens = output[0][prompt_length:]
        if hasattr(self.tokenizer, "decode"):
            return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return " ".join(str(token_id) for token_id in generated_tokens.tolist())


def _serialize_eval_result(result: EvalResult) -> dict[str, Any]:
    return asdict(result)


def run_period_evaluation(
    *,
    model: Any,
    tokenizer: Any,
    dataset: Any,
    cfg: TrainConfig,
    unit: str,
    run_root: str | Path,
) -> dict[str, dict[str, Any]]:
    evaluator = TemporalEvaluator()
    generation_model = GenerationAdapter(model, tokenizer)

    split_results: dict[str, dict[str, Any]] = {}
    for split in determine_eval_splits(cfg):
        dataset.load(split)
        result = evaluator.evaluate(generation_model, dataset, split=split)
        split_results[split] = _serialize_eval_result(result)

    summary = {
        "schema_version": EVAL_SCHEMA_VERSION,
        "unit": unit,
        "dataset_name": cfg.dataset_name,
        "results": split_results,
    }
    summary_path = eval_summary_path(run_root, unit)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    metrics_path = eval_metrics_path(run_root)
    with open(metrics_path, "a") as f:
        for split, result in split_results.items():
            payload = {
                "schema_version": EVAL_SCHEMA_VERSION,
                "event_type": "eval",
                "unit": unit,
                "split": split,
                **result,
            }
            f.write(json.dumps(payload, sort_keys=True) + "\n")

    return split_results
