"""evaluation_runner.py – Thin adapter called by train_runner.py.

train_runner.py already imports ``run_period_evaluation`` from
``training.evaluation_runner``.  This module lives in eval_and_metrics/
but is re-exported from training/ so the existing import path works.

Alternatively you can drop this file in training/ directly — the logic
is identical.
"""

from __future__ import annotations

from typing import Any

from casf_dataset_api import TemporalDataset, MemoryRegistry

from .evaluation import evaluate_period


def run_period_evaluation(
    *,
    model: Any,
    tokenizer: Any,
    dataset: TemporalDataset,
    cfg: Any,
    unit: str,
    run_root: str,
    registry: MemoryRegistry | None = None,
    eval_batch_size: int = 4,
    max_new_tokens: int = 32,
) -> dict[str, Any]:
    """Called by ``train_runner.run_training()`` after each period when
    ``cfg.eval_after_each_period`` is True.

    Returns the evaluation result dict which is stored under
    ``result["evaluation"]`` in the period result.
    """
    if registry is None:
        # Fallback: create an empty registry so evaluate_period doesn't crash.
        registry = MemoryRegistry()

    return evaluate_period(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        cfg=cfg,
        unit=unit,
        registry=registry,
        eval_batch_size=eval_batch_size,
        max_new_tokens=max_new_tokens,
    )
