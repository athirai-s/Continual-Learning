"""metric_helpers.py – Reusable metric computation utilities.

These helpers are method-agnostic and can be imported by evaluation.py,
tests, or any reporting script without pulling in model dependencies.
"""

from __future__ import annotations

import math
from typing import Any


# ---------------------------------------------------------------------------
# String-level metrics
# ---------------------------------------------------------------------------

def token_f1(prediction: str, reference: str) -> float:
    """Unigram token F1 between two whitespace-tokenised strings."""
    pred_tokens = set(prediction.lower().split())
    ref_tokens = set(reference.lower().split())
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = pred_tokens & ref_tokens
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, reference: str) -> float:
    """1.0 if strings match (case-insensitive, stripped), else 0.0."""
    return float(prediction.strip().lower() == reference.strip().lower())


# ---------------------------------------------------------------------------
# Period-level aggregation
# ---------------------------------------------------------------------------

def compute_retention(
    period_results: list[dict[str, Any]],
    metric_key: str,
    current_period_index: int,
) -> float:
    """Average of ``metric_key`` across all *earlier* periods up to but not
    including ``current_period_index``.  Returns NaN if no earlier periods."""
    earlier = period_results[:current_period_index]
    values = [r[metric_key] for r in earlier if metric_key in r and not _is_nan(r[metric_key])]
    return _mean(values)


def compute_forgetting(
    period_results: list[dict[str, Any]],
    metric_key: str,
    current_period_index: int,
) -> float:
    """Score drop between a period's best-ever value and its most recent value.

    Positive forgetting = the model got worse on this period after subsequent
    training.  Negative forgetting = the model improved (forward transfer).
    Returns NaN if there is only one period.
    """
    values = [
        r[metric_key]
        for r in period_results[: current_period_index + 1]
        if metric_key in r and not _is_nan(r[metric_key])
    ]
    if len(values) < 2:
        return float("nan")
    peak = max(values[:-1])
    current = values[-1]
    return peak - current


def compute_average_metric(
    period_results: list[dict[str, Any]],
    metric_key: str,
) -> float:
    """Mean of ``metric_key`` across all periods where it was recorded."""
    values = [r[metric_key] for r in period_results if metric_key in r and not _is_nan(r[metric_key])]
    return _mean(values)


def compute_backward_transfer(
    period_results: list[dict[str, Any]],
    metric_key: str,
) -> float:
    """Average change in earlier-period performance after learning new periods.

    Negative BWT = the model forgot things; positive BWT = new learning helped
    old tasks.  Following the standard CL definition:
        BWT = (1 / T-1) * sum_{i=1}^{T-1} (R_{T,i} - R_{i,i})
    where R_{t,i} is performance on task i evaluated after period t.
    """
    n = len(period_results)
    if n < 2:
        return float("nan")
    diffs = []
    for i in range(n - 1):
        initial = period_results[i].get(metric_key)
        final = period_results[-1].get(metric_key)
        if initial is not None and final is not None and not _is_nan(initial) and not _is_nan(final):
            diffs.append(final - initial)
    return _mean(diffs)


# ---------------------------------------------------------------------------
# CASM-specific helpers
# ---------------------------------------------------------------------------

def compute_version_coverage(
    registry: Any,
    period: str,
) -> dict[str, Any]:
    """How many slots are active, closed, and branched as of ``period``."""
    try:
        slots = registry._slots
    except AttributeError:
        return {}

    active = sum(1 for s in slots if getattr(s, "valid_until", None) is None)
    closed = sum(1 for s in slots if getattr(s, "valid_until", None) is not None)
    branched = sum(1 for s in slots if getattr(s, "parent_slot_id", None) is not None)
    total = len(slots)
    return {
        "casm/slots_total": total,
        "casm/slots_active": active,
        "casm/slots_closed": closed,
        "casm/slots_branched": branched,
    }


def compute_slot_utilisation(registry: Any) -> float:
    """Mean usage count across all slots.  Low values may indicate the router
    is ignoring most slots."""
    try:
        slots = registry._slots
    except AttributeError:
        return float("nan")
    if not slots:
        return float("nan")
    counts = [getattr(s, "usage_count", 0) for s in slots]
    return _mean(counts)


# ---------------------------------------------------------------------------
# SMF-specific helpers
# ---------------------------------------------------------------------------

def compute_smf_active_ratio(model: Any) -> float:
    """Fraction of SMF parameters that are non-zero (sparsity complement)."""
    try:
        from training.smf_model import SMFModelWrapper
    except ImportError:
        return float("nan")

    if not isinstance(model, SMFModelWrapper):
        return float("nan")

    total = 0
    active = 0
    for p in model.smf_parameters():
        total += p.numel()
        active += int((p.abs() > 1e-6).sum().item())
    return active / total if total > 0 else float("nan")


# ---------------------------------------------------------------------------
# Private utilities
# ---------------------------------------------------------------------------

def _mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return sum(values) / len(values)


def _is_nan(x: Any) -> bool:
    try:
        return math.isnan(x)
    except (TypeError, ValueError):
        return False
