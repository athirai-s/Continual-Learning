"""report.py – Generate a per-period evaluation report from a completed run.

Usage:
    python eval_and_metrics/report.py --run-root checkpoints/my_run_id
    python eval_and_metrics/report.py --run-root checkpoints/my_run_id --format json
    python eval_and_metrics/report.py --run-root checkpoints/my_run_id --format csv

The script reads the metrics.jsonl file written by MetricsLogger and
prints a structured per-period summary table, including forgetting and
retention calculations across the full temporal training plan.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import sys
from pathlib import Path
from typing import Any

from eval_and_metrics.metric_helpers import (
    compute_average_metric,
    compute_backward_transfer,
    compute_forgetting,
    compute_retention,
    _is_nan,
)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_metrics_jsonl(run_root: Path) -> list[dict[str, Any]]:
    """Load all event dicts from the run's metrics.jsonl file."""
    candidates = (
        list(run_root.glob("metrics*.jsonl"))
        + list(run_root.glob("logs/metrics*.jsonl"))
        + list(run_root.glob("metrics/*.jsonl"))
    )
    if not candidates:
        # flat layout used by MetricsLogger
        candidate = run_root / "metrics.jsonl"
        if candidate.exists():
            candidates = [candidate]
    if not candidates:
        raise FileNotFoundError(
            f"No metrics.jsonl found under {run_root}.  "
            "Make sure the run has completed at least one period."
        )
    events: list[dict[str, Any]] = []
    for path in candidates:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    return events


def extract_period_end_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [e for e in events if e.get("event_type") == "period_end"]


def extract_eval_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [e for e in events if e.get("event_type") in ("evaluation", "eval")]


def merge_period_results(
    period_ends: list[dict[str, Any]],
    eval_events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Join training-end and evaluation events by period unit name.

    Multiple eval events for the same unit (e.g. old per-split "eval" events)
    are merged in order so later keys win — but plasticity/stability from
    different splits are both preserved.
    """
    eval_by_unit: dict[str, dict[str, Any]] = {}
    for ev in eval_events:
        unit = ev.get("unit", "")
        if unit not in eval_by_unit:
            eval_by_unit[unit] = {}
        eval_by_unit[unit].update(ev)

    results: list[dict[str, Any]] = []
    for pe in period_ends:
        unit = pe.get("unit", "")
        merged = dict(pe)
        merged.update(eval_by_unit.get(unit, {}))
        results.append(merged)
    return results


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

_FLOAT_METRICS = [
    "train_loss_final",
    "casm/plasticity",
    "casm/stability",
    "casm/contradiction_acc",
    "casm/routing_acc",
    "smf/plasticity",
    "smf/stability",
    "smf/sparsity",
    "plasticity",
    "stability",
    "token_f1",
    "routing_acc",
    "train_duration_sec",
]

_INT_METRICS = [
    "n_passages_trained",
    "n_contradiction_passages",
    "optimizer_steps_total",
    "casm/slots_total",
    "casm/slots_active",
    "casm/slots_branched",
    "smf/active_params",
]


def _fmt(value: Any) -> str:
    if value is None or (isinstance(value, float) and _is_nan(value)):
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _pct(value: Any) -> str:
    if value is None or (isinstance(value, float) and _is_nan(value)):
        return "—"
    return f"{value * 100:.1f}%"


def render_table(results: list[dict[str, Any]], method: str) -> str:
    """Render an ASCII table of per-period results."""
    if not results:
        return "(no results)"

    is_casm = method == "casm"
    is_smf = method == "smf"

    buf = io.StringIO()

    # Header
    cols = ["Period", "Loss", "Passages"]
    if is_casm or is_smf:
        cols += ["Plasticity", "Stability"]
    if is_casm:
        cols += ["Contradict%", "Routing%", "Slots"]
    if is_smf:
        cols += ["Sparsity%"]
    cols += ["Duration(s)"]

    col_w = [max(14, len(c)) for c in cols]

    def row_str(cells: list[str]) -> str:
        return "  ".join(c.ljust(w) for c, w in zip(cells, col_w))

    buf.write(row_str(cols) + "\n")
    buf.write("  ".join("-" * w for w in col_w) + "\n")

    plasticity_key = f"{method}/plasticity" if method in ("smf", "casm") else "plasticity"
    stability_key = f"{method}/stability" if method in ("smf", "casm") else "stability"

    for r in results:
        cells = [
            r.get("unit", "?"),
            _fmt(r.get("train_loss_final")),
            _fmt(r.get("n_passages_trained")),
        ]
        if is_casm or is_smf:
            cells += [
                _pct(r.get(plasticity_key)),
                _pct(r.get(stability_key)),
            ]
        if is_casm:
            cells += [
                _pct(r.get("casm/contradiction_acc")),
                _pct(r.get("casm/routing_acc")),
                _fmt(r.get("casm/slots_total")),
            ]
        if is_smf:
            cells += [_pct(r.get("smf/sparsity"))]
        cells += [_fmt(r.get("train_duration_sec"))]

        buf.write(row_str(cells) + "\n")

    # Summary row
    buf.write("  ".join("=" * w for w in col_w) + "\n")
    avg_loss = compute_average_metric(results, "train_loss_final")
    avg_plast = compute_average_metric(results, plasticity_key)
    avg_stab = compute_average_metric(results, stability_key)
    summary_cells = ["AVERAGE", _fmt(avg_loss), ""]
    if is_casm or is_smf:
        summary_cells += [_pct(avg_plast), _pct(avg_stab)]
    if is_casm:
        avg_contra = compute_average_metric(results, "casm/contradiction_acc")
        avg_route = compute_average_metric(results, "casm/routing_acc")
        summary_cells += [_pct(avg_contra), _pct(avg_route), ""]
    if is_smf:
        avg_sparse = compute_average_metric(results, "smf/sparsity")
        summary_cells += [_pct(avg_sparse)]
    summary_cells += [""]
    buf.write(row_str(summary_cells) + "\n")

    # Forgetting / BWT
    if len(results) > 1 and plasticity_key in results[0]:
        bwt = compute_backward_transfer(results, plasticity_key)
        buf.write(f"\nBackward Transfer (plasticity): {_fmt(bwt)}\n")
        buf.write(
            "  (negative = forgetting, positive = forward transfer)\n"
        )

    return buf.getvalue()


def render_json(results: list[dict[str, Any]]) -> str:
    return json.dumps(results, indent=2, default=str)


def render_csv(results: list[dict[str, Any]]) -> str:
    if not results:
        return ""
    keys = list(results[0].keys())
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=keys, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(results)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Report evaluation metrics for a training run.")
    parser.add_argument("--run-root", required=True, help="Path to the run root directory.")
    parser.add_argument(
        "--format",
        choices=["table", "json", "csv"],
        default="table",
        help="Output format (default: table).",
    )
    parser.add_argument(
        "--method",
        default=None,
        help="Override method name for column display (auto-detected if omitted).",
    )
    args = parser.parse_args(argv)

    run_root = Path(args.run_root)
    if not run_root.exists():
        print(f"ERROR: run root does not exist: {run_root}", file=sys.stderr)
        sys.exit(1)

    try:
        events = load_metrics_jsonl(run_root)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    period_ends = extract_period_end_events(events)
    eval_events = extract_eval_events(events)
    results = merge_period_results(period_ends, eval_events)

    if not results:
        print("No period_end events found in metrics log.  "
              "Has training completed at least one period?")
        sys.exit(0)

    # Auto-detect method from first result that carries one
    method = args.method
    if method is None:
        for r in results:
            m = r.get("method")
            if m:
                method = m
                break
        if method is None:
            # Infer from metric keys present
            first = results[0]
            if any(k.startswith("casm/") for k in first):
                method = "casm"
            elif any(k.startswith("smf/") for k in first):
                method = "smf"
            else:
                method = "full_ft"

    print(f"Run root : {run_root}")
    print(f"Method   : {method}")
    print(f"Periods  : {len(results)}")
    print()

    if args.format == "table":
        print(render_table(results, method))
    elif args.format == "json":
        print(render_json(results))
    elif args.format == "csv":
        print(render_csv(results))


if __name__ == "__main__":
    main()
