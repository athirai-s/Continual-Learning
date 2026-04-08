"""compare_runs.py – Side-by-side metric comparison across multiple runs.

Usage:
    python eval_and_metrics/compare_runs.py \\
        --runs checkpoints/run_full_ft checkpoints/run_smf checkpoints/run_casm \\
        --metric casm/plasticity smf/plasticity plasticity \\
        --format table

Each run root is expected to contain a metrics.jsonl file produced by
MetricsLogger.  The script prints a period-aligned table showing how each
run performed on the requested metrics.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path
from typing import Any

from eval_and_metrics.report import load_metrics_jsonl, extract_period_end_events, extract_eval_events, merge_period_results
from eval_and_metrics.metric_helpers import compute_average_metric, compute_backward_transfer, _is_nan


_DEFAULT_METRICS = [
    "train_loss_final",
    "casm/plasticity",
    "casm/stability",
    "casm/contradiction_acc",
    "casm/routing_acc",
    "smf/plasticity",
    "smf/stability",
    "plasticity",
    "stability",
]


def _fmt(v: Any) -> str:
    if v is None or (isinstance(v, float) and _is_nan(v)):
        return "—"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _pct(v: Any) -> str:
    if v is None or (isinstance(v, float) and _is_nan(v)):
        return "—"
    return f"{v * 100:.1f}%"


def is_pct_metric(key: str) -> bool:
    for kw in ("plasticity", "stability", "acc", "sparsity", "f1"):
        if kw in key:
            return True
    return False


def load_run(run_root: Path) -> tuple[str, list[dict[str, Any]]]:
    events = load_metrics_jsonl(run_root)
    period_ends = extract_period_end_events(events)
    eval_events = extract_eval_events(events)
    results = merge_period_results(period_ends, eval_events)
    # Infer method
    method = None
    for r in results:
        method = r.get("method")
        if method:
            break
    if method is None:
        first = results[0] if results else {}
        if any(k.startswith("casm/") for k in first):
            method = "casm"
        elif any(k.startswith("smf/") for k in first):
            method = "smf"
        else:
            method = run_root.name
    return method or run_root.name, results


def render_comparison_table(
    runs: list[tuple[str, list[dict[str, Any]]]],
    metrics: list[str],
) -> str:
    """One column per (run × metric), one row per period."""
    # Collect all period names in order
    period_order: list[str] = []
    seen: set[str] = set()
    for _, results in runs:
        for r in results:
            u = r.get("unit", "?")
            if u not in seen:
                period_order.append(u)
                seen.add(u)

    # Filter metrics to only those actually present in at least one run
    active_metrics: list[str] = []
    for m in metrics:
        for _, results in runs:
            if any(m in r for r in results):
                active_metrics.append(m)
                break

    if not active_metrics:
        return "(none of the requested metrics were found in any run)"

    buf = io.StringIO()

    # Header: Period | run1/metric1 | run1/metric2 | run2/metric1 ...
    col_headers = ["Period"]
    for run_name, _ in runs:
        for m in active_metrics:
            col_headers.append(f"{run_name}\n{m}")

    # Flatten to single-line headers
    flat_headers = ["Period"] + [
        f"{rn}/{m.split('/')[-1]}"
        for rn, _ in runs
        for m in active_metrics
    ]

    col_w = [max(14, len(h)) for h in flat_headers]

    def row_str(cells: list[str]) -> str:
        return "  ".join(c.ljust(w) for c, w in zip(cells, col_w))

    buf.write(row_str(flat_headers) + "\n")
    buf.write("  ".join("-" * w for w in col_w) + "\n")

    # Index results by unit
    run_by_unit: list[dict[str, dict[str, Any]]] = []
    for _, results in runs:
        by_unit = {r.get("unit", "?"): r for r in results}
        run_by_unit.append(by_unit)

    for period in period_order:
        cells = [period]
        for i, (run_name, _) in enumerate(runs):
            r = run_by_unit[i].get(period, {})
            for m in active_metrics:
                v = r.get(m)
                cells.append(_pct(v) if is_pct_metric(m) else _fmt(v))
        buf.write(row_str(cells) + "\n")

    # Average row
    buf.write("  ".join("=" * w for w in col_w) + "\n")
    avg_cells = ["AVERAGE"]
    for _, results in runs:
        for m in active_metrics:
            avg = compute_average_metric(results, m)
            avg_cells.append(_pct(avg) if is_pct_metric(m) else _fmt(avg))
    buf.write(row_str(avg_cells) + "\n")

    # BWT row
    bwt_cells = ["BWT"]
    for _, results in runs:
        for m in active_metrics:
            if is_pct_metric(m):
                bwt = compute_backward_transfer(results, m)
                bwt_cells.append(_fmt(bwt))
            else:
                bwt_cells.append("—")
    buf.write(row_str(bwt_cells) + "\n")
    buf.write("  BWT = Backward Transfer: negative = forgetting\n")

    return buf.getvalue()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Compare evaluation metrics across multiple training runs."
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        metavar="RUN_ROOT",
        help="Paths to run root directories.",
    )
    parser.add_argument(
        "--metric",
        nargs="+",
        default=_DEFAULT_METRICS,
        metavar="METRIC_KEY",
        help="Metric keys to compare (default: plasticity, stability, contradiction_acc, routing_acc, loss).",
    )
    parser.add_argument(
        "--format",
        choices=["table", "json"],
        default="table",
    )
    args = parser.parse_args(argv)

    runs: list[tuple[str, list[dict[str, Any]]]] = []
    for run_path_str in args.runs:
        run_path = Path(run_path_str)
        if not run_path.exists():
            print(f"WARNING: run root not found, skipping: {run_path}", file=sys.stderr)
            continue
        try:
            name, results = load_run(run_path)
            runs.append((name, results))
            print(f"Loaded {len(results)} period(s) from {run_path} (method={name})")
        except FileNotFoundError as e:
            print(f"WARNING: {e}", file=sys.stderr)

    if not runs:
        print("No valid runs found.", file=sys.stderr)
        sys.exit(1)

    print()
    if args.format == "table":
        print(render_comparison_table(runs, args.metric))
    elif args.format == "json":
        out = []
        for name, results in runs:
            out.append({"run": name, "periods": results})
        print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
