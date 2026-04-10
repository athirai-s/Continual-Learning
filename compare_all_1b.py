"""
compare_all_1b.py — Compare all 4 methods across all periods for the 1B model.

Shows:
  1. ASCII table  — period-by-period training loss side by side
  2. Period 1 retention table — from step3_eval_results.json
     (proves catastrophic forgetting + shows CASM helps)
  3. Plasticity / Stability / Loss / BWT plots

Run in Colab or on CARC after all step2 jobs + step3 eval are complete:
    python compare_all_1b.py
"""

import json
import os
from pathlib import Path

from eval_and_metrics.report import load_metrics_jsonl, extract_period_end_events, extract_eval_events, merge_period_results
from eval_and_metrics.compare_runs import render_comparison_table, load_run
from eval_and_metrics.metric_helpers import compute_backward_transfer, compute_average_metric

CHECKPOINT_DIR = Path("/scratch1/ramyakri/checkpoints")

# All v2 runs train on all 4 periods: aug_sep → sep_oct → oct_nov → nov_dec
RUNS = {
    "full_ft" : CHECKPOINT_DIR / "step2_fullft_1b",
    "lora"    : CHECKPOINT_DIR / "step2_lora_1b",
    "smf"     : CHECKPOINT_DIR / "step2_smf_1b",
    "casm"    : CHECKPOINT_DIR / "step2_casm_1b",
}

PERIODS = ["aug_sep", "sep_oct", "oct_nov", "nov_dec"]

COLORS = {
    "full_ft" : "#e74c3c",
    "lora"    : "#f39c12",
    "smf"     : "#2ecc71",
    "casm"    : "#3498db",
}

# ---------------------------------------------------------------------------
# Load all runs
# ---------------------------------------------------------------------------
loaded = {}
for method, path in RUNS.items():
    if not path.exists():
        print(f"WARNING: {path} not found — skipping {method}")
        continue
    try:
        _, results = load_run(path)
        loaded[method] = results
        print(f"Loaded {method}: {len(results)} period(s)")
    except FileNotFoundError as e:
        print(f"WARNING: {e}")

if not loaded:
    print("No runs found. Make sure all step2 jobs have completed.")
    raise SystemExit

# ---------------------------------------------------------------------------
# 1. ASCII comparison table — training loss per period
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("COMPARISON TABLE — Training Loss, All Methods, All Periods")
print("=" * 80)

run_list = [(m, r) for m, r in loaded.items()]
metrics = [
    "train_loss_final",
    "plasticity", "stability",
    "smf/plasticity", "smf/stability",
    "casm/plasticity", "casm/stability",
    "casm/contradiction_acc", "casm/routing_acc",
]
print(render_comparison_table(run_list, metrics))

# ---------------------------------------------------------------------------
# 2. Period 1 RETENTION TABLE — catastrophic forgetting proof
# ---------------------------------------------------------------------------
STEP3_RESULTS_PATH = "/project2/jieyuz_1727/Continual-Learning/step3_eval_results.json"

print("\n" + "=" * 80)
print("PERIOD 1 RETENTION — After Training on Periods 2-4")
print("Catastrophic Forgetting Proof: does the model still know Period 1 facts?")
print("  pretrain_p1 = upper bound (trained only on P1, no forgetting possible)")
print("  full_ft     = should score LOW  (all weights updated → P1 overwritten)")
print("  casm        = should score HIGH (backbone frozen  → P1 preserved)")
print("=" * 80)

if os.path.exists(STEP3_RESULTS_PATH):
    with open(STEP3_RESULTS_PATH) as f:
        step3 = json.load(f)

    METHOD_ORDER = ["pretrain_p1", "full_ft", "lora", "smf", "casm"]
    col_w = [15, 12, 8, 10, 10, 8]
    header = ["Method", "Split", "N", "Exact", "Contains", "F1"]

    def row(cells):
        return "  ".join(str(c).ljust(w) for c, w in zip(cells, col_w))

    print(row(header))
    print("  ".join("-" * w for w in col_w))

    for name in METHOD_ORDER:
        if name not in step3:
            continue
        for split in ["changed", "unchanged"]:
            m = step3[name].get(split, {})
            if not m:
                continue
            print(row([
                name if split == "changed" else "",
                split,
                m.get("n", "—"),
                f"{m.get('exact', 0):.4f}",
                f"{m.get('contains', 0):.4f}",
                f"{m.get('f1', 0):.5f}",
            ]))
        print()

    # Forgetting delta: full_ft vs casm on changed probes
    if "full_ft" in step3 and "casm" in step3:
        ft_f1  = step3["full_ft"].get("changed", {}).get("f1", None)
        casm_f1 = step3["casm"].get("changed", {}).get("f1", None)
        p1_f1  = step3.get("pretrain_p1", {}).get("changed", {}).get("f1", None)
        print(f"  Forgetting delta (full_ft vs casm, changed F1):")
        if ft_f1 is not None and casm_f1 is not None:
            delta = casm_f1 - ft_f1
            print(f"    CASM retains {delta:+.4f} more F1 on P1 changed facts than full_ft")
        if p1_f1 is not None and ft_f1 is not None:
            forgetting = p1_f1 - ft_f1
            print(f"    full_ft forgot {forgetting:.4f} F1 relative to pretrain baseline")
        if p1_f1 is not None and casm_f1 is not None:
            forgetting_casm = p1_f1 - casm_f1
            print(f"    casm    forgot {forgetting_casm:.4f} F1 relative to pretrain baseline")
else:
    print(f"  (step3_eval_results.json not found at {STEP3_RESULTS_PATH})")
    print("  Run:  sbatch run_step3_eval_job.sh   then re-run this script.")

# ---------------------------------------------------------------------------
# 3. Plots
# ---------------------------------------------------------------------------
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams.update({"font.size": 11})

    def get_metric_by_period(results, key):
        by_unit = {r.get("unit", "?"): r.get(key) for r in results}
        return [by_unit.get(p) for p in PERIODS]

    def safe_vals(vals):
        return [v if v is not None else float("nan") for v in vals]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Continual Learning — 1B Model v2: Catastrophic Forgetting + CASM Retention",
                 fontsize=13, fontweight="bold")

    # --- Plot 1: Plasticity (per-period) ---
    ax = axes[0, 0]
    for method, results in loaded.items():
        key = f"{method}/plasticity" if method in ("smf", "casm") else "plasticity"
        vals = safe_vals(get_metric_by_period(results, key))
        ax.plot(PERIODS, [v * 100 if v == v else v for v in vals],
                marker="o", label=method, color=COLORS.get(method))
    ax.set_title("Plasticity — new fact learning %")
    ax.set_ylabel("%")
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Plot 2: Stability (per-period) ---
    ax = axes[0, 1]
    for method, results in loaded.items():
        key = f"{method}/stability" if method in ("smf", "casm") else "stability"
        vals = safe_vals(get_metric_by_period(results, key))
        ax.plot(PERIODS, [v * 100 if v == v else v for v in vals],
                marker="o", label=method, color=COLORS.get(method))
    ax.set_title("Stability — old fact retention %")
    ax.set_ylabel("%")
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Plot 3: Training Loss ---
    ax = axes[1, 0]
    for method, results in loaded.items():
        vals = safe_vals(get_metric_by_period(results, "train_loss_final"))
        ax.plot(PERIODS, vals, marker="o", label=method, color=COLORS.get(method))
    ax.set_title("Training Loss per Period")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Plot 4: Period 1 Retention bar chart (from step3) ---
    ax = axes[1, 1]
    if os.path.exists(STEP3_RESULTS_PATH):
        with open(STEP3_RESULTS_PATH) as f:
            step3_data = json.load(f)
        bar_methods = [m for m in ["pretrain_p1", "full_ft", "lora", "smf", "casm"]
                       if m in step3_data]
        bar_vals = [step3_data[m].get("changed", {}).get("f1", 0.0) for m in bar_methods]
        bar_colors = [COLORS.get(m, "#95a5a6") for m in bar_methods]
        bars = ax.bar(bar_methods, [v * 100 for v in bar_vals], color=bar_colors)
        ax.set_title("Period 1 Retention After P2-P4 Training\n(changed probes F1 — higher = less forgetting)")
        ax.set_ylabel("F1 %")
        ax.set_ylim(0, 100)
        for bar, val in zip(bars, bar_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{val*100:.1f}%", ha="center", va="bottom", fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
    else:
        ax.text(0.5, 0.5, "Run step3 eval first\n(sbatch run_step3_eval_job.sh)",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)
        ax.set_title("Period 1 Retention (step3 eval needed)")

    plt.tight_layout()
    out_path = "/scratch1/ramyakri/checkpoints/compare_1b.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {out_path}")
    plt.show()

except ImportError:
    print("matplotlib not available — skipping plots")

# ---------------------------------------------------------------------------
# 4. Summary stats
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("SUMMARY — Average Training Metrics Across All Periods")
print("=" * 80)
for method, results in loaded.items():
    p_key = f"{method}/plasticity" if method in ("smf", "casm") else "plasticity"
    s_key = f"{method}/stability" if method in ("smf", "casm") else "stability"
    avg_p = compute_average_metric(results, p_key)
    avg_s = compute_average_metric(results, s_key)
    avg_l = compute_average_metric(results, "train_loss_final")
    bwt   = compute_backward_transfer(results, p_key)

    def _pct(v: float) -> str:
        return f"{v*100:.1f}%" if v == v else "nan%"

    print(f"  {method:<10} loss={avg_l:.4f}  plasticity={_pct(avg_p)}  stability={_pct(avg_s)}  BWT={_pct(bwt)}")
