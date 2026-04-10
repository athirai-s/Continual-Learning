"""
compare_all_1b.py — Compare all 4 methods across all periods for the 1B model.

Shows:
  1. ASCII table  — period-by-period side by side
  2. Plasticity plot  — how well each method learns new facts
  3. Stability plot   — how well each method retains old facts
  4. Loss plot        — training loss per period
  5. BWT bar chart    — backward transfer (forgetting summary)

Run in Colab after all step2 jobs are complete:
    !python compare_all_1b.py
"""

from pathlib import Path
from eval_and_metrics.report import load_metrics_jsonl, extract_period_end_events, extract_eval_events, merge_period_results
from eval_and_metrics.compare_runs import render_comparison_table, load_run
from eval_and_metrics.metric_helpers import compute_backward_transfer, compute_average_metric

CHECKPOINT_DIR = Path("/content/drive/MyDrive/checkpoints")

RUNS = {
    "full_ft" : CHECKPOINT_DIR / "step2_fullft_1b",
    "lora"    : CHECKPOINT_DIR / "step2_lora_1b",
    "smf"     : CHECKPOINT_DIR / "step2_smf_1b",
    "casm"    : CHECKPOINT_DIR / "step2_casm_1b",
}

PERIODS = ["sep_oct", "oct_nov", "nov_dec"]

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
# 1. ASCII comparison table
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("COMPARISON TABLE — All Methods, All Periods")
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
# 2. Plots
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
    fig.suptitle("Continual Learning — 1B Model: All Methods, Periods 2-4", fontsize=14, fontweight="bold")

    # --- Plot 1: Plasticity ---
    ax = axes[0, 0]
    for method, results in loaded.items():
        key = f"{method}/plasticity" if method in ("smf", "casm") else "plasticity"
        vals = safe_vals(get_metric_by_period(results, key))
        ax.plot(PERIODS, [v * 100 if v == v else v for v in vals],
                marker="o", label=method, color=COLORS.get(method))
    ax.set_title("Plasticity (new fact learning %)")
    ax.set_ylabel("%")
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Plot 2: Stability ---
    ax = axes[0, 1]
    for method, results in loaded.items():
        key = f"{method}/stability" if method in ("smf", "casm") else "stability"
        vals = safe_vals(get_metric_by_period(results, key))
        ax.plot(PERIODS, [v * 100 if v == v else v for v in vals],
                marker="o", label=method, color=COLORS.get(method))
    ax.set_title("Stability (old fact retention %)")
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

    # --- Plot 4: BWT bar chart ---
    ax = axes[1, 1]
    methods_bwt = []
    bwt_vals = []
    for method, results in loaded.items():
        key = f"{method}/plasticity" if method in ("smf", "casm") else "plasticity"
        bwt = compute_backward_transfer(results, key)
        if bwt == bwt:  # not nan
            methods_bwt.append(method)
            bwt_vals.append(bwt * 100)
    bars = ax.bar(methods_bwt, bwt_vals,
                  color=[COLORS.get(m) for m in methods_bwt])
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("Backward Transfer (BWT)\nNegative = Forgetting")
    ax.set_ylabel("BWT (%)")
    for bar, val in zip(bars, bwt_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out_path = "/content/drive/MyDrive/checkpoints/compare_1b.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {out_path}")
    plt.show()

except ImportError:
    print("matplotlib not available — skipping plots")

# ---------------------------------------------------------------------------
# 3. Summary stats
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("SUMMARY — Average across all periods")
print("=" * 80)
for method, results in loaded.items():
    p_key = f"{method}/plasticity" if method in ("smf", "casm") else "plasticity"
    s_key = f"{method}/stability" if method in ("smf", "casm") else "stability"
    avg_p = compute_average_metric(results, p_key)
    avg_s = compute_average_metric(results, s_key)
    avg_l = compute_average_metric(results, "train_loss_final")
    bwt   = compute_backward_transfer(results, p_key)
    print(f"  {method:<10} loss={avg_l:.4f}  plasticity={avg_p*100:.1f}%  stability={avg_s*100:.1f}%  BWT={bwt*100:.1f}%")
