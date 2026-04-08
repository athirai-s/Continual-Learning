"""eval_and_metrics – PR 6: Evaluation and metrics for SMF and CASM.

Public API
----------
evaluate_period(...)
    Full per-period evaluation returning a flat metric dict.

run_period_evaluation(...)
    Thin adapter called by train_runner.run_training() when
    cfg.eval_after_each_period is True.

metric_helpers
    Standalone utilities: token_f1, compute_retention, compute_forgetting,
    compute_backward_transfer, compute_version_coverage, etc.

CLI scripts
-----------
report.py          Per-run period table / JSON / CSV report.
compare_runs.py    Side-by-side comparison across multiple runs.
"""

from .evaluation import evaluate_period
from .evaluation_runner import run_period_evaluation
from . import metric_helpers

__all__ = [
    "evaluate_period",
    "run_period_evaluation",
    "metric_helpers",
]
