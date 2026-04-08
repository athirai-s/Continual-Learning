"""tests/test_metric_helpers.py

Unit tests for eval_and_metrics/metric_helpers.py.
No model or GPU required.
"""

from __future__ import annotations

import math
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from eval_and_metrics.metric_helpers import (
    token_f1,
    exact_match,
    compute_retention,
    compute_forgetting,
    compute_average_metric,
    compute_backward_transfer,
    compute_version_coverage,
    compute_slot_utilisation,
    _mean,
    _is_nan,
)


# ---------------------------------------------------------------------------
# token_f1
# ---------------------------------------------------------------------------

class TestTokenF1:
    def test_perfect(self):
        assert token_f1("yes it is", "yes it is") == 1.0

    def test_zero(self):
        assert token_f1("foo bar", "baz qux") == 0.0

    def test_symmetric(self):
        a, b = "the cat", "cat sat"
        assert abs(token_f1(a, b) - token_f1(b, a)) < 1e-9

    def test_empty_pred(self):
        assert token_f1("", "reference") == 0.0

    def test_empty_ref(self):
        assert token_f1("prediction", "") == 0.0


# ---------------------------------------------------------------------------
# exact_match
# ---------------------------------------------------------------------------

class TestExactMatch:
    def test_match(self):
        assert exact_match("Paris", "paris") == 1.0

    def test_no_match(self):
        assert exact_match("London", "Paris") == 0.0

    def test_strip(self):
        assert exact_match("  yes ", "yes") == 1.0


# ---------------------------------------------------------------------------
# _mean
# ---------------------------------------------------------------------------

class TestMean:
    def test_basic(self):
        assert _mean([1.0, 2.0, 3.0]) == 2.0

    def test_empty(self):
        assert math.isnan(_mean([]))

    def test_single(self):
        assert _mean([5.0]) == 5.0


# ---------------------------------------------------------------------------
# _is_nan
# ---------------------------------------------------------------------------

class TestIsNan:
    def test_nan_float(self):
        assert _is_nan(float("nan"))

    def test_normal_float(self):
        assert not _is_nan(0.5)

    def test_none(self):
        assert not _is_nan(None)  # None is not NaN

    def test_string(self):
        assert not _is_nan("hello")


# ---------------------------------------------------------------------------
# compute_average_metric
# ---------------------------------------------------------------------------

class TestComputeAverageMetric:
    def _results(self, values: list[float]) -> list[dict]:
        return [{"score": v} for v in values]

    def test_basic(self):
        assert compute_average_metric(self._results([0.8, 0.6, 0.4]), "score") == 0.6

    def test_skips_nan(self):
        results = [{"score": 0.8}, {"score": float("nan")}, {"score": 0.6}]
        avg = compute_average_metric(results, "score")
        assert avg == pytest.approx(0.7)

    def test_missing_key(self):
        results = [{"other": 1.0}]
        assert math.isnan(compute_average_metric(results, "score"))

    def test_empty(self):
        assert math.isnan(compute_average_metric([], "score"))


# ---------------------------------------------------------------------------
# compute_retention
# ---------------------------------------------------------------------------

class TestComputeRetention:
    def _results(self):
        return [
            {"plasticity": 0.9},
            {"plasticity": 0.7},
            {"plasticity": 0.5},
        ]

    def test_after_second_period(self):
        results = self._results()
        # retention for period index 2 should average over indices 0 and 1
        ret = compute_retention(results, "plasticity", current_period_index=2)
        assert ret == pytest.approx(0.8)

    def test_first_period_no_earlier(self):
        results = self._results()
        ret = compute_retention(results, "plasticity", current_period_index=0)
        assert math.isnan(ret)

    def test_single_earlier(self):
        results = self._results()
        ret = compute_retention(results, "plasticity", current_period_index=1)
        assert ret == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# compute_forgetting
# ---------------------------------------------------------------------------

class TestComputeForgetting:
    def test_positive_forgetting(self):
        # Period 0 peaked at 0.9, current period the value dropped to 0.6
        results = [{"s": 0.9}, {"s": 0.8}, {"s": 0.6}]
        f = compute_forgetting(results, "s", current_period_index=2)
        assert f == pytest.approx(0.3)

    def test_no_forgetting(self):
        results = [{"s": 0.5}, {"s": 0.9}]
        f = compute_forgetting(results, "s", current_period_index=1)
        assert f == pytest.approx(-0.4)  # forward transfer

    def test_single_period_is_nan(self):
        results = [{"s": 0.8}]
        assert math.isnan(compute_forgetting(results, "s", current_period_index=0))


# ---------------------------------------------------------------------------
# compute_backward_transfer
# ---------------------------------------------------------------------------

class TestComputeBackwardTransfer:
    def test_negative_bwt(self):
        # After the last period, earlier values dropped
        results = [{"p": 0.9}, {"p": 0.7}, {"p": 0.5}]
        # BWT compares each early period's value to the *final* value in results
        bwt = compute_backward_transfer(results, "p")
        # final period value is 0.5; earlier: 0.9-0.5=-0.4, 0.7-0.5=-0.2 → mean -0.3
        assert bwt == pytest.approx(-0.3)

    def test_positive_bwt(self):
        results = [{"p": 0.5}, {"p": 0.7}, {"p": 0.9}]
        bwt = compute_backward_transfer(results, "p")
        # 0.9-0.5=0.4, 0.9-0.7=0.2 → mean 0.3
        assert bwt == pytest.approx(0.3)

    def test_single_period_is_nan(self):
        assert math.isnan(compute_backward_transfer([{"p": 0.8}], "p"))


# ---------------------------------------------------------------------------
# compute_version_coverage
# ---------------------------------------------------------------------------

class TestComputeVersionCoverage:
    def _make_slot(self, slot_id: int, valid_until=None, parent=None):
        s = type("Slot", (), {})()
        s.slot_id = slot_id
        s.valid_until = valid_until
        s.parent_slot_id = parent
        return s

    def _make_registry(self, slots):
        r = type("Registry", (), {})()
        r._slots = slots
        return r

    def test_basic_counts(self):
        slots = [
            self._make_slot(1),
            self._make_slot(2, valid_until="oct_nov"),
            self._make_slot(3, parent=1),
        ]
        r = self._make_registry(slots)
        cov = compute_version_coverage(r, "nov_dec")
        assert cov["casm/slots_total"] == 3
        assert cov["casm/slots_active"] == 2   # slot 2 is closed
        assert cov["casm/slots_closed"] == 1
        assert cov["casm/slots_branched"] == 1

    def test_empty_registry(self):
        r = self._make_registry([])
        cov = compute_version_coverage(r, "aug_sep")
        assert cov["casm/slots_total"] == 0


# ---------------------------------------------------------------------------
# compute_slot_utilisation
# ---------------------------------------------------------------------------

class TestComputeSlotUtilisation:
    def _make_registry(self, counts: list[int]):
        class _Slot:
            def __init__(self, c):
                self.usage_count = c
        r = type("Registry", (), {})()
        r._slots = [_Slot(c) for c in counts]
        return r

    def test_mean_usage(self):
        r = self._make_registry([4, 6])
        assert compute_slot_utilisation(r) == pytest.approx(5.0)

    def test_empty(self):
        r = self._make_registry([])
        assert math.isnan(compute_slot_utilisation(r))


# Need pytest.approx in scope for the classes above
import pytest
