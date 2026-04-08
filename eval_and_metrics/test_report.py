"""tests/test_report.py

Tests for eval_and_metrics/report.py – report generation from metrics.jsonl.
No model or real run root needed; we write synthetic JSONL fixtures.
"""

from __future__ import annotations

import json
import pathlib
import sys
import tempfile

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from eval_and_metrics.report import (
    load_metrics_jsonl,
    extract_period_end_events,
    extract_eval_events,
    merge_period_results,
    render_table,
    render_json,
    render_csv,
    main,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_PERIOD_EVENTS = [
    {
        "event_type": "period_end",
        "unit": "aug_sep",
        "train_loss_final": 2.5,
        "n_passages_trained": 100,
        "n_contradiction_passages": 5,
        "optimizer_steps_total": 50,
        "train_duration_sec": 12.3,
        "method": "casm",
    },
    {
        "event_type": "period_end",
        "unit": "sep_oct",
        "train_loss_final": 2.1,
        "n_passages_trained": 110,
        "n_contradiction_passages": 8,
        "optimizer_steps_total": 55,
        "train_duration_sec": 14.1,
        "method": "casm",
    },
]

_EVAL_EVENTS = [
    {
        "event_type": "evaluation",
        "unit": "aug_sep",
        "method": "casm",
        "casm/plasticity": 0.70,
        "casm/stability": 0.80,
        "casm/contradiction_acc": 0.60,
        "casm/routing_acc": 0.75,
        "n_changed_probes": 5,
        "n_unchanged_probes": 10,
    },
    {
        "event_type": "evaluation",
        "unit": "sep_oct",
        "method": "casm",
        "casm/plasticity": 0.75,
        "casm/stability": 0.72,
        "casm/contradiction_acc": 0.65,
        "casm/routing_acc": 0.80,
        "n_changed_probes": 8,
        "n_unchanged_probes": 12,
    },
]


def _write_jsonl(path: pathlib.Path, events: list[dict]) -> None:
    with open(path, "w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")


# ---------------------------------------------------------------------------
# load_metrics_jsonl
# ---------------------------------------------------------------------------

class TestLoadMetricsJsonl:
    def test_loads_all_events(self, tmp_path):
        _write_jsonl(tmp_path / "metrics.jsonl", _PERIOD_EVENTS + _EVAL_EVENTS)
        events = load_metrics_jsonl(tmp_path)
        assert len(events) == len(_PERIOD_EVENTS) + len(_EVAL_EVENTS)

    def test_raises_when_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_metrics_jsonl(tmp_path)

    def test_skips_malformed_lines(self, tmp_path):
        path = tmp_path / "metrics.jsonl"
        with open(path, "w") as f:
            f.write(json.dumps(_PERIOD_EVENTS[0]) + "\n")
            f.write("NOT JSON\n")
            f.write(json.dumps(_PERIOD_EVENTS[1]) + "\n")
        events = load_metrics_jsonl(tmp_path)
        assert len(events) == 2


# ---------------------------------------------------------------------------
# extract helpers
# ---------------------------------------------------------------------------

class TestExtractHelpers:
    def test_period_ends(self):
        events = _PERIOD_EVENTS + _EVAL_EVENTS
        pe = extract_period_end_events(events)
        assert all(e["event_type"] == "period_end" for e in pe)
        assert len(pe) == 2

    def test_eval_events(self):
        events = _PERIOD_EVENTS + _EVAL_EVENTS
        ev = extract_eval_events(events)
        assert all(e["event_type"] == "evaluation" for e in ev)
        assert len(ev) == 2


# ---------------------------------------------------------------------------
# merge_period_results
# ---------------------------------------------------------------------------

class TestMergePeriodResults:
    def test_merges_eval_into_period(self):
        results = merge_period_results(_PERIOD_EVENTS, _EVAL_EVENTS)
        assert len(results) == 2
        r = results[0]
        assert r["unit"] == "aug_sep"
        assert "train_loss_final" in r
        assert "casm/plasticity" in r

    def test_missing_eval_leaves_training_fields(self):
        results = merge_period_results(_PERIOD_EVENTS, [])
        assert results[0]["train_loss_final"] == 2.5
        assert "casm/plasticity" not in results[0]


# ---------------------------------------------------------------------------
# render_table
# ---------------------------------------------------------------------------

class TestRenderTable:
    def _results(self):
        return merge_period_results(_PERIOD_EVENTS, _EVAL_EVENTS)

    def test_contains_period_names(self):
        out = render_table(self._results(), "casm")
        assert "aug_sep" in out
        assert "sep_oct" in out

    def test_casm_columns_present(self):
        out = render_table(self._results(), "casm")
        assert "Plasticity" in out
        assert "Stability" in out
        assert "Contradict%" in out
        assert "Routing%" in out

    def test_smf_columns_present(self):
        smf_results = [
            {
                "unit": "aug_sep",
                "train_loss_final": 2.0,
                "n_passages_trained": 50,
                "train_duration_sec": 5.0,
                "smf/plasticity": 0.7,
                "smf/stability": 0.8,
                "smf/sparsity": 0.9,
                "method": "smf",
            }
        ]
        out = render_table(smf_results, "smf")
        assert "Plasticity" in out
        assert "Sparsity%" in out
        assert "Contradict%" not in out

    def test_empty_returns_placeholder(self):
        out = render_table([], "casm")
        assert "no results" in out

    def test_average_row_present(self):
        out = render_table(self._results(), "casm")
        assert "AVERAGE" in out


# ---------------------------------------------------------------------------
# render_json / render_csv
# ---------------------------------------------------------------------------

class TestRenderJsonCsv:
    def _results(self):
        return merge_period_results(_PERIOD_EVENTS, _EVAL_EVENTS)

    def test_render_json_is_valid(self):
        out = render_json(self._results())
        parsed = json.loads(out)
        assert len(parsed) == 2

    def test_render_csv_has_header(self):
        out = render_csv(self._results())
        lines = out.strip().split("\n")
        assert len(lines) >= 2
        assert "unit" in lines[0]

    def test_render_csv_empty(self):
        assert render_csv([]) == ""


# ---------------------------------------------------------------------------
# CLI main()
# ---------------------------------------------------------------------------

class TestMainCli:
    def test_table_output(self, tmp_path, capsys):
        _write_jsonl(tmp_path / "metrics.jsonl", _PERIOD_EVENTS + _EVAL_EVENTS)
        main(["--run-root", str(tmp_path), "--format", "table"])
        captured = capsys.readouterr()
        assert "aug_sep" in captured.out

    def test_json_output(self, tmp_path, capsys):
        _write_jsonl(tmp_path / "metrics.jsonl", _PERIOD_EVENTS + _EVAL_EVENTS)
        main(["--run-root", str(tmp_path), "--format", "json"])
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert isinstance(parsed, list)

    def test_missing_run_root_exits(self, tmp_path):
        with pytest.raises(SystemExit):
            main(["--run-root", str(tmp_path / "nonexistent")])

    def test_method_override(self, tmp_path, capsys):
        _write_jsonl(tmp_path / "metrics.jsonl", _PERIOD_EVENTS)
        main(["--run-root", str(tmp_path), "--method", "smf"])
        captured = capsys.readouterr()
        assert "smf" in captured.out
