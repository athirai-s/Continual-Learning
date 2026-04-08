"""tests/test_evaluation.py

Unit tests for eval_and_metrics/evaluation.py.
All tests use lightweight stubs — no real model or GPU required.
"""

from __future__ import annotations

import math
import sys
import types
from unittest.mock import MagicMock

import torch
import pytest

# ---------------------------------------------------------------------------
# Stubs so imports work without the full training package
# ---------------------------------------------------------------------------

def _stub(name):
    m = types.ModuleType(name)
    sys.modules[name] = m
    return m

_cda = _stub("casf_dataset_api")

class _FakeRegistry:
    def __init__(self): self._slots = []

class _FakeProbe:
    def __init__(self, prompt, ground_truth, is_contradiction=False, valid_from=None):
        self.prompt = prompt
        self.ground_truth = ground_truth   # matches TemporalWikiDataset / run_eval.py
        self.is_contradiction = is_contradiction
        self.valid_from = valid_from
        self.context = prompt

_cda.TemporalDataset = object
_cda.MemoryRegistry = _FakeRegistry

for name in ["training", "training.smf_model", "training.casm_model"]:
    sys.modules.setdefault(name, types.ModuleType(name))

import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from eval_and_metrics.evaluation import (
    _token_f1,
    _get_ground_truth,
    _score_probe,
    compute_plasticity,
    compute_stability,
    compute_contradiction_acc,
    compute_probe_scores,
    evaluate_period,
)
from eval_and_metrics.metric_helpers import token_f1


# ---------------------------------------------------------------------------
# _token_f1 (internal — must match run_eval.py behaviour)
# ---------------------------------------------------------------------------

class TestTokenF1Internal:
    def test_identical(self):
        assert _token_f1("the cat sat", "the cat sat") == pytest.approx(1.0)

    def test_disjoint(self):
        assert _token_f1("apple orange", "dog cat") == pytest.approx(0.0)

    def test_case_insensitive(self):
        assert _token_f1("Paris", "paris") == pytest.approx(1.0)

    def test_empty_pred(self):
        assert _token_f1("", "reference") == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _get_ground_truth — field name resolution
# ---------------------------------------------------------------------------

class TestGetGroundTruth:
    def test_ground_truth_field(self):
        """run_eval.py uses probe.ground_truth — must be resolved first."""
        p = _FakeProbe("q", "correct answer")
        assert _get_ground_truth(p) == "correct answer"

    def test_expected_answer_fallback(self):
        p = MagicMock(spec=[])
        p.expected_answer = "expected"
        assert _get_ground_truth(p) == "expected"

    def test_answer_fallback(self):
        p = MagicMock(spec=[])
        p.answer = "ans"
        assert _get_ground_truth(p) == "ans"

    def test_no_field_returns_empty(self):
        p = MagicMock(spec=[])
        assert _get_ground_truth(p) == ""


# ---------------------------------------------------------------------------
# _score_probe — mirrors run_eval.py's exact/contains/f1 columns
# ---------------------------------------------------------------------------

class TestScoreProbe:
    def test_exact_match(self):
        s = _score_probe("paris", "paris")
        assert s["exact"] == 1.0
        assert s["contains"] == 1.0
        assert s["f1"] == pytest.approx(1.0)

    def test_contains_but_not_exact(self):
        s = _score_probe("the answer is paris", "paris")
        assert s["exact"] == 0.0
        assert s["contains"] == 1.0

    def test_no_match(self):
        s = _score_probe("london", "paris")
        assert s["exact"] == 0.0
        assert s["contains"] == 0.0
        assert s["f1"] == pytest.approx(0.0)

    def test_case_insensitive_exact(self):
        s = _score_probe("Paris", "paris")
        assert s["exact"] == 1.0


# ---------------------------------------------------------------------------
# Stub model/tokenizer factory
# ---------------------------------------------------------------------------

def _make_model_and_tokenizer(answers: list[str]):
    """Return stubs whose generate() cycles through ``answers``."""
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 2
    tokenizer.return_value = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
    }
    _gen_count = [0]
    def _generate(**kwargs):
        idx = _gen_count[0] % len(answers)
        _gen_count[0] += 1
        return torch.tensor([[1, 2, 3, 10 + idx]])
    _dec_count = [0]
    def _decode(ids, **kwargs):
        idx = _dec_count[0] % len(answers)
        _dec_count[0] += 1
        return answers[idx]
    tokenizer.decode = _decode
    model = MagicMock()
    model.eval = MagicMock()
    model.train = MagicMock()
    model.generate = _generate
    out = MagicMock()
    out.loss = torch.tensor(1.0)
    model.return_value = out
    return model, tokenizer


# ---------------------------------------------------------------------------
# compute_probe_scores
# ---------------------------------------------------------------------------

class TestComputeProbeScores:
    def test_perfect_match_all_three_columns(self):
        probes = [_FakeProbe("Q", "paris"), _FakeProbe("Q2", "rome")]
        model, tok = _make_model_and_tokenizer(["paris", "rome"])
        s = compute_probe_scores(model, tok, probes, torch.device("cpu"))
        assert s["exact"] == pytest.approx(1.0)
        assert s["contains"] == pytest.approx(1.0)
        assert s["f1"] == pytest.approx(1.0)

    def test_no_match(self):
        probes = [_FakeProbe("Q", "paris")]
        model, tok = _make_model_and_tokenizer(["london"])
        s = compute_probe_scores(model, tok, probes, torch.device("cpu"))
        assert s["exact"] == pytest.approx(0.0)
        assert s["f1"] == pytest.approx(0.0)

    def test_empty_probes_returns_nan(self):
        model, tok = _make_model_and_tokenizer(["x"])
        s = compute_probe_scores(model, tok, [], torch.device("cpu"))
        assert math.isnan(s["f1"])
        assert math.isnan(s["exact"])
        assert math.isnan(s["contains"])


# ---------------------------------------------------------------------------
# compute_plasticity / compute_stability
# ---------------------------------------------------------------------------

class TestComputePlasticity:
    def test_perfect(self):
        probes = [_FakeProbe("Q", "new answer")]
        model, tok = _make_model_and_tokenizer(["new answer"])
        assert compute_plasticity(model, tok, probes, torch.device("cpu")) == pytest.approx(1.0)

    def test_zero(self):
        probes = [_FakeProbe("Q", "new answer")]
        model, tok = _make_model_and_tokenizer(["wrong"])
        assert compute_plasticity(model, tok, probes, torch.device("cpu")) == pytest.approx(0.0)

    def test_empty_is_nan(self):
        model, tok = _make_model_and_tokenizer(["x"])
        assert math.isnan(compute_plasticity(model, tok, [], torch.device("cpu")))


class TestComputeStability:
    def test_full_retention(self):
        probes = [_FakeProbe("Old Q", "old answer")]
        model, tok = _make_model_and_tokenizer(["old answer"])
        assert compute_stability(model, tok, probes, torch.device("cpu")) == pytest.approx(1.0)

    def test_full_forgetting(self):
        probes = [_FakeProbe("Old Q", "old answer")]
        model, tok = _make_model_and_tokenizer(["completely different"])
        assert compute_stability(model, tok, probes, torch.device("cpu")) == pytest.approx(0.0)

    def test_empty_is_nan(self):
        model, tok = _make_model_and_tokenizer(["x"])
        assert math.isnan(compute_stability(model, tok, [], torch.device("cpu")))


# ---------------------------------------------------------------------------
# compute_contradiction_acc
# ---------------------------------------------------------------------------

class TestComputeContradictionAcc:
    def test_correct_new_answer(self):
        probes = [
            _FakeProbe("Q1", "new answer", is_contradiction=True),
            _FakeProbe("Q2", "stable", is_contradiction=False),
        ]
        model, tok = _make_model_and_tokenizer(["new answer"])
        registry = _FakeRegistry()
        score = compute_contradiction_acc(model, tok, probes, registry, torch.device("cpu"))
        assert score == pytest.approx(1.0)

    def test_no_contradiction_probes_is_nan(self):
        probes = [_FakeProbe("Q", "A", is_contradiction=False)]
        model, tok = _make_model_and_tokenizer(["A"])
        assert math.isnan(
            compute_contradiction_acc(model, tok, probes, _FakeRegistry(), torch.device("cpu"))
        )

    def test_wrong_new_answer(self):
        probes = [_FakeProbe("Q", "correct new", is_contradiction=True)]
        model, tok = _make_model_and_tokenizer(["stale old answer"])
        score = compute_contradiction_acc(model, tok, probes, _FakeRegistry(), torch.device("cpu"))
        assert score == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# evaluate_period integration smoke tests
# ---------------------------------------------------------------------------

class TestEvaluatePeriod:
    def _make_dataset(self, changed, unchanged):
        ds = MagicMock()
        def get_probes(split=None):
            if split == "changed": return changed
            if split == "unchanged": return unchanged
            return changed
        ds.get_probes = get_probes
        return ds

    def test_casm_has_casm_keys(self):
        cfg = MagicMock(); cfg.method = "casm"
        changed = [_FakeProbe("Q1", "new ans", is_contradiction=True)]
        unchanged = [_FakeProbe("Q2", "old ans")]
        model, tok = _make_model_and_tokenizer(["new ans", "old ans"])
        result = evaluate_period(
            model=model, tokenizer=tok,
            dataset=self._make_dataset(changed, unchanged),
            cfg=cfg, unit="sep_oct", registry=_FakeRegistry(),
        )
        assert "casm/plasticity" in result
        assert "casm/stability" in result
        assert "casm/contradiction_acc" in result
        assert result["unit"] == "sep_oct"
        assert result["method"] == "casm"

    def test_casm_has_all_three_score_columns(self):
        """evaluate_period should expose exact/contains/f1 matching run_eval.py."""
        cfg = MagicMock(); cfg.method = "casm"
        changed = [_FakeProbe("Q", "paris")]
        model, tok = _make_model_and_tokenizer(["paris"])
        result = evaluate_period(
            model=model, tokenizer=tok,
            dataset=self._make_dataset(changed, []),
            cfg=cfg, unit="aug_sep", registry=_FakeRegistry(),
        )
        assert "casm/changed_exact" in result
        assert "casm/changed_contains" in result
        assert "casm/changed_f1" in result

    def test_smf_no_casm_keys(self):
        cfg = MagicMock(); cfg.method = "smf"
        changed = [_FakeProbe("Q", "ans")]
        model, tok = _make_model_and_tokenizer(["ans"])
        result = evaluate_period(
            model=model, tokenizer=tok,
            dataset=self._make_dataset(changed, []),
            cfg=cfg, unit="aug_sep", registry=_FakeRegistry(),
        )
        assert "smf/plasticity" in result
        assert "casm/contradiction_acc" not in result
        assert "casm/routing_acc" not in result

    def test_probe_counts_correct(self):
        cfg = MagicMock(); cfg.method = "full_ft"
        changed = [_FakeProbe("Q", "A")]
        unchanged = [_FakeProbe("Q2", "B"), _FakeProbe("Q3", "C")]
        model, tok = _make_model_and_tokenizer(["A"])
        result = evaluate_period(
            model=model, tokenizer=tok,
            dataset=self._make_dataset(changed, unchanged),
            cfg=cfg, unit="oct_nov", registry=_FakeRegistry(),
        )
        assert result["n_changed_probes"] == 1
        assert result["n_unchanged_probes"] == 2

    def test_eval_duration_present_and_nonneg(self):
        cfg = MagicMock(); cfg.method = "full_ft"
        model, tok = _make_model_and_tokenizer(["x"])
        result = evaluate_period(
            model=model, tokenizer=tok,
            dataset=self._make_dataset([], []),
            cfg=cfg, unit="nov_dec", registry=_FakeRegistry(),
        )
        assert "eval_duration_sec" in result
        assert result["eval_duration_sec"] >= 0
