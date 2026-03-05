import copy
from collections import defaultdict
from typing import Optional
from .casf_types import Probe, EvalResult
from .dataset import TemporalDataset


def _token_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = prediction.lower().split()
    gt_tokens = ground_truth.lower().split()
    if not pred_tokens or not gt_tokens:
        return 0.0
    common = set(pred_tokens) & set(gt_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


class TemporalEvaluator:

    def score_probe(self, model, probe: Probe) -> tuple[bool, float]:
        """Score a single probe. Returns (exact_match, token_f1)."""
        output = model.generate(probe.prompt)
        exact = probe.ground_truth.lower() in output.lower()
        f1 = _token_f1(output, probe.ground_truth)
        return exact, f1

    def evaluate(
        self,
        model,
        dataset: TemporalDataset,
        split: Optional[str] = None,
    ) -> EvalResult:
        probes = dataset.get_probes(split)
        if not probes:
            return EvalResult(plasticity=0.0, stability=0.0, token_f1=0.0, n_correct=0, n_total=0)

        changed_correct, changed_total = 0, 0
        unchanged_correct, unchanged_total = 0, 0
        total_f1, n_correct = 0.0, 0
        per_relation: dict[str, list[bool]] = defaultdict(list)

        for probe in probes:
            exact, f1 = self.score_probe(model, probe)
            total_f1 += f1
            if exact:
                n_correct += 1
            per_relation[probe.relation].append(exact)

            if probe.is_changed:
                changed_total += 1
                if exact:
                    changed_correct += 1
            else:
                unchanged_total += 1
                if exact:
                    unchanged_correct += 1

        plasticity = changed_correct / changed_total if changed_total else 0.0
        stability = unchanged_correct / unchanged_total if unchanged_total else 0.0
        avg_f1 = total_f1 / len(probes)
        per_rel_acc = {rel: sum(hits) / len(hits) for rel, hits in per_relation.items()}

        return EvalResult(
            plasticity=plasticity,
            stability=stability,
            token_f1=avg_f1,
            n_correct=n_correct,
            n_total=len(probes),
            per_relation=per_rel_acc,
        )

    def evaluate_versioned(
        self,
        model,
        dataset: TemporalDataset,
        query_period: str,
        fact_period: str,
    ) -> EvalResult:
        all_probes = dataset.get_probes()
        period_probes = [
            p for p in all_probes
            if p.valid_from == fact_period or (
                p.valid_from and p.valid_from <= fact_period
                and (p.valid_until is None or p.valid_until >= fact_period)
            )
        ]

        if not period_probes:
            return EvalResult(plasticity=0.0, stability=0.0, token_f1=0.0,
                              n_correct=0, n_total=0, routing_acc=0.0)

        correct_routing = 0
        total_f1, n_correct = 0.0, 0
        per_relation: dict[str, list[bool]] = defaultdict(list)

        for orig_probe in period_probes:
            aug_probe = copy.copy(orig_probe)
            aug_probe.prompt = f"[As of {query_period}] {orig_probe.prompt}"
            exact, f1 = self.score_probe(model, aug_probe)
            total_f1 += f1
            if exact:
                n_correct += 1
                correct_routing += 1
            per_relation[orig_probe.relation].append(exact)

        n = len(period_probes)
        per_rel_acc = {rel: sum(hits) / len(hits) for rel, hits in per_relation.items()}

        return EvalResult(
            plasticity=0.0,
            stability=0.0,
            token_f1=total_f1 / n,
            n_correct=n_correct,
            n_total=n,
            per_relation=per_rel_acc,
            routing_acc=correct_routing / n,
        )

    def evaluate_contradiction(self, model, dataset: TemporalDataset) -> EvalResult:
        """Evaluate only is_contradiction=True probes."""
        all_probes = dataset.get_probes()
        contradiction_probes = [p for p in all_probes if p.is_contradiction]

        if not contradiction_probes:
            return EvalResult(plasticity=0.0, stability=0.0, token_f1=0.0,
                              n_correct=0, n_total=0)

        total_f1, n_correct = 0.0, 0
        per_relation: dict[str, list[bool]] = defaultdict(list)

        for probe in contradiction_probes:
            exact, f1 = self.score_probe(model, probe)
            total_f1 += f1
            if exact:
                n_correct += 1
            per_relation[probe.relation].append(exact)

        n = len(contradiction_probes)
        per_rel_acc = {rel: sum(hits) / len(hits) for rel, hits in per_relation.items()}

        return EvalResult(
            plasticity=n_correct / n,
            stability=0.0,
            token_f1=total_f1 / n,
            n_correct=n_correct,
            n_total=n,
            per_relation=per_rel_acc,
        )