"""evaluation.py – Per-period evaluation for SMF and CASM.

Computes plasticity, stability, contradiction_acc, and routing_acc after
each training period.  All scoring (exact, contains, f1) matches run_eval.py
so results are directly comparable without re-running that script.

Usage (called from train_runner.py via run_period_evaluation):
    result = evaluate_period(
        model=trainer.model,
        tokenizer=trainer.tokenizer,
        dataset=eval_dataset,
        cfg=cfg,
        unit=period_name,
        registry=trainer.registry,
    )
"""

from __future__ import annotations

import math
import time
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from casf_dataset_api import TemporalDataset, MemoryRegistry


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class _TextDataset(Dataset):
    def __init__(self, texts: list[str], tokenizer, max_length: int = 512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding="do_not_pad",
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = item["input_ids"].clone()
        mask = item.get("attention_mask")
        if mask is not None:
            item["labels"][mask == 0] = -100
        return item


def _collate(batch: list[dict[str, torch.Tensor]], pad_id: int) -> dict[str, torch.Tensor]:
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [x["input_ids"] for x in batch], batch_first=True, padding_value=pad_id
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        [x["attention_mask"] for x in batch], batch_first=True, padding_value=0
    )
    labels = torch.nn.utils.rnn.pad_sequence(
        [x["labels"] for x in batch], batch_first=True, padding_value=-100
    )
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def _compute_perplexity(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    device: torch.device,
    batch_size: int = 4,
    max_length: int = 512,
) -> float:
    """Returns perplexity over a list of texts (lower = better)."""
    if not texts:
        return float("nan")

    pad_id = getattr(tokenizer, "pad_token_id", 0) or 0
    ds = _TextDataset(texts, tokenizer, max_length=max_length)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: _collate(b, pad_id),
    )

    total_loss = 0.0
    total_tokens = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            n_tokens = int((batch["labels"] != -100).sum().item())
            total_loss += float(out.loss.item()) * n_tokens
            total_tokens += n_tokens

    if total_tokens == 0:
        return float("nan")
    return math.exp(total_loss / total_tokens)


def _token_f1(prediction: str, reference: str) -> float:
    """Unigram token F1 — same implementation as casf_dataset_api.evaluator._token_f1
    and run_eval.py so scores are identical."""
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


def _get_ground_truth(probe: Any) -> str:
    """Return the ground-truth answer from a probe, supporting all field names
    used in the codebase:
      - probe.ground_truth   (TemporalWikiDataset probes, run_eval.py)
      - probe.expected_answer
      - probe.answer
    """
    for attr in ("ground_truth", "expected_answer", "answer"):
        val = getattr(probe, attr, None)
        if val is not None:
            return str(val)
    return ""


def _generate_answer(
    model: Any,
    tokenizer: Any,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 8,
) -> str:
    """Greedy-decode a short answer for a probe prompt.

    max_new_tokens=8 matches run_eval.py so scores are directly comparable
    with the team's existing evaluation script.
    """
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    enc = {k: v.to(device) for k, v in enc.items()}
    prompt_len = enc["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=getattr(tokenizer, "pad_token_id", None) or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = out[0][prompt_len:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def _score_probe(answer: str, gt: str) -> dict[str, float]:
    """Return exact, contains, and f1 scores for a single probe.

    Matches all three columns from run_eval.py so per-period results
    are directly comparable without re-running that script.
    """
    gt_lower = gt.lower()
    ans_lower = answer.lower()
    return {
        "exact": float(ans_lower == gt_lower),
        "contains": float(gt_lower in ans_lower),
        "f1": _token_f1(answer, gt),
    }


# ---------------------------------------------------------------------------
# Public metric functions
# ---------------------------------------------------------------------------

def compute_probe_scores(
    model: Any,
    tokenizer: Any,
    probes: list[Any],
    device: torch.device,
    max_new_tokens: int = 8,
) -> dict[str, float]:
    """Return mean exact, contains, and f1 over a probe list.

    Mirrors all three metric columns from run_eval.py exactly.
    """
    if not probes:
        return {"exact": float("nan"), "contains": float("nan"), "f1": float("nan")}

    total_exact = total_contains = total_f1 = 0.0
    for probe in probes:
        answer = _generate_answer(model, tokenizer, probe.prompt, device, max_new_tokens)
        gt = _get_ground_truth(probe)
        s = _score_probe(answer, gt)
        total_exact += s["exact"]
        total_contains += s["contains"]
        total_f1 += s["f1"]
    n = len(probes)
    return {"exact": total_exact / n, "contains": total_contains / n, "f1": total_f1 / n}


def compute_plasticity(
    model: Any,
    tokenizer: Any,
    probes: list[Any],
    device: torch.device,
    max_new_tokens: int = 8,
) -> float:
    """Mean token-F1 on *changed* probes against the new (post-update) answer.

    Measures how well the model has absorbed updated facts this period.
    Uses the same ground_truth field resolution and token_f1 as run_eval.py.
    """
    scores = compute_probe_scores(model, tokenizer, probes, device, max_new_tokens)
    return scores["f1"]


def compute_stability(
    model: Any,
    tokenizer: Any,
    probes: list[Any],
    device: torch.device,
    max_new_tokens: int = 8,
) -> float:
    """Mean token-F1 on *unchanged* probes.

    Measures resistance to catastrophic forgetting: high stability means
    the model still answers pre-existing facts correctly after new training.
    """
    scores = compute_probe_scores(model, tokenizer, probes, device, max_new_tokens)
    return scores["f1"]


def compute_contradiction_acc(
    model: Any,
    tokenizer: Any,
    probes: list[Any],
    registry: MemoryRegistry,
    device: torch.device,
    max_new_tokens: int = 8,
) -> float:
    """For CASM: mean token-F1 on contradiction probes against the *new* answer.

    A high score means the model branched memory correctly and is serving the
    latest version of each contradicted fact rather than the stale old one.
    Uses the same ground_truth field resolution as run_eval.py.
    """
    contradiction_probes = [p for p in probes if getattr(p, "is_contradiction", False)]
    if not contradiction_probes:
        return float("nan")

    scores = compute_probe_scores(
        model, tokenizer, contradiction_probes, device, max_new_tokens
    )
    return scores["f1"]


def compute_routing_acc(
    model: Any,
    probes: list[Any],
    registry: MemoryRegistry,
    device: torch.device,
) -> float:
    """For CASM: fraction of probes routed to the slot that owns the most
    recent version of the relevant fact.

    Requires the CASMModelWrapper to expose ``model.route(input_ids) -> list[int]``.
    Returns NaN for non-CASM models or when probes lack valid_from metadata.
    """
    try:
        from training.casm_model import CASMModelWrapper
    except ImportError:
        return float("nan")

    if not isinstance(model, CASMModelWrapper):
        return float("nan")
    if not probes:
        return float("nan")

    # Build slot_id → period map from the registry for ground truth
    slot_period: dict[int, str] = {}
    for slot in registry._slots:
        slot_period[slot.slot_id] = getattr(slot, "valid_from", None)

    hits = 0
    total = 0
    model.eval()
    for probe in probes:
        expected_period = getattr(probe, "valid_from", None)
        if expected_period is None:
            continue
        try:
            enc = model.tokenizer(probe.prompt, return_tensors="pt",
                                  truncation=True, max_length=128)
        except Exception:
            continue
        enc = {k: v.to(device) for k, v in enc.items()}
        try:
            with torch.no_grad():
                top_slot_ids = model.route(enc["input_ids"])
        except Exception:
            continue
        if hasattr(top_slot_ids, "tolist"):
            top_slot_ids = top_slot_ids.tolist()
        chosen_period = slot_period.get(top_slot_ids[0] if top_slot_ids else -1)
        hits += int(chosen_period == expected_period)
        total += 1

    return hits / total if total > 0 else float("nan")


def compute_perplexity_on_probes(
    model: Any,
    tokenizer: Any,
    probes: list[Any],
    device: torch.device,
) -> float:
    """Perplexity over probe context/prompt texts."""
    texts = []
    for p in probes:
        if hasattr(p, "context") and p.context:
            texts.append(p.context)
        elif hasattr(p, "prompt"):
            texts.append(p.prompt)
    return _compute_perplexity(model, tokenizer, texts, device)


# ---------------------------------------------------------------------------
# Per-period evaluation entry point
# ---------------------------------------------------------------------------

def evaluate_period(
    *,
    model: Any,
    tokenizer: Any,
    dataset: TemporalDataset,
    cfg: Any,
    unit: str,
    registry: MemoryRegistry,
    eval_batch_size: int = 4,
    max_new_tokens: int = 8,
) -> dict[str, Any]:
    """Run all applicable metrics for one training period.

    Returns a flat dict whose keys are prefixed by method name for SMF/CASM
    (e.g. ``casm/plasticity``) and un-prefixed for other methods.
    Unknown/inapplicable metrics are omitted rather than set to NaN.

    Scoring (exact, contains, f1) matches run_eval.py exactly so results
    can be compared directly with the team's existing evaluation script.
    """
    start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    method = getattr(cfg, "method", "full_ft")
    prefix = f"{method}/" if method in ("smf", "casm") else ""

    # Collect probes — mirror run_eval.py: load both splits
    try:
        changed_probes = dataset.get_probes("changed")
    except TypeError:
        changed_probes = dataset.get_probes()
    try:
        unchanged_probes = dataset.get_probes("unchanged")
    except (TypeError, ValueError):
        unchanged_probes = []

    all_probes = changed_probes + unchanged_probes

    result: dict[str, Any] = {
        "unit": unit,
        "method": method,
        "n_changed_probes": len(changed_probes),
        "n_unchanged_probes": len(unchanged_probes),
    }

    # --- Changed probes: exact + contains + f1 (matches run_eval.py columns) ---
    if changed_probes:
        changed_scores = compute_probe_scores(
            model, tokenizer, changed_probes, device, max_new_tokens
        )
        result[f"{prefix}plasticity"] = changed_scores["f1"]
        result[f"{prefix}changed_exact"] = changed_scores["exact"]
        result[f"{prefix}changed_contains"] = changed_scores["contains"]
        result[f"{prefix}changed_f1"] = changed_scores["f1"]
    else:
        result[f"{prefix}plasticity"] = float("nan")

    # --- Unchanged probes: exact + contains + f1 ---
    if unchanged_probes:
        unchanged_scores = compute_probe_scores(
            model, tokenizer, unchanged_probes, device, max_new_tokens
        )
        result[f"{prefix}stability"] = unchanged_scores["f1"]
        result[f"{prefix}unchanged_exact"] = unchanged_scores["exact"]
        result[f"{prefix}unchanged_contains"] = unchanged_scores["contains"]
        result[f"{prefix}unchanged_f1"] = unchanged_scores["f1"]
    else:
        result[f"{prefix}stability"] = float("nan")

    # --- Perplexity on changed probes ---
    ppl = compute_perplexity_on_probes(model, tokenizer, changed_probes, device)
    result[f"{prefix}perplexity_changed"] = ppl

    # --- CASM-specific metrics ---
    if method == "casm":
        result["casm/contradiction_acc"] = compute_contradiction_acc(
            model, tokenizer, all_probes, registry, device, max_new_tokens
        )
        result["casm/routing_acc"] = compute_routing_acc(model, all_probes, registry, device)

    # --- SMF-specific metrics ---
    if method == "smf":
        try:
            from training.smf_model import SMFModelWrapper
            if isinstance(model, SMFModelWrapper):
                total_params = 0
                active_params = 0
                for p in model.smf_parameters():
                    total_params += p.numel()
                    active_params += int((p.abs() > 1e-6).sum().item())
                result["smf/active_params"] = active_params
                result["smf/total_params"] = total_params
                result["smf/sparsity"] = (
                    1.0 - active_params / total_params if total_params > 0 else float("nan")
                )
        except ImportError:
            pass

    result["eval_duration_sec"] = time.time() - start
    return result
