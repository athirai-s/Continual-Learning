"""Per-layer activation capture for off-machine visualization.

Registers forward hooks on every transformer layer, runs a forward pass on
a small fixed set of diagnostic probes, and saves:

    periods/<period>/activations.pt
        layer_norms  — (n_probes, n_layers, max_seq_len) float32
        seq_lens     — list[int] actual length of each probe (before padding)
        token_ids    — list[list[int]]

    periods/<period>/activation_metadata.json
        probe text, ground truth, split, tokens, model/method info

The .pt file is designed to be copied to a separate analysis machine where
no model or GPU is needed to produce plots.

Usage
-----
Enabled via TrainConfig.capture_activations = True.
Called from train_runner after run_period_evaluation.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from casf_dataset_api.casf_types import Probe
from artifacts.run_artifacts import period_root
from .train_config import TrainConfig


_MAX_PROBES_PER_SPLIT = 10
_MAX_SEQ_LEN = 64


# ---------------------------------------------------------------------------
# Helpers


def _get_backbone(model: Any) -> Any:
    """Unwrap SMFModelWrapper / CASMModelWrapper to get the bare backbone."""
    return getattr(model, "backbone", model)


def _get_transformer_layers(model: Any) -> Any:
    from .smf_model import _get_transformer_layers as _gtl
    return _gtl(_get_backbone(model))


def _select_probes(probes: list[Probe], n: int, seed: int) -> list[Probe]:
    """Deterministic sample of up to n probes (same selection every period)."""
    rng = random.Random(seed)
    if len(probes) <= n:
        return list(probes)
    return rng.sample(probes, n)


def _decode_tokens(tokenizer: Any, token_ids: list[int]) -> list[str]:
    tokens = []
    for tid in token_ids:
        try:
            tokens.append(tokenizer.decode([tid], skip_special_tokens=False))
        except Exception:
            tokens.append(str(tid))
    return tokens


# ---------------------------------------------------------------------------
# Main entry point


def capture_period_activations(
    *,
    model: Any,
    tokenizer: Any,
    probes_by_split: dict[str, list[Probe]],
    period: str,
    run_root: str | Path,
    cfg: TrainConfig,
) -> None:
    """Capture and save per-layer activation norms for diagnostic probes.

    Args:
        model:            training model (wrapper or bare backbone)
        tokenizer:        the tokenizer
        probes_by_split:  {split_name: [Probe, ...]} for each eval split
        period:           period name, e.g. "aug_sep"
        run_root:         run root directory (same as used by the runner)
        cfg:              training config (used for seed and model metadata)
    """
    out_dir = Path(period_root(run_root, period))
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Select probes deterministically -----------------------------------
    selected: list[tuple[str, Probe]] = []
    for split_name, split_probes in probes_by_split.items():
        for probe in _select_probes(split_probes, _MAX_PROBES_PER_SPLIT, seed=cfg.seed):
            selected.append((split_name, probe))

    if not selected:
        return

    # --- Locate transformer layers and register hooks ----------------------
    try:
        layers = _get_transformer_layers(model)
    except ValueError:
        return  # unsupported architecture — skip silently

    n_layers = len(layers)
    device = next(model.parameters()).device

    # One list per forward pass; hooks append to it in layer order.
    _captured: list[torch.Tensor] = []

    def _make_hook(layer_idx: int):
        def _hook(module: Any, inputs: Any, output: Any) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            # hidden: (batch, seq_len, hidden_size) — keep on CPU immediately
            _captured.append(hidden.detach().cpu().float())
        return _hook

    handles = [layer.register_forward_hook(_make_hook(i)) for i, layer in enumerate(layers)]

    # --- Run forward passes ------------------------------------------------
    all_layer_norms: list[torch.Tensor] = []   # (n_layers, seq_len) per probe
    all_token_ids:   list[list[int]]   = []
    probe_metadata:  list[dict]        = []

    model.eval()
    try:
        with torch.no_grad():
            for split_name, probe in selected:
                _captured.clear()

                enc = tokenizer(
                    probe.prompt,
                    truncation=True,
                    max_length=_MAX_SEQ_LEN,
                    padding="do_not_pad",
                    return_tensors="pt",
                )
                input_ids     = enc["input_ids"].to(device)
                attention_mask = enc.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)

                try:
                    # Call through the full model so SMF/CASM memory injection
                    # is active — captured hidden states reflect actual inference.
                    model(input_ids=input_ids, attention_mask=attention_mask)
                except Exception:
                    continue

                if len(_captured) != n_layers:
                    continue  # hook count mismatch — architecture changed mid-run

                seq_len = _captured[0].shape[1]

                # (n_layers, seq_len) — L2 norm of hidden state per position
                layer_norms = torch.stack(
                    [_captured[i][0].norm(dim=-1) for i in range(n_layers)],
                    dim=0,
                )

                all_layer_norms.append(layer_norms)
                all_token_ids.append(input_ids[0].cpu().tolist())
                probe_metadata.append({
                    "split":           split_name,
                    "prompt":          probe.prompt[:300],
                    "ground_truth":    probe.ground_truth,
                    "subject":         probe.subject,
                    "relation":        probe.relation,
                    "is_changed":      probe.is_changed,
                    "is_contradiction": probe.is_contradiction,
                    "tokens":          _decode_tokens(tokenizer, input_ids[0].cpu().tolist()),
                    "seq_len":         seq_len,
                })
    finally:
        for h in handles:
            h.remove()

    if not all_layer_norms:
        return

    # --- Pad to uniform length and stack -----------------------------------
    max_len = max(t.shape[1] for t in all_layer_norms)
    padded = [
        F.pad(t, (0, max_len - t.shape[1])) if t.shape[1] < max_len else t
        for t in all_layer_norms
    ]
    layer_norms_tensor = torch.stack(padded, dim=0)  # (n_probes, n_layers, max_len)

    # --- Save ---------------------------------------------------------------
    torch.save(
        {
            "layer_norms": layer_norms_tensor,
            "seq_lens":    [m["seq_len"] for m in probe_metadata],
            "token_ids":   all_token_ids,
        },
        out_dir / "activations.pt",
    )

    with open(out_dir / "activation_metadata.json", "w") as f:
        json.dump(
            {
                "period":      period,
                "method":      cfg.method,
                "model_name":  cfg.model_name,
                "n_layers":    n_layers,
                "n_probes":    len(probe_metadata),
                "max_seq_len": _MAX_SEQ_LEN,
                "probes":      probe_metadata,
            },
            f,
            indent=2,
        )

    print(f"Activations saved: {len(probe_metadata)} probes × {n_layers} layers → {out_dir / 'activations.pt'}")
