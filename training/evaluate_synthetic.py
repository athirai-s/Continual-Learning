"""Evaluation script for CASM on the synthetic dataset.

Runs per-period evaluation across all four time periods using TemporalEvaluator,
then prints a summary table.  Also optionally evaluates routing accuracy using
SimilarityRouter (zero-shot) or a trained MLPRouter (if a checkpoint path is
given).

This script is designed for offline analysis; it does not require the full
training runner.

Metrics computed
----------------
Per period:
    plasticity      — accuracy on is_changed=True probes (changed facts)
    stability       — accuracy on is_changed=False probes (stable facts)
    token_f1        — average token-level F1 across all probes
    routing_acc     — fraction of probes routed to the correct slot
                      (requires router and registry)

Usage:
    uv run python training/evaluate_synthetic.py \\
        --checkpoint /path/to/checkpoint \\
        --probes data/probes.json \\
        --passages data/passages.json

    # With routing evaluation:
    uv run python training/evaluate_synthetic.py \\
        --checkpoint /path/to/checkpoint \\
        --registry /path/to/memory_registry.json \\
        --router similarity

    # With a trained MLPRouter checkpoint:
    uv run python training/evaluate_synthetic.py \\
        --checkpoint /path/to/checkpoint \\
        --registry /path/to/memory_registry.json \\
        --router mlp --router-checkpoint /path/to/router.pt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from casf_dataset_api import (
    MemoryRegistry,
    SyntheticDataset,
    TemporalEvaluator,
)
from casf_dataset_api.casf_types import EvalResult
from training.train_config import TrainConfig

PERIODS = ["2018", "2020", "2022", "2024"]


# ---------------------------------------------------------------------------
# Model wrapper for TemporalEvaluator.score_probe()

class HFModelWrapper:
    """Thin wrapper so TemporalEvaluator.score_probe() can call model.generate()."""

    def __init__(
        self,
        model: Any,
        tokenizer: AutoTokenizer,
        device: torch.device,
        max_new_tokens: int = 20,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_new_tokens = max_new_tokens

    def generate(self, prompt: str) -> str:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        # Decode only the newly generated tokens
        new_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Method-aware checkpoint loader
#
# Mirrors run_step3_eval.load_model() and train_runner.load_real_model_and_tokenizer:
# reads train_config.json from the checkpoint, rebuilds the method wrapper
# (CASM/SMF/LoRA), and restores memory state via load_memory_into.  Without
# this, a CASM checkpoint evaluates as the raw frozen backbone.

def load_checkpoint_for_eval(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[Any, AutoTokenizer, str, Optional[TrainConfig]]:
    """Return (model_or_wrapper, tokenizer, method, cfg).

    model_or_wrapper is ready for inference: CASM/SMF are wrapped and memory
    loaded; LoRA has its adapter attached; full_ft / pretrained returns plain
    AutoModelForCausalLM.
    """
    checkpoint_path = Path(checkpoint_path)
    config_path = checkpoint_path / "train_config.json"
    cfg: Optional[TrainConfig] = None
    cfg_dict: dict = {}
    if config_path.exists():
        with open(config_path) as f:
            cfg_dict = json.load(f)
        cfg = TrainConfig.from_dict(cfg_dict)
        method = cfg.method
    else:
        method = "full_ft"

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    # LoRA: load base model then attach adapter.
    lora_path = checkpoint_path / "adapter_config.json"
    if lora_path.exists():
        from peft import PeftModel
        base_model_name = cfg_dict.get("model_name", str(checkpoint_path))
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name, torch_dtype=dtype
        )
        model = PeftModel.from_pretrained(base_model, str(checkpoint_path))
        model.to(device).eval()
        return model, tokenizer, "lora", cfg

    base_model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path, torch_dtype=dtype
    )

    # SMF: wrap and restore memory.
    smf_path = checkpoint_path / "smf_memory.pt"
    if smf_path.exists():
        from training.smf_model import SMFModelWrapper
        if cfg is None:
            raise RuntimeError(
                f"SMF checkpoint at {checkpoint_path} missing train_config.json; "
                "cannot reconstruct wrapper."
            )
        wrapper = SMFModelWrapper(base_model, cfg)
        SMFModelWrapper.load_memory_into(wrapper, str(checkpoint_path))
        wrapper.to(device).eval()
        return wrapper, tokenizer, "smf", cfg

    # CASM: wrap and restore slot bank + router state.
    casm_path = checkpoint_path / "casm_memory.pt"
    if casm_path.exists():
        from training.casm_model import CASMModelWrapper
        if cfg is None:
            raise RuntimeError(
                f"CASM checkpoint at {checkpoint_path} missing train_config.json; "
                "cannot reconstruct wrapper."
            )
        wrapper = CASMModelWrapper(base_model, cfg)
        CASMModelWrapper.load_memory_into(wrapper, str(checkpoint_path))
        wrapper.to(device).eval()
        return wrapper, tokenizer, "casm", cfg

    # Plain model (full_ft or pretrained).
    base_model.to(device).eval()
    return base_model, tokenizer, method, cfg


# ---------------------------------------------------------------------------
# Routing accuracy helpers

def compute_routing_acc_similarity(
    router,
    probes,
    slot_map: dict,
) -> float:
    """Measure routing accuracy using SimilarityRouter."""
    correct = 0
    total = 0
    for probe in probes:
        key = (probe.subject, probe.relation, probe.timestamp)
        true_slot = slot_map.get(key)
        if true_slot is None:
            continue
        predicted = router.route(probe.prompt, period=probe.timestamp)
        if predicted == true_slot:
            correct += 1
        total += 1
    return correct / total if total > 0 else 0.0


def compute_routing_acc_mlp(
    router,
    probes,
    slot_map: dict,
) -> float:
    """Measure routing accuracy using MLPRouter."""
    from training.router import PERIOD_MAP
    correct = 0
    total = 0
    for probe in probes:
        key = (probe.subject, probe.relation, probe.timestamp)
        true_slot = slot_map.get(key)
        if true_slot is None:
            continue
        period_id = PERIOD_MAP.get(probe.timestamp)
        predicted = router.route(probe.prompt, period_id=period_id)
        if predicted == true_slot:
            correct += 1
        total += 1
    return correct / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Per-period evaluation

def evaluate_period(
    period: str,
    model_wrapper: HFModelWrapper,
    evaluator: TemporalEvaluator,
    *,
    probes_path: Path,
    passages_path: Path,
    router=None,
    slot_map: Optional[dict] = None,
    use_augmented: bool = False,
) -> dict:
    # For CASM with period-deterministic slot masking, the underlying wrapper
    # decides which slot subset to route to based on _current_period.  Without
    # this the eval always uses the last training period's slots, which silently
    # mis-routes earlier periods' probes.  See run_period_evaluation().
    underlying = model_wrapper.model
    if hasattr(underlying, "_current_period"):
        underlying._current_period = period

    ds = SyntheticDataset(
        period,
        probes_path=probes_path,
        passages_path=passages_path,
        use_augmented=use_augmented,
    )

    changed_result: Optional[EvalResult] = None
    unchanged_result: Optional[EvalResult] = None

    try:
        ds.load("changed")
        changed_probes = ds.get_probes("changed")
        if changed_probes:
            changed_result = evaluator.evaluate(model_wrapper, ds, split="changed")
    except (FileNotFoundError, ValueError):
        changed_probes = []

    try:
        ds.load("unchanged")
        unchanged_probes = ds.get_probes("unchanged")
        if unchanged_probes:
            unchanged_result = evaluator.evaluate(model_wrapper, ds, split="unchanged")
    except (FileNotFoundError, ValueError):
        unchanged_probes = []

    plasticity = changed_result.plasticity if changed_result else 0.0
    stability = unchanged_result.stability if unchanged_result else 0.0
    token_f1 = (
        (changed_result.token_f1 if changed_result else 0.0)
        if not unchanged_result
        else (
            (
                (changed_result.token_f1 if changed_result else 0.0)
                + unchanged_result.token_f1
            )
            / 2
        )
    )

    result = {
        "period": period,
        "plasticity": plasticity,
        "stability": stability,
        "token_f1": token_f1,
        "n_changed": len(changed_probes),
        "n_unchanged": len(unchanged_probes),
        "routing_acc": None,
    }

    if router is not None and slot_map is not None:
        all_probes = list(changed_probes) + list(unchanged_probes)
        if hasattr(router, "_slot_embeddings"):
            result["routing_acc"] = compute_routing_acc_similarity(
                router, all_probes, slot_map
            )
        else:
            result["routing_acc"] = compute_routing_acc_mlp(
                router, all_probes, slot_map
            )

    return result


# ---------------------------------------------------------------------------
# Summary table

def print_table(results: list[dict]) -> None:
    header = f"{'Period':>8}  {'Plast':>7}  {'Stab':>7}  {'F1':>7}  {'RoutAcc':>8}  {'#Changed':>9}  {'#Unchanged':>10}"
    print()
    print(header)
    print("-" * len(header))
    for r in results:
        rt = f"{r['routing_acc']:.3f}" if r["routing_acc"] is not None else "   N/A"
        print(
            f"{r['period']:>8}  "
            f"{r['plasticity']:>7.3f}  "
            f"{r['stability']:>7.3f}  "
            f"{r['token_f1']:>7.3f}  "
            f"{rt:>8}  "
            f"{r['n_changed']:>9}  "
            f"{r['n_unchanged']:>10}"
        )
    print()


# ---------------------------------------------------------------------------
# Entry point

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on synthetic data")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to a transformers checkpoint directory",
    )
    parser.add_argument(
        "--probes",
        type=Path,
        default=Path("data/probes.json"),
    )
    parser.add_argument(
        "--passages",
        type=Path,
        default=Path("data/passages.json"),
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=None,
        help="Path to memory_registry.json (required for routing evaluation)",
    )
    parser.add_argument(
        "--router",
        choices=["none", "similarity", "mlp"],
        default="none",
    )
    parser.add_argument(
        "--router-checkpoint",
        type=Path,
        default=None,
        help="Path to a saved MLPRouter state_dict (required for --router mlp)",
    )
    parser.add_argument(
        "--augmented",
        action="store_true",
        help="Use augmented passages instead of thin templates",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Save results JSON to this path",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading checkpoint: {args.checkpoint}")

    model, tokenizer, method, cfg = load_checkpoint_for_eval(args.checkpoint, device)
    print(f"Method: {method}")
    if cfg is not None and method == "casm":
        print(
            f"CASM config: num_slots={cfg.casm_num_slots} "
            f"top_k={cfg.casm_top_k} "
            f"slots_per_period={cfg.casm_slots_per_period}"
        )

    wrapper = HFModelWrapper(model, tokenizer, device)
    evaluator = TemporalEvaluator()

    # Optional routing setup
    router = None
    slot_map: Optional[dict] = None

    if args.router != "none" and args.registry is not None:
        registry = MemoryRegistry()
        registry.load(str(args.registry))
        from training.train_router import build_slot_map
        slot_map = build_slot_map(registry)
        print(f"Registry: {len(registry)} slots, slot_map: {len(slot_map)} entries")

        if args.router == "similarity":
            from training.router_baseline import SimilarityRouter
            router = SimilarityRouter()
            for slot in registry._slots:
                router.register_slot(slot.slot_id, {
                    "entity": slot.subject,
                    "relation": slot.relation,
                    "period": slot.valid_from,
                    "value": slot.value,
                })
            print(f"SimilarityRouter: {len(router.registered_slots())} slots registered")

        elif args.router == "mlp":
            from training.router import MLPRouter
            if args.router_checkpoint is None:
                parser.error("--router-checkpoint required when --router mlp")
            num_slots = max(slot_map.values()) + 1
            router = MLPRouter(num_slots=num_slots)
            state = torch.load(args.router_checkpoint, map_location="cpu")
            router.load_state_dict(state)
            router.to(device).eval()
            print(f"MLPRouter loaded: {num_slots} slots")

    # Run evaluation
    all_results: list[dict] = []
    for period in PERIODS:
        print(f"\nEvaluating period {period} ...")
        result = evaluate_period(
            period,
            wrapper,
            evaluator,
            probes_path=args.probes,
            passages_path=args.passages,
            router=router,
            slot_map=slot_map,
            use_augmented=args.augmented,
        )
        all_results.append(result)
        print(
            f"  plasticity={result['plasticity']:.3f}"
            f"  stability={result['stability']:.3f}"
            f"  token_f1={result['token_f1']:.3f}"
            + (f"  routing_acc={result['routing_acc']:.3f}" if result["routing_acc"] is not None else "")
        )

    print_table(all_results)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(all_results, indent=2))
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
