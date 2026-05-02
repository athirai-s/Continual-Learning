"""Sanity diagnostic for a trained CASM checkpoint.

Run this after training to confirm:
    1. Memory injection is live  (wrapper.generate != backbone.generate)
    2. Gate logits moved off init (mean != 0 across slots)
    3. Router is content-dependent (query_proj.weight.norm > 0 per slot)
    4. Dead-slot clustering under period-deterministic masking
       (does a period block have disproportionately many dead slots?)

Takes one CLI arg: --checkpoint <path to ckpt dir>

Runs in under 2 min on one A100.  Uses load_checkpoint_for_eval() from
training.evaluate_synthetic so the loading path matches what the fixed
offline eval does — no drift between diagnostic and eval.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from training.evaluate_synthetic import load_checkpoint_for_eval

SYNTH_PROBES = Path(__file__).resolve().parent / "data" / "probes.json"
DEAD_NORM_THRESHOLD = 1e-3


def pick_test_prompts(periods: list[str], n_per_period: int = 1) -> list[tuple[str, str]]:
    """Return [(period, prompt), ...] from real changed probes if available."""
    if not SYNTH_PROBES.exists():
        return [
            (p, f"[{p}] The harbour master of Veldris Corp is")
            for p in periods
        ]
    with open(SYNTH_PROBES) as f:
        data = json.load(f)
    out: list[tuple[str, str]] = []
    for period in periods:
        probes = data.get(period, {}).get("changed", [])
        for probe in probes[:n_per_period]:
            prompt = probe.get("prompt") or f"[{period}] {probe.get('subject','')}"
            out.append((period, prompt))
    return out


def print_slot_health(model) -> dict:
    """Print per-slot gate/query-proj stats, return summary dict."""
    rows: list[dict] = []
    for key, block in model.slot_bank.items():
        gate_mean = float(block.gate_logits.mean().item())
        gate_norm = float(block.gate_logits.norm().item())
        if block.query_proj is not None:
            w_norm = float(block.query_proj.weight.norm().item())
        else:
            w_norm = float("nan")
        rows.append(
            {
                "slot_id": int(key),
                "gate_mean": gate_mean,
                "gate_norm": gate_norm,
                "query_proj_norm": w_norm,
                "dead": w_norm < DEAD_NORM_THRESHOLD,
            }
        )
    rows.sort(key=lambda r: r["slot_id"])

    print(f"\n{'SlotID':>6}  {'gate_mean':>10}  {'gate_norm':>9}  {'q_proj_norm':>12}  status")
    print("-" * 60)
    for r in rows:
        status = "DEAD" if r["dead"] else "ok"
        print(
            f"{r['slot_id']:>6}  {r['gate_mean']:>10.4f}  {r['gate_norm']:>9.3f}  "
            f"{r['query_proj_norm']:>12.4f}  {status}"
        )

    dead_ids = [r["slot_id"] for r in rows if r["dead"]]
    print(f"\nDead slots: {len(dead_ids)}/{len(rows)}  ids={dead_ids}")

    return {"rows": rows, "dead_ids": dead_ids}


def print_period_clustering(model, dead_ids: list[int]) -> None:
    """Show which period blocks the dead slots fall into."""
    period_map = getattr(model, "_period_slot_map", None)
    if not period_map:
        print("\n(no _period_slot_map — period-deterministic masking off)")
        return
    print("\nDead slots by period block:")
    per_period_total = {p: len(ids) for p, ids in period_map.items()}
    per_period_dead: Counter = Counter()
    for p, ids in period_map.items():
        for sid in dead_ids:
            if sid in ids:
                per_period_dead[p] += 1
    for p, total in per_period_total.items():
        dead = per_period_dead.get(p, 0)
        frac = dead / total if total else 0.0
        flag = "  <-- CLUSTERED" if dead >= max(2, total // 2) else ""
        print(f"  {p:>6}: {dead}/{total} dead ({frac:.0%}){flag}")


def compare_generate(model, tokenizer, device, prompts: list[tuple[str, str]]) -> None:
    """Print wrapper.generate vs backbone.generate for each prompt."""
    print("\nWrapper vs backbone generation:")
    for period, prompt in prompts:
        if hasattr(model, "_current_period"):
            model._current_period = period
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out_w = model.generate(
                **enc, max_new_tokens=8,
                do_sample=False, pad_token_id=tokenizer.eos_token_id,
            )
            out_b = model.backbone.generate(
                **enc, max_new_tokens=8,
                do_sample=False, pad_token_id=tokenizer.eos_token_id,
            )
        tail_w = tokenizer.decode(out_w[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        tail_b = tokenizer.decode(out_b[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        diff = "DIFFER" if tail_w != tail_b else "SAME (!)"
        print(f"  [{period}] {prompt}")
        print(f"    wrapper : {tail_w!r}")
        print(f"    backbone: {tail_b!r}  ->  {diff}")


def main() -> None:
    ap = argparse.ArgumentParser(description="CASM checkpoint sanity diagnostic")
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--periods", nargs="+", default=["2018", "2020", "2022", "2024"])
    ap.add_argument("--output", type=Path, default=None, help="Save JSON summary")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")

    model, tokenizer, method, cfg = load_checkpoint_for_eval(args.checkpoint, device)
    print(f"Method: {method}")
    if method != "casm":
        print("Not a CASM checkpoint — nothing to diagnose.")
        return
    if cfg is not None:
        print(
            f"CASM config: num_slots={cfg.casm_num_slots}  "
            f"top_k={cfg.casm_top_k}  "
            f"slots_per_period={cfg.casm_slots_per_period}  "
            f"memory_size={cfg.casm_memory_size}  "
            f"injection_layers={cfg.casm_num_injection_layers}"
        )

    # 1. Per-slot health
    slot_summary = print_slot_health(model)

    # 2. Dead-slot clustering per period block
    print_period_clustering(model, slot_summary["dead_ids"])

    # 3. Wrapper vs backbone divergence
    prompts = pick_test_prompts(args.periods, n_per_period=1)
    compare_generate(model, tokenizer, device, prompts)

    # 4. Summary line
    dead = len(slot_summary["dead_ids"])
    total = len(slot_summary["rows"])
    print("\n=== SUMMARY ===")
    print(f"Injection: {'yes' if total > 0 else 'no slots'}")
    print(f"Dead slots: {dead}/{total}")
    print("See DIFFER markers above for per-period wrapper-vs-backbone divergence.")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "checkpoint": str(args.checkpoint),
            "method": method,
            "num_slots": total,
            "dead_slot_ids": slot_summary["dead_ids"],
            "slot_rows": slot_summary["rows"],
        }
        with open(args.output, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
