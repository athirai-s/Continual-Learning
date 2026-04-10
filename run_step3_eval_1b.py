"""Step 3: Evaluate all methods on Period 1 probes to measure forgetting.

Loads each method's final checkpoint (after P2-P4 training) and tests
on Period 1 (aug_sep) changed + unchanged probes. Also evaluates the
Period 1 pretrained checkpoint as the "before forgetting" reference.

Uses Llama 3 instruct chat format for better answer extraction.
"""

import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from casf_dataset_api import TemporalWikiDataset
from casf_dataset_api.evaluator import _token_f1

# All checkpoints to evaluate — 1B model, all trained on P1→P4 (4 periods)
# pretrain_p1 = upper bound  (trained on P1 only, knows P1 perfectly)
# full_ft     = lower bound  (all weights updated across P1-P4 → should forget P1)
# lora        = frozen backbone + adapters (moderate retention)
# smf         = frozen backbone + sparse memory (better retention)
# casm        = frozen backbone + versioned slots (best retention — key claim)
#
# All step2 runs have 4 period checkpoints → final is ckpt-000004
CHECKPOINTS = {
    "pretrain_p1": "/scratch1/ramyakri/checkpoints/pretrain_period1_1b/checkpoints/ckpt-000001",
    "full_ft":     "/scratch1/ramyakri/checkpoints/step2_fullft_1b/checkpoints/ckpt-000004",
    "lora":        "/scratch1/ramyakri/checkpoints/step2_lora_1b/checkpoints/ckpt-000004",
    "smf":         "/scratch1/ramyakri/checkpoints/step2_smf_1b/checkpoints/ckpt-000004",
    "casm":        "/scratch1/ramyakri/checkpoints/step2_casm_1b/checkpoints/ckpt-000004",
}


class EvalModel:
    """Generation wrapper using plain cloze completion (no instruct format).
    The model was fine-tuned on raw passages, so direct completion works best.
    """
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def generate(self, prompt: str) -> str:
        # Use plain cloze prompt — model completes "The CEO of X is" → "Tim Cook"
        encoded = self.tokenizer(
            prompt,
            truncation=True,
            max_length=512,
            padding="do_not_pad",
            return_tensors="pt",
        )
        batch = {k: v.to(self.device) for k, v in encoded.items()}
        prompt_len = batch["input_ids"].shape[1]
        with torch.no_grad():
            output = self.model.generate(
                **batch,
                max_new_tokens=8,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        generated = output[0][prompt_len:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        # Take only the first meaningful chunk (before newline or period)
        text = text.split("\n")[0].split(".")[0].strip()
        return text


def load_model(name, checkpoint_path, device):
    """Load model from checkpoint, handling SMF/CASM/LoRA wrappers."""
    config_path = os.path.join(checkpoint_path, "train_config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg_dict = json.load(f)
        method = cfg_dict.get("method", "full_ft")
    else:
        method = "full_ft"

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # LoRA: load base + adapter
    lora_path = os.path.join(checkpoint_path, "adapter_config.json")
    if os.path.exists(lora_path):
        from peft import PeftModel
        base_model_name = cfg_dict.get("model_name", checkpoint_path)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name, torch_dtype=torch.bfloat16
        )
        model = PeftModel.from_pretrained(base_model, checkpoint_path)
        model.to(device)
        return EvalModel(model, tokenizer, device)

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path, torch_dtype=torch.bfloat16
    )

    # SMF: wrap and load memory
    smf_path = os.path.join(checkpoint_path, "smf_memory.pt")
    if os.path.exists(smf_path):
        from training.smf_model import SMFModelWrapper
        from training.train_config import TrainConfig
        cfg = TrainConfig.from_dict(cfg_dict)
        wrapper = SMFModelWrapper(model, cfg)
        SMFModelWrapper.load_memory_into(wrapper, checkpoint_path)
        wrapper.to(device)
        return EvalModel(wrapper, tokenizer, device)

    # CASM: wrap and load memory
    casm_path = os.path.join(checkpoint_path, "casm_memory.pt")
    if os.path.exists(casm_path):
        from training.casm_model import CASMModelWrapper
        from training.train_config import TrainConfig
        cfg = TrainConfig.from_dict(cfg_dict)
        wrapper = CASMModelWrapper(model, cfg)
        CASMModelWrapper.load_memory_into(wrapper, checkpoint_path)
        wrapper.to(device)
        return EvalModel(wrapper, tokenizer, device)

    # Plain model (full_ft or pretrain)
    model.to(device)
    return EvalModel(model, tokenizer, device)


def evaluate_on_period1(gen_model, split="changed"):
    """Evaluate on Period 1 (aug_sep) probes. Returns metrics dict."""
    dataset = TemporalWikiDataset(period="aug_sep")
    dataset.load(split)
    probes = dataset.get_probes(split)

    if not probes:
        return {"n": 0, "exact": 0.0, "contains": 0.0, "f1": 0.0}

    n_exact = 0
    n_contains = 0
    total_f1 = 0.0

    for probe in probes:
        output = gen_model.generate(probe.prompt)
        gt = probe.ground_truth.lower()
        out_lower = output.lower()

        if out_lower == gt:
            n_exact += 1
        if gt in out_lower:
            n_contains += 1
        total_f1 += _token_f1(output, probe.ground_truth)

    n = len(probes)
    return {
        "n": n,
        "exact": n_exact / n,
        "contains": n_contains / n,
        "f1": total_f1 / n,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    all_results = {}

    for name, ckpt_path in CHECKPOINTS.items():
        if not os.path.exists(ckpt_path):
            print(f"Skipping {name}: checkpoint not found at {ckpt_path}")
            continue

        print(f"\n{'='*60}")
        print(f"  Evaluating: {name}")
        print(f"  Checkpoint: {ckpt_path}")
        print(f"{'='*60}")

        gen_model = load_model(name, ckpt_path, device)
        results = {}

        for split in ["changed", "unchanged"]:
            print(f"\n  --- {split} probes ---")
            metrics = evaluate_on_period1(gen_model, split)
            results[split] = metrics
            print(f"  n={metrics['n']:4d}  exact={metrics['exact']:.4f}  "
                  f"contains={metrics['contains']:.4f}  f1={metrics['f1']:.5f}")

        all_results[name] = results

        # Free memory for next model
        del gen_model
        torch.cuda.empty_cache()

    # Print comparison table
    print(f"\n\n{'='*80}")
    print("  COMPARISON TABLE — Period 1 Retention After Training on P2-P4")
    print(f"{'='*80}")
    print(f"\n{'Method':<15} {'Split':<12} {'N':>5} {'Exact':>8} {'Contains':>10} {'F1':>8}")
    print("-" * 60)

    for name in CHECKPOINTS:
        if name not in all_results:
            continue
        for split in ["changed", "unchanged"]:
            m = all_results[name][split]
            print(f"{name:<15} {split:<12} {m['n']:>5} {m['exact']:>8.4f} "
                  f"{m['contains']:>10.4f} {m['f1']:>8.5f}")
        print()

    # Save results to JSON
    output_path = "step3_eval_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
