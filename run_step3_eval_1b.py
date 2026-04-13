"""Step 3: Evaluate all methods on Period 1 probes to measure forgetting.

Loads each method's final checkpoint (after P2-P4 training) and tests
on Period 1 (aug_sep) changed + unchanged probes. Also evaluates the
Period 1 pretrained checkpoint as the "before forgetting" reference.

Uses plain cloze completion — model completes the prompt directly.
"""

import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from casf_dataset_api import TemporalWikiDataset

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
    "casm":        "/scratch1/ramyakri/checkpoints/step2_casm_1b_v7/checkpoints/ckpt-000004",
}


class EvalModel:
    """Scoring wrapper using log-probability of the ground-truth answer.

    The model was fine-tuned on raw passages, so generation-based eval fails
    (model continues in passage style). Instead we score how likely the model
    thinks the correct answer is given the cloze prompt — this directly
    measures retention without requiring the model to generate short answers.

    score_probe() returns the mean log-prob per answer token (higher = better).
    """
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def score_probe(self, prompt: str, answer: str) -> float:
        """Return mean log-prob of answer tokens given prompt. Higher = better retention."""
        # Tokenize prompt alone to find where answer tokens start
        prompt_ids = self.tokenizer(
            prompt, return_tensors="pt", add_special_tokens=True
        )["input_ids"].to(self.device)
        prompt_len = prompt_ids.shape[1]

        # Tokenize full text (prompt + space + answer)
        full_text = prompt + " " + answer
        full_ids = self.tokenizer(
            full_text, return_tensors="pt", add_special_tokens=True,
            truncation=True, max_length=256,
        )["input_ids"].to(self.device)

        if full_ids.shape[1] <= prompt_len:
            return -999.0  # answer tokenized to nothing

        with torch.no_grad():
            logits = self.model(input_ids=full_ids).logits  # (1, seq, vocab)

        # Answer token positions: indices [prompt_len .. end]
        # Logit at position i predicts token i+1, so shift by 1
        answer_ids = full_ids[0, prompt_len:]          # tokens to predict
        pred_logits = logits[0, prompt_len - 1 : -1]   # logits that predict them

        log_probs = torch.nn.functional.log_softmax(pred_logits, dim=-1)
        token_lp = log_probs.gather(1, answer_ids.unsqueeze(1)).squeeze(1)
        return token_lp.mean().item()


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
    """Evaluate on Period 1 (aug_sep) probes using log-prob scoring.

    Returns mean log-prob of the correct answer given the cloze prompt.
    Higher (less negative) = model retains more P1 knowledge.
    pretrain_p1 should score highest; full_ft lowest (most forgetting).
    """
    dataset = TemporalWikiDataset(period="aug_sep")
    dataset.load(split)
    probes = dataset.get_probes(split)

    if not probes:
        return {"n": 0, "mean_logprob": 0.0}

    total_lp = 0.0
    DEBUG_SAMPLES = 3

    for i, probe in enumerate(probes):
        lp = gen_model.score_probe(probe.prompt, probe.ground_truth)

        if i < DEBUG_SAMPLES:
            print(f"  [DBG#{i}] prompt={probe.prompt!r}  gt={probe.ground_truth!r}  logp={lp:.3f}")

        total_lp += lp

    n = len(probes)
    return {
        "n": n,
        "mean_logprob": total_lp / n,
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
            print(f"  n={metrics['n']:4d}  mean_logprob={metrics['mean_logprob']:.4f}")

        all_results[name] = results

        # Free memory for next model
        del gen_model
        torch.cuda.empty_cache()

    # Print comparison table
    print(f"\n\n{'='*80}")
    print("  COMPARISON TABLE — Period 1 Retention After Training on P2-P4")
    print(f"{'='*80}")
    print(f"\n{'Method':<15} {'Split':<12} {'N':>5} {'MeanLogProb':>13}")
    print("-" * 50)

    for name in CHECKPOINTS:
        if name not in all_results:
            continue
        for split in ["changed", "unchanged"]:
            m = all_results[name][split]
            print(f"{name:<15} {split:<12} {m['n']:>5} {m['mean_logprob']:>13.4f}")
        print()

    # Save results to JSON
    output_path = "step3_eval_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
