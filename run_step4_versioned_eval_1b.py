"""Step 4: Versioned fact retention eval — tests whether each model retains
the P1 version of a changed fact after training on P2-P4.

For changed probes (facts that changed across periods):
  - P1 answer = the OLD value (what was true in aug_sep)
  - P4 answer = the NEW value (what is true in nov_dec)

A model that truly retains P1 knowledge should score higher on P1 answers.
A model that forgets (full_ft) should score higher on P4 answers.
CASM with versioned slots should retain P1 better than LoRA/SMF.

Metrics per model:
  p1_score       — mean log-prob of P1 (old) answer on changed probes
  p4_score       — mean log-prob of P4 (new) answer on changed probes
  retention_gap  — p1_score - p4_score (positive = remembers old fact)
  version_pref   — fraction of changed probes where model prefers P1 answer
"""

import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from casf_dataset_api import TemporalWikiDataset

CHECKPOINTS = {
    "pretrain_p1": "/scratch1/ramyakri/checkpoints/pretrain_period1_1b/checkpoints/ckpt-000001",
    "full_ft":     "/scratch1/ramyakri/checkpoints/step2_fullft_1b/checkpoints/ckpt-000004",
    "lora":        "/scratch1/ramyakri/checkpoints/step2_lora_1b/checkpoints/ckpt-000004",
    "smf":         "/scratch1/ramyakri/checkpoints/step2_smf_1b/checkpoints/ckpt-000004",
    "casm":        "/scratch1/ramyakri/checkpoints/step2_casm_1b_v4/checkpoints/ckpt-000004",
}


class EvalModel:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def score_probe(self, prompt: str, answer: str) -> float:
        """Return mean log-prob of answer tokens given prompt. Higher = better."""
        prompt_ids = self.tokenizer(
            prompt, return_tensors="pt", add_special_tokens=True
        )["input_ids"].to(self.device)
        prompt_len = prompt_ids.shape[1]

        full_text = prompt + " " + answer
        full_ids = self.tokenizer(
            full_text, return_tensors="pt", add_special_tokens=True,
            truncation=True, max_length=256,
        )["input_ids"].to(self.device)

        if full_ids.shape[1] <= prompt_len:
            return -999.0

        with torch.no_grad():
            logits = self.model(input_ids=full_ids).logits

        answer_ids = full_ids[0, prompt_len:]
        pred_logits = logits[0, prompt_len - 1: -1]
        log_probs = torch.nn.functional.log_softmax(pred_logits, dim=-1)
        token_lp = log_probs.gather(1, answer_ids.unsqueeze(1)).squeeze(1)
        return token_lp.mean().item()


def load_model(name, checkpoint_path, device):
    config_path = os.path.join(checkpoint_path, "train_config.json")
    cfg_dict = {}
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg_dict = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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

    smf_path = os.path.join(checkpoint_path, "smf_memory.pt")
    if os.path.exists(smf_path):
        from training.smf_model import SMFModelWrapper
        from training.train_config import TrainConfig
        cfg = TrainConfig.from_dict(cfg_dict)
        wrapper = SMFModelWrapper(model, cfg)
        SMFModelWrapper.load_memory_into(wrapper, checkpoint_path)
        wrapper.to(device)
        return EvalModel(wrapper, tokenizer, device)

    casm_path = os.path.join(checkpoint_path, "casm_memory.pt")
    if os.path.exists(casm_path):
        from training.casm_model import CASMModelWrapper
        from training.train_config import TrainConfig
        cfg = TrainConfig.from_dict(cfg_dict)
        wrapper = CASMModelWrapper(model, cfg)
        CASMModelWrapper.load_memory_into(wrapper, checkpoint_path)
        wrapper.to(device)
        return EvalModel(wrapper, tokenizer, device)

    model.to(device)
    return EvalModel(model, tokenizer, device)


def evaluate_versioned(eval_model):
    """Compare model's preference for P1 (old) vs P4 (new) answers on changed probes."""
    # Load P1 probes to get old values
    p1_dataset = TemporalWikiDataset(period="aug_sep")
    p1_dataset.load("changed")
    p1_probes = p1_dataset.get_probes("changed")

    # Load P4 probes to get new values for the same facts
    p4_dataset = TemporalWikiDataset(period="nov_dec")
    p4_dataset.load("changed")
    p4_probes = p4_dataset.get_probes("changed")

    # Build lookup: (subject, relation) -> P4 probe
    p4_lookup = {}
    for p in p4_probes:
        p4_lookup[(p.subject, p.relation)] = p

    p1_scores = []
    p4_scores = []
    prefers_p1 = 0
    n_compared = 0
    DEBUG_SAMPLES = 3

    for i, p1_probe in enumerate(p1_probes):
        key = (p1_probe.subject, p1_probe.relation)
        p4_probe = p4_lookup.get(key)
        if p4_probe is None:
            continue

        # P1 answer = old value, P4 answer = new value
        p1_ans = p1_probe.ground_truth
        p4_ans = p4_probe.ground_truth

        if p1_ans == p4_ans:
            continue  # fact didn't actually change — skip

        score_p1 = eval_model.score_probe(p1_probe.prompt, p1_ans)
        score_p4 = eval_model.score_probe(p1_probe.prompt, p4_ans)

        p1_scores.append(score_p1)
        p4_scores.append(score_p4)
        if score_p1 > score_p4:
            prefers_p1 += 1
        n_compared += 1

        if i < DEBUG_SAMPLES:
            print(f"  [DBG#{i}] subject={p1_probe.subject!r}")
            print(f"           P1 answer={p1_ans!r}  logp={score_p1:.3f}")
            print(f"           P4 answer={p4_ans!r}  logp={score_p4:.3f}")
            print(f"           prefers={'P1 (OLD)' if score_p1 > score_p4 else 'P4 (NEW)'}")

    if n_compared == 0:
        return {"n": 0, "p1_score": 0.0, "p4_score": 0.0, "retention_gap": 0.0, "version_pref_p1": 0.0}

    mean_p1 = sum(p1_scores) / n_compared
    mean_p4 = sum(p4_scores) / n_compared
    return {
        "n": n_compared,
        "p1_score": mean_p1,
        "p4_score": mean_p4,
        "retention_gap": mean_p1 - mean_p4,  # positive = prefers old fact
        "version_pref_p1": prefers_p1 / n_compared,  # fraction preferring P1
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
        print(f"{'='*60}")

        eval_model = load_model(name, ckpt_path, device)
        results = evaluate_versioned(eval_model)
        all_results[name] = results

        print(f"  n={results['n']}  p1_score={results['p1_score']:.4f}  p4_score={results['p4_score']:.4f}")
        print(f"  retention_gap={results['retention_gap']:.4f}  version_pref_p1={results['version_pref_p1']:.2%}")

        del eval_model
        torch.cuda.empty_cache()

    # Print comparison table
    print(f"\n\n{'='*80}")
    print("  VERSIONED FACT RETENTION — P1 (old) vs P4 (new) answer preference")
    print("  retention_gap > 0 = model prefers old P1 answer (good retention)")
    print("  version_pref_p1   = fraction of probes where model prefers P1 answer")
    print(f"{'='*80}")
    print(f"\n{'Method':<15} {'N':>5} {'P1_score':>10} {'P4_score':>10} {'Gap':>8} {'PrefP1%':>10}")
    print("-" * 65)

    for name in CHECKPOINTS:
        if name not in all_results:
            continue
        r = all_results[name]
        print(f"{name:<15} {r['n']:>5} {r['p1_score']:>10.4f} {r['p4_score']:>10.4f} {r['retention_gap']:>8.4f} {r['version_pref_p1']:>9.1%}")

    output_path = "step4_versioned_eval_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
