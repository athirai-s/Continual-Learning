"""Evaluate full_ft, LoRA, SMF, and CASM checkpoints on TemporalWiki.

Loads the final checkpoint for each method, evaluates on changed and
unchanged probes from all 4 periods, and prints a comparison table.
"""

import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from casf_dataset_api import TemporalWikiDataset
from casf_dataset_api.evaluator import TemporalEvaluator, _token_f1
from training.train_config import TrainConfig

PERIODS = ["aug_sep", "sep_oct", "oct_nov", "nov_dec"]
MODEL_PATH = "/scratch1/ashanmug/models/Llama-3.2-3B-Instruct"

# Checkpoint paths — update these as experiments complete
CHECKPOINTS = {
    # "full_ft": "/scratch1/ashanmug/checkpoints/full_ft_3b_experiment/checkpoints/ckpt-000004",
    # "lora": "/scratch1/ashanmug/checkpoints/lora_3b_experiment/checkpoints/ckpt-000004",
    "smf": "/scratch1/ashanmug/checkpoints/smf_3b_experiment/checkpoints/ckpt-000004",
    # "casm": "/scratch1/ashanmug/checkpoints/casm_3b_experiment/checkpoints/ckpt-000004",
}


def _load_train_config(checkpoint_path):
    config_path = os.path.join(checkpoint_path, "train_config.json")
    if not os.path.exists(config_path):
        return None
    with open(config_path) as f:
        return TrainConfig.from_dict(json.load(f))


def _resolve_eval_dtype(cfg):
    if cfg is None:
        return torch.bfloat16
    if cfg.precision == "bfloat16":
        return torch.bfloat16
    if cfg.precision == "float16":
        return torch.float16
    raise ValueError("run_eval.py does not support precision='int8'")


def _print_period_scores(gen_model):
    for period in PERIODS:
        print(f"\n--- Period: {period} ---")
        dataset = TemporalWikiDataset(period=period)

        for split in ["changed", "unchanged"]:
            dataset.load(split)
            probes = dataset.get_probes(split)
            if not probes:
                print(f"  {split}: no probes")
                continue

            n_correct = 0
            total_f1 = 0.0
            contains = 0
            for probe in probes:
                output = gen_model.generate(probe.prompt)
                gt = probe.ground_truth.lower()
                out_lower = output.lower()
                if gt in out_lower:
                    contains += 1
                if out_lower == gt:
                    n_correct += 1
                total_f1 += _token_f1(output, probe.ground_truth)

            n = len(probes)
            print(
                f"  {split:12s}  n={n:4d}  exact={n_correct/n:.3f}  "
                f"contains={contains/n:.3f}  f1={total_f1/n:.5f}"
            )


class GenerationModel:
    """Wraps a model for generation, compatible with TemporalEvaluator."""
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def generate(self, prompt: str) -> str:
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
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        generated = output[0][prompt_len:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)


def evaluate_method(name, checkpoint_path):
    print(f"\n{'='*60}")
    print(f"  Evaluating: {name}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"{'='*60}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_cfg = _load_train_config(checkpoint_path)
    eval_dtype = _resolve_eval_dtype(train_cfg)

    # Load model from checkpoint
    lora_adapter_path = os.path.join(checkpoint_path, "adapter_config.json")
    if os.path.exists(lora_adapter_path):
        # LoRA checkpoint: adapter only, load base model from original path
        from peft import PeftModel

        if train_cfg is None:
            raise FileNotFoundError(
                f"LoRA evaluation requires train_config.json in {checkpoint_path}"
            )

        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        base_model = AutoModelForCausalLM.from_pretrained(
            train_cfg.model_name,
            torch_dtype=eval_dtype,
        )
        model = PeftModel.from_pretrained(base_model, checkpoint_path)
        model.to(device)
        gen_model = GenerationModel(model, tokenizer, device)
        _print_period_scores(gen_model)
        return

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    model = AutoModelForCausalLM.from_pretrained(checkpoint_path, torch_dtype=eval_dtype)

    # Load SMF memory if present
    smf_memory_path = os.path.join(checkpoint_path, "smf_memory.pt")
    if os.path.exists(smf_memory_path):
        from training.smf_model import SMFModelWrapper
        if train_cfg is None:
            raise FileNotFoundError(
                f"SMF evaluation requires train_config.json in {checkpoint_path}"
            )
        cfg = train_cfg
        wrapper = SMFModelWrapper(model, cfg)
        SMFModelWrapper.load_memory_into(wrapper, checkpoint_path)
        wrapper.to(device)
        gen_model = GenerationModel(wrapper, tokenizer, device)
    # Load CASM state if present
    elif os.path.exists(os.path.join(checkpoint_path, "casm_memory.pt")):
        from training.casm_model import CASMModelWrapper
        if train_cfg is None:
            raise FileNotFoundError(
                f"CASM evaluation requires train_config.json in {checkpoint_path}"
            )
        cfg = train_cfg
        wrapper = CASMModelWrapper(model, cfg)
        CASMModelWrapper.load_memory_into(wrapper, checkpoint_path)
        wrapper.to(device)
        gen_model = GenerationModel(wrapper, tokenizer, device)
    else:
        model.to(device)
        gen_model = GenerationModel(model, tokenizer, device)
    _print_period_scores(gen_model)


def evaluate_baseline():
    """Evaluate the base model (no fine-tuning) as reference."""
    print(f"\n{'='*60}")
    print(f"  Evaluating: baseline (no fine-tuning)")
    print(f"  Model: {MODEL_PATH}")
    print(f"{'='*60}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16)
    model.to(device)
    gen_model = GenerationModel(model, tokenizer, device)

    for period in PERIODS:
        print(f"\n--- Period: {period} ---")
        dataset = TemporalWikiDataset(period=period)

        for split in ["changed", "unchanged"]:
            dataset.load(split)
            probes = dataset.get_probes(split)
            if not probes:
                print(f"  {split}: no probes")
                continue

            n_correct = 0
            total_f1 = 0.0
            contains = 0
            for probe in probes:
                output = gen_model.generate(probe.prompt)
                gt = probe.ground_truth.lower()
                out_lower = output.lower()
                if gt in out_lower:
                    contains += 1
                if out_lower == gt:
                    n_correct += 1
                total_f1 += _token_f1(output, probe.ground_truth)

            n = len(probes)
            print(f"  {split:12s}  n={n:4d}  exact={n_correct/n:.3f}  contains={contains/n:.3f}  f1={total_f1/n:.5f}")


if __name__ == "__main__":
    # Baseline first
    evaluate_baseline()

    # Then each method's checkpoint
    for name, path in CHECKPOINTS.items():
        if os.path.exists(path):
            evaluate_method(name, path)
        else:
            print(f"\nSkipping {name}: checkpoint not found at {path}")

    print("\n\nDone!")
