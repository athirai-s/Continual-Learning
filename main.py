# import sys
# import os
# from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch
# from casf_dataset_api import (
#     TemporalWikiDataset,
#     TSQADataset,
#     TGQADataset,
#     ContradictionDetector,
#     MemoryRegistry,
#     TemporalEvaluator,
# )
# from trainer_config import TrainConfig
# from trainer import CASFTrainer

# if __name__ == "__main__":
#     cfg = TrainConfig.make_full_ft_1b_config(run_id="debug_full_ft_1b")
#     print("Training config:")
#     print(cfg)
#     cfg.save_json("debug_config.json")
#     print("Saved config to debug_config.json\n")
#     print("Loading dataset...")
#     dataset = TemporalWikiDataset(period="aug_sep")
#     dataset.load("changed")
#     changed_probes = dataset.get_probes("changed")
#     print(f"  Changed probes loaded: {len(changed_probes)}")

#     dataset.load("unchanged")
#     unchanged_probes = dataset.get_probes("unchanged")
#     print(f"  Unchanged probes loaded: {len(unchanged_probes)}")

#     # Spot check a few probes
#     print("\nSample probes:")
#     for p in changed_probes[:3]:
#         print(f"  [{p.relation}] {p.prompt}")
#         print(f"    ground_truth:     {p.ground_truth}")
#         print(f"    previous_value:   {p.previous_value}")
#         print(f"    is_contradiction: {p.is_contradiction}")
#         print()

#     # Test contradiction detection
#     print("Testing ContradictionDetector...")
#     registry = MemoryRegistry()
#     detector = ContradictionDetector()
#     conflicts = detector.check(changed_probes, registry)
#     print(f"  Conflicts against empty registry: {len(conflicts)} (expected 0)")

#     for p in changed_probes:
#         registry.write(p, "aug_sep")
#     print(f"  Registry size after writing: {len(registry)}")

#     # Test memory lookups
#     sample = changed_probes[0]
#     active = registry.get_active(sample.subject, sample.relation)
#     print(f"\nMemory lookup for ({sample.subject}, {sample.relation}):")
#     print(f"  Active slot value: {active.value if active else None}")

#     # Save registry
#     registry.save("registry.json")
#     print("\nRegistry saved to registry.json")

#     # Test TSQADataset
#     print("\nLoading TSQADataset...")
#     tsqa = TSQADataset()
#     tsqa.load("train")
#     tsqa_probes = tsqa.get_probes()
#     print(f"  Train probes loaded: {len(tsqa_probes)}")
#     n_hard = sum(1 for p in tsqa_probes if p.metadata.get("is_hard_negative"))
#     print(f"  Hard negatives: {n_hard}")
#     print("\nSample TSQA probes:")
#     for p in tsqa_probes[:3]:
#         print(f"  [{p.relation}] {p.prompt[:80]}...")
#         print(f"    ground_truth: {p.ground_truth}")
#         print(f"    is_hard_negative: {p.metadata.get('is_hard_negative')}")
#         print()

#     # Test TGQADataset
#     print("\nLoading TGQADataset...")
#     tgqa = TGQADataset()
#     tgqa.load("train")
#     tgqa_probes = tgqa.get_probes()
#     print(f"  Train probes loaded: {len(tgqa_probes)}")
#     n_changed = sum(1 for p in tgqa_probes if p.is_changed)
#     n_contra = sum(1 for p in tgqa_probes if p.is_contradiction)
#     print(f"  Changed probes: {n_changed}")
#     print(f"  Contradiction probes: {n_contra}")
#     pairs = tgqa.get_contradiction_pairs()
#     print(f"  Contradiction pairs: {len(pairs)}")
#     print("\nSample TGQA probes:")
#     for p in tgqa_probes[:3]:
#         print(f"  [{p.relation}] {p.prompt}")
#         print(f"    ground_truth: {p.ground_truth}")
#         print(f"    timestamp:    {p.timestamp}")
#         print()

#     print("\nDone.")

import os
from transformers import AutoTokenizer, AutoModelForCausalLM

from casf_dataset_api import TemporalWikiDataset, TSQADataset, TGQADataset, MemoryRegistry
from train_config import TrainConfig
from trainer import CASFTrainer


# PERIODS = ["aug_sep", "sep_oct", "oct_nov", "nov_dec"]
PERIODS = ["aug_sep"]


def build_dataset(dataset_name: str, period: str | None = None):
    if dataset_name == "temporal_wiki":
        return TemporalWikiDataset(period=period)
    if dataset_name == "tsqa":
        return TSQADataset()
    if dataset_name == "tgqa":
        return TGQADataset()
    raise ValueError(f"Unsupported dataset_name: {dataset_name}")


if __name__ == "__main__":
    cfg = TrainConfig.make_config(
        run_id="debug_run",
        model_name="meta-llama/Llama-3.2-3B",
        method="full_ft",
        dataset_name="temporal_wiki",
        batch_size=1,
        grad_accum_steps=1,
        max_passages_per_period=20,
        log_every_n_steps=1,
    )

    print("Training config:")
    print(cfg)

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    cfg.save_json(os.path.join(cfg.checkpoint_dir, f"{cfg.run_id}_config.json"))
    print(f"Saved config to {cfg.checkpoint_dir}/{cfg.run_id}_config.json\n")

    print(f"Loading model: {cfg.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    registry = MemoryRegistry()
    trainer = CASFTrainer(model, tokenizer, cfg, registry)

    if cfg.dataset_name == "temporal_wiki":
        training_units = PERIODS
    else:
        training_units = [cfg.dataset_name]

    for unit in training_units:
        print(f"\n=== Training unit: {unit} ===")

        if cfg.dataset_name == "temporal_wiki":
            dataset = build_dataset(cfg.dataset_name, period=unit)
            dataset.load("changed")
            dataset.load("unchanged")
            period_name = unit
        else:
            dataset = build_dataset(cfg.dataset_name)
            dataset.load("train")
            period_name = unit

        result = trainer.train_period(dataset, period_name)

        print("Training result:")
        print(f"  Final loss: {result['train_loss_final']}")
        print(f"  Passages trained: {result['n_passages_trained']}")
        print(f"  Contradiction passages: {result['n_contradiction_passages']}")
        print(f"  Train duration (sec): {result['train_duration_sec']:.2f}")

        checkpoint_path = os.path.join(cfg.checkpoint_dir, cfg.run_id, str(period_name))
        trainer.checkpoint(str(period_name), checkpoint_path)
        print(f"Checkpoint saved to: {checkpoint_path}")

    print("\nDone.")