import sys
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from casf_dataset_api import (
    TemporalWikiDataset,
    TSQADataset,
    TGQADataset,
    ContradictionDetector,
    MemoryRegistry,
    TemporalEvaluator,
)

if __name__ == "__main__":
    print("Loading dataset...")
    dataset = TemporalWikiDataset(period="aug_sep")
    dataset.load("changed")
    changed_probes = dataset.get_probes("changed")
    print(f"  Changed probes loaded: {len(changed_probes)}")

    dataset.load("unchanged")
    unchanged_probes = dataset.get_probes("unchanged")
    print(f"  Unchanged probes loaded: {len(unchanged_probes)}")

    # Spot check a few probes
    print("\nSample probes:")
    for p in changed_probes[:3]:
        print(f"  [{p.relation}] {p.prompt}")
        print(f"    ground_truth:     {p.ground_truth}")
        print(f"    previous_value:   {p.previous_value}")
        print(f"    is_contradiction: {p.is_contradiction}")
        print()

    # Test contradiction detection
    print("Testing ContradictionDetector...")
    registry = MemoryRegistry()
    detector = ContradictionDetector()
    conflicts = detector.check(changed_probes, registry)
    print(f"  Conflicts against empty registry: {len(conflicts)} (expected 0)")

    for p in changed_probes:
        registry.write(p, "aug_sep")
    print(f"  Registry size after writing: {len(registry)}")

    # Test memory lookups
    sample = changed_probes[0]
    active = registry.get_active(sample.subject, sample.relation)
    print(f"\nMemory lookup for ({sample.subject}, {sample.relation}):")
    print(f"  Active slot value: {active.value if active else None}")

    # Save registry
    registry.save("registry.json")
    print("\nRegistry saved to registry.json")

    # Test TSQADataset
    print("\nLoading TSQADataset...")
    tsqa = TSQADataset()
    tsqa.load("train")
    tsqa_probes = tsqa.get_probes()
    print(f"  Train probes loaded: {len(tsqa_probes)}")
    n_hard = sum(1 for p in tsqa_probes if p.metadata.get("is_hard_negative"))
    print(f"  Hard negatives: {n_hard}")
    print("\nSample TSQA probes:")
    for p in tsqa_probes[:3]:
        print(f"  [{p.relation}] {p.prompt[:80]}...")
        print(f"    ground_truth: {p.ground_truth}")
        print(f"    is_hard_negative: {p.metadata.get('is_hard_negative')}")
        print()

    # Test TGQADataset
    print("\nLoading TGQADataset...")
    tgqa = TGQADataset()
    tgqa.load("train")
    tgqa_probes = tgqa.get_probes()
    print(f"  Train probes loaded: {len(tgqa_probes)}")
    n_changed = sum(1 for p in tgqa_probes if p.is_changed)
    n_contra = sum(1 for p in tgqa_probes if p.is_contradiction)
    print(f"  Changed probes: {n_changed}")
    print(f"  Contradiction probes: {n_contra}")
    pairs = tgqa.get_contradiction_pairs()
    print(f"  Contradiction pairs: {len(pairs)}")
    print("\nSample TGQA probes:")
    for p in tgqa_probes[:3]:
        print(f"  [{p.relation}] {p.prompt}")
        print(f"    ground_truth: {p.ground_truth}")
        print(f"    timestamp:    {p.timestamp}")
        print()

    print("\nDone.")