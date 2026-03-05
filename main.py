import sys
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from casf_dataset_api import (
    TemporalWikiDataset,
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

    print("\nDone.")