"""Router training loop for MLPRouter.

Trains the router with full supervision: the synthetic dataset has ground-truth
slot assignments (one slot per entity/relation/period combination), so we use
cross-entropy loss directly.

The slot_map argument maps (subject, relation, period) -> slot_id.  Build it
from the MemoryRegistry after seeding facts for all periods.

Usage:
    from casf_dataset_api import SyntheticDataset, MemoryRegistry
    from training.router import MLPRouter, PERIOD_MAP
    from training.train_router import build_slot_map, RouterDataset, train_router

    # 1. Seed the registry with all periods
    registry = MemoryRegistry()
    for period in ["2018", "2020", "2022", "2024"]:
        ds = SyntheticDataset(period)
        for probe in ds.get_probes("changed"):
            registry.write(probe, period)
        for probe in ds.get_probes("unchanged"):
            registry.write(probe, period)  # only writes new value if changed

    # 2. Build slot map from registry
    slot_map = build_slot_map(registry)

    # 3. Collect all probes
    all_probes = []
    for period in ["2018", "2020", "2022", "2024"]:
        ds = SyntheticDataset(period)
        all_probes.extend(ds.get_probes("changed"))
        all_probes.extend(ds.get_probes("unchanged"))

    # 4. Train
    router = MLPRouter(num_slots=len(slot_map))
    train_router(router, all_probes, slot_map)
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from casf_dataset_api.casf_types import Probe
from casf_dataset_api.memory import MemoryRegistry

from .router import MLPRouter, PERIOD_MAP


# ---------------------------------------------------------------------------
# Slot map construction

def build_slot_map(registry: MemoryRegistry) -> dict[tuple[str, str, str], int]:
    """Build a (subject, relation, period) -> slot_id mapping from the registry.

    For each (subject, relation) chain, each slot is valid during a range of
    periods.  We map every period within that range to the slot's ID.

    Parameters
    ----------
    registry:
        A MemoryRegistry populated with slots from all training periods.

    Returns
    -------
    dict mapping (subject, relation, period) -> slot_id.
    """
    from casf_dataset_api.memory import PERIOD_ORDER

    slot_map: dict[tuple[str, str, str], int] = {}
    for slot in registry._slots:
        from_idx = PERIOD_ORDER.index(slot.valid_from) if slot.valid_from in PERIOD_ORDER else None
        if from_idx is None:
            continue
        # Determine end index (exclusive)
        if slot.valid_until is not None and slot.valid_until in PERIOD_ORDER:
            until_idx = PERIOD_ORDER.index(slot.valid_until)
        else:
            until_idx = len(PERIOD_ORDER)

        for i in range(from_idx, until_idx):
            period = PERIOD_ORDER[i]
            key = (slot.subject, slot.relation, period)
            slot_map[key] = slot.slot_id

    return slot_map


# ---------------------------------------------------------------------------
# Dataset

class RouterDataset(Dataset):
    """PyTorch dataset of (prompt, period_id, slot_id) tuples.

    Filters out probes whose (subject, relation, period) combination is not
    in slot_map.
    """

    def __init__(self, probes: list[Probe], slot_map: dict[tuple[str, str, str], int]) -> None:
        self.items: list[tuple[str, int, int]] = []
        skipped = 0
        for probe in probes:
            period = probe.timestamp
            if period is None:
                skipped += 1
                continue
            key = (probe.subject, probe.relation, period)
            slot_id = slot_map.get(key)
            if slot_id is None:
                skipped += 1
                continue
            period_id = PERIOD_MAP.get(period)
            if period_id is None:
                skipped += 1
                continue
            self.items.append((probe.prompt, period_id, slot_id))
        if skipped:
            print(f"  [RouterDataset] skipped {skipped} probes (not in slot_map)")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[str, int, int]:
        return self.items[idx]


def _collate(batch: list[tuple[str, int, int]]):
    """Collate a batch of (prompt, period_id, slot_id) items."""
    prompts = [item[0] for item in batch]
    period_ids = [item[1] for item in batch]
    slot_ids = torch.tensor([item[2] for item in batch], dtype=torch.long)
    return prompts, period_ids, slot_ids


# ---------------------------------------------------------------------------
# Training loop

def train_router(
    router: MLPRouter,
    probes: list[Probe],
    slot_map: dict[tuple[str, str, str], int],
    *,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: Optional[torch.device] = None,
    log_every: int = 1,
) -> list[float]:
    """Train the MLPRouter with cross-entropy supervision.

    Parameters
    ----------
    router:
        An MLPRouter instance.  Should already be expanded to cover all
        slot IDs present in slot_map.
    probes:
        List of Probe instances (all periods).
    slot_map:
        Mapping from (subject, relation, period) -> slot_id.
    epochs:
        Number of full passes over the data.
    batch_size:
        Training batch size.
    lr:
        Learning rate for AdamW.
    device:
        Torch device.  Defaults to CUDA if available, else CPU.
    log_every:
        Log loss every N epochs.

    Returns
    -------
    list[float]
        Per-epoch average loss.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    router = router.to(device)
    dataset = RouterDataset(probes, slot_map)
    if len(dataset) == 0:
        print("  [train_router] No valid training examples in slot_map. Aborting.")
        return []

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=_collate,
    )

    # Only train the MLP head (encoder is frozen)
    trainable = [p for p in router.mlp.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr)
    criterion = nn.CrossEntropyLoss()

    epoch_losses: list[float] = []

    for epoch in range(1, epochs + 1):
        router.train()
        total_loss = 0.0
        total_correct = 0
        n_examples = 0

        for prompts, period_ids, slot_ids in dataloader:
            slot_ids = slot_ids.to(device)
            logits = router.forward(prompts, period_ids=period_ids)
            loss = criterion(logits, slot_ids)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()

            total_loss += loss.item() * len(prompts)
            preds = logits.argmax(dim=-1)
            total_correct += (preds == slot_ids).sum().item()
            n_examples += len(prompts)

        avg_loss = total_loss / max(n_examples, 1)
        acc = total_correct / max(n_examples, 1)
        epoch_losses.append(avg_loss)

        if epoch % log_every == 0:
            print(
                f"  [router] epoch {epoch:3d}/{epochs}"
                f"  loss={avg_loss:.4f}"
                f"  acc={acc:.3f}"
                f"  n={n_examples}"
            )

    return epoch_losses
