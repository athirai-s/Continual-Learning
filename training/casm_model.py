"""CASM (Contradiction-Aware Sparse Memory) model wrapper.

Wraps any causal LM backbone with:
- A frozen backbone
- A bank of trainable sparse memory slots (SparseMemoryBlocks)
- A CASMRouter that selects top-k slots per input query

The routing query is derived from the mean of the input token embeddings.
The weighted sum of selected slot contributions is injected additively into
the backbone hidden states at the last transformer layer.

New slots can be added at any time (e.g., on contradiction detection), but
only the slots that existed at construction time are reachable via the router.
Closed slots retain their weights and remain queryable but are excluded from
routing.
"""
from __future__ import annotations

import os
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .smf_model import SparseMemoryBlock, _get_hidden_size, _get_transformer_layers
from .train_config import TrainConfig

_DEFAULT_MEMORY_SIZE = 16


class CASMRouter(nn.Module):
    """Routes a query vector to the top-k most relevant memory slots.

    Architecture: two-layer MLP (hidden_size → router_hidden_size → num_slots).
    Temperature scales the logits before top-k selection.
    """

    def __init__(
        self,
        hidden_size: int,
        num_slots: int,
        router_hidden_size: int,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if num_slots < 1:
            raise ValueError("num_slots must be >= 1")
        if router_hidden_size < 1:
            raise ValueError("router_hidden_size must be >= 1")
        self.num_slots = num_slots
        self.temperature = temperature
        self.net = nn.Sequential(
            nn.Linear(hidden_size, router_hidden_size),
            nn.ReLU(),
            nn.Linear(router_hidden_size, num_slots),
        )

    def forward(
        self,
        query: torch.Tensor,
        top_k: int = 1,
        time_signal: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute routing decisions.

        Args:
            query: (..., hidden_size) — query representation per example.
            top_k: number of slots to select; must be <= num_slots.
            time_signal: optional temporal context; currently unused.

        Returns:
            slot_ids: (..., top_k) long tensor — indices into the slot bank.
            weights: (..., top_k) float tensor — softmax-normalised routing weights.
        """
        if top_k > self.num_slots:
            raise ValueError(
                f"top_k={top_k} exceeds num_slots={self.num_slots}"
            )
        logits = self.net(query.to(self.net[0].weight.dtype)) / self.temperature  # (..., num_slots)
        top_values, slot_ids = torch.topk(logits, k=top_k, dim=-1)
        weights = F.softmax(top_values, dim=-1)
        return slot_ids, weights


# ---------------------------------------------------------------------------
# Internal helpers


def _slot_contribution(block: SparseMemoryBlock) -> torch.Tensor:
    """Return the (hidden_size,) additive contribution of a slot block."""
    gate = torch.sigmoid(block.gate_logits)
    return (gate.unsqueeze(-1) * block.memory).sum(0)


def _get_input_embeddings(backbone: Any, input_ids: torch.Tensor) -> torch.Tensor:
    """Extract token embeddings from the backbone without a full forward pass.

    Supports GPT-2 (backbone.transformer.wte) and LLaMA/Mistral style
    (backbone.model.embed_tokens).
    """
    # GPT-2
    if hasattr(backbone, "transformer") and hasattr(backbone.transformer, "wte"):
        return backbone.transformer.wte(input_ids)
    # LLaMA / Mistral
    if hasattr(backbone, "model") and hasattr(backbone.model, "embed_tokens"):
        return backbone.model.embed_tokens(input_ids)
    raise ValueError(
        "Cannot find token embedding table in backbone. "
        "Expected backbone.transformer.wte or backbone.model.embed_tokens."
    )


# ---------------------------------------------------------------------------
# Main wrapper


class CASMModelWrapper(nn.Module):
    """Wraps a causal LM for Contradiction-Aware Sparse Memory fine-tuning.

    Only the slot bank and router parameters are trainable; the backbone
    is fully frozen.

    Slot lifecycle
    --------------
    - Slots 0..casm_num_slots-1 are created at construction time and are all
      reachable by the router.
    - ``add_memory_slot()`` allocates a new slot beyond the router's initial
      capacity (useful for contradiction-triggered branching; the router will
      not select it automatically without re-initialisation).
    - ``close_memory_slot(slot_id)`` excludes a slot from future routing while
      keeping its weights accessible.
    """

    def __init__(self, backbone: nn.Module, cfg: TrainConfig) -> None:
        super().__init__()
        self.backbone = backbone
        self._casm_cfg = cfg

        # Freeze backbone
        for p in self.backbone.parameters():
            p.requires_grad_(False)

        self._hidden_size = _get_hidden_size(backbone)
        transformer_layers = _get_transformer_layers(backbone)

        # --- Slot bank ---
        self.slot_bank: nn.ModuleDict = nn.ModuleDict()
        self._active_slot_ids: list[int] = []
        self._closed_slot_ids: set[int] = set()
        self._next_slot_idx: int = 0
        self._memory_size: int = cfg.casm_memory_size or _DEFAULT_MEMORY_SIZE

        for _ in range(cfg.casm_num_slots):  # type: ignore[arg-type]
            self._create_slot()

        self._slot_usage_counts: dict[int, int] = {sid: 0 for sid in self._active_slot_ids}

        # --- Router ---
        self.router = CASMRouter(
            hidden_size=self._hidden_size,
            num_slots=len(self._active_slot_ids),
            router_hidden_size=cfg.casm_router_hidden_size,  # type: ignore[arg-type]
            temperature=cfg.casm_router_temperature,
        )

        # Holds the routing-weighted memory contribution computed in forward(),
        # consumed by the post-hook on the last transformer layer.
        self._current_memory_contribution: Optional[torch.Tensor] = None

        # Hook on the last transformer layer
        self._hook_handles: list[Any] = []
        handle = transformer_layers[-1].register_forward_hook(self._memory_hook)
        self._hook_handles.append(handle)

    # ------------------------------------------------------------------
    # Slot management

    def _create_slot(self) -> int:
        idx = self._next_slot_idx
        self.slot_bank[str(idx)] = SparseMemoryBlock(
            memory_size=self._memory_size,
            hidden_size=self._hidden_size,
            query_dependent=False,
        )
        self._active_slot_ids.append(idx)
        self._next_slot_idx += 1
        return idx

    def add_memory_slot(self) -> int:
        """Allocate a new slot (e.g., after contradiction detection).

        The new slot is immediately added to the router via _expand_router(),
        so it is selectable from the next forward pass onward.
        """
        new_id = self._create_slot()
        self._slot_usage_counts[new_id] = 0
        self._expand_router()
        return new_id

    def _expand_router(self) -> None:
        """Grow the router's output layer by one neuron for the latest new slot.

        Preserves existing weights; zero-initialises the new neuron.
        After this call router.num_slots == len(_active_slot_ids).
        """
        old_layer = self.router.net[2]  # nn.Linear(router_hidden_size, old_num_slots)
        old_n = old_layer.out_features
        new_n = old_n + 1
        new_layer = nn.Linear(
            old_layer.in_features,
            new_n,
            bias=(old_layer.bias is not None),
        )
        new_layer = new_layer.to(old_layer.weight.device)
        with torch.no_grad():
            new_layer.weight[:old_n] = old_layer.weight
            new_layer.weight[old_n].zero_()
            if old_layer.bias is not None:
                new_layer.bias[:old_n] = old_layer.bias
                new_layer.bias[old_n].zero_()
        self.router.net[2] = new_layer
        self.router.num_slots = new_n

    def close_memory_slot(self, slot_id: int) -> None:
        """Exclude a slot from future routing (weights are retained)."""
        if slot_id in self._active_slot_ids:
            self._active_slot_ids.remove(slot_id)
        self._closed_slot_ids.add(slot_id)

    # ------------------------------------------------------------------
    # Forward hook

    def _memory_hook(self, module: nn.Module, inputs: tuple, output: Any) -> Any:
        if self._current_memory_contribution is None:
            return output
        contrib = self._current_memory_contribution  # (batch, hidden_size)
        if isinstance(output, tuple):
            hidden = output[0]  # (batch, seq_len, hidden_size)
            new_hidden = hidden + contrib.unsqueeze(1)
            return (new_hidden,) + output[1:]
        return output + contrib.unsqueeze(1)

    # ------------------------------------------------------------------
    # nn.Module interface

    def forward(self, input_ids: Optional[torch.Tensor] = None, **kwargs: Any) -> Any:
        """Forward pass with routing-weighted memory injection.

        Computes the routing query from input token embeddings, selects the
        top-k slots, accumulates their weighted contributions, then injects
        the combined contribution into the last transformer layer via a hook.
        """
        if input_ids is not None and len(self._active_slot_ids) > 0:
            embeds = _get_input_embeddings(self.backbone, input_ids)  # (B, T, H)
            query = embeds.mean(dim=1)  # (B, H)

            top_k = min(self._casm_cfg.casm_top_k, len(self._active_slot_ids))  # type: ignore[arg-type]
            slot_ids, weights = self.router(query, top_k=top_k)  # (B, top_k)

            for idx in slot_ids.view(-1).tolist():
                if idx in self._slot_usage_counts:
                    self._slot_usage_counts[idx] += 1

            # Build contribution matrix for all router-reachable slots.
            # torch.stack preserves the autograd graph; in-place index
            # assignment on a zeros tensor does not reliably track gradients.
            device = query.device
            n = self.router.num_slots
            zero = torch.zeros(self._hidden_size, device=device)
            contrib_list = [
                _slot_contribution(self.slot_bank[str(i)])
                if str(i) in self.slot_bank and i not in self._closed_slot_ids
                else zero
                for i in range(n)
            ]
            all_contribs = torch.stack(contrib_list, dim=0)  # (n, H)

            # Index and weight: (B, top_k, H) → sum over top_k → (B, H)
            batch_size = query.shape[0]
            selected = all_contribs[slot_ids.view(-1)].view(batch_size, top_k, self._hidden_size)
            self._current_memory_contribution = (weights.unsqueeze(-1) * selected).sum(1).to(embeds.dtype)
        else:
            self._current_memory_contribution = None

        result = self.backbone(input_ids=input_ids, **kwargs)
        self._current_memory_contribution = None
        return result

    # ------------------------------------------------------------------
    # CASM-specific helpers

    def casm_parameters(self):
        """Yield only the trainable CASM parameters (slot bank + router)."""
        yield from self.slot_bank.parameters()
        yield from self.router.parameters()

    def compute_sparsity_loss(self) -> torch.Tensor:
        """Sum of sparsity losses from all active slot blocks."""
        device = next(self.slot_bank.parameters()).device
        total = torch.zeros(1, device=device)
        for sid in self._active_slot_ids:
            key = str(sid)
            if key in self.slot_bank:
                total = total + self.slot_bank[key].sparsity_loss()
        return total.squeeze()

    def compute_overlap_loss(self) -> torch.Tensor:
        """Pairwise cosine similarity penalty across active slot contributions.

        Encourages slots to learn distinct representations. Returns a scalar;
        zero when fewer than two active slots exist.
        """
        device = next(self.slot_bank.parameters()).device
        total = torch.zeros(1, device=device)
        contribs = []
        for sid in self._active_slot_ids:
            key = str(sid)
            if key in self.slot_bank:
                contribs.append(_slot_contribution(self.slot_bank[key]))  # (H,)
        if len(contribs) < 2:
            return total.squeeze()
        C = torch.stack(contribs, dim=0)                          # (n, H)
        norms = C.norm(dim=1, keepdim=True).clamp(min=1e-8)
        C_norm = C / norms                                        # (n, H)
        sim_matrix = C_norm @ C_norm.t()                         # (n, n)
        n = sim_matrix.shape[0]
        mask = torch.triu(torch.ones(n, n, device=device, dtype=torch.bool), diagonal=1)
        return sim_matrix[mask].sum()

    # ------------------------------------------------------------------
    # Persistence

    def save_pretrained(self, path: str) -> None:
        """Save backbone weights and all CASM state to *path*."""
        self.backbone.save_pretrained(path)
        state = {
            "slot_bank": {k: v.state_dict() for k, v in self.slot_bank.items()},
            "router": self.router.state_dict(),
            "active_slot_ids": list(self._active_slot_ids),
            "closed_slot_ids": list(self._closed_slot_ids),
            "next_slot_idx": self._next_slot_idx,
            "memory_size": self._memory_size,
            "slot_usage_counts": dict(self._slot_usage_counts),
        }
        torch.save(state, os.path.join(path, "casm_memory.pt"))

    @staticmethod
    def load_memory_into(wrapper: "CASMModelWrapper", path: str) -> None:
        """Restore slot bank and router state from a checkpoint directory.

        Handles the case where contradiction branching added slots during the
        saved run (checkpoint has more slots than the freshly-built wrapper).
        """
        memory_path = os.path.join(path, "casm_memory.pt")
        if not os.path.exists(memory_path):
            return
        state = torch.load(memory_path, map_location="cpu", weights_only=True)

        # Create any slots present in the checkpoint but missing from the wrapper
        # (slots added via add_memory_slot() during the saved run).
        memory_size = state.get("memory_size", wrapper._memory_size)
        for key in state["slot_bank"]:
            if key not in wrapper.slot_bank:
                wrapper.slot_bank[key] = SparseMemoryBlock(
                    memory_size=memory_size,
                    hidden_size=wrapper._hidden_size,
                    query_dependent=False,
                )

        # Load slot weights.
        for key, sd in state["slot_bank"].items():
            if key in wrapper.slot_bank:
                wrapper.slot_bank[key].load_state_dict(sd)

        # Resize the router output layer to match the checkpoint before loading
        # its state dict (router may have grown via _expand_router during the run).
        checkpoint_num_slots = state["router"]["net.2.weight"].shape[0]
        if checkpoint_num_slots != wrapper.router.num_slots:
            old_layer = wrapper.router.net[2]
            new_layer = nn.Linear(
                old_layer.in_features,
                checkpoint_num_slots,
                bias=(old_layer.bias is not None),
            )
            wrapper.router.net[2] = new_layer
            wrapper.router.num_slots = checkpoint_num_slots

        wrapper.router.load_state_dict(state["router"])
        wrapper._active_slot_ids = list(state["active_slot_ids"])
        wrapper._closed_slot_ids = set(state["closed_slot_ids"])
        wrapper._next_slot_idx = state["next_slot_idx"]

        # Restore per-slot usage counts (saved post-reset at period boundaries,
        # or mid-period when a within-period checkpoint was written).
        if "slot_usage_counts" in state:
            wrapper._slot_usage_counts = {int(k): v for k, v in state["slot_usage_counts"].items()}
        else:
            # Backward compat: checkpoint predates usage-count persistence.
            all_slot_ids = set(state["active_slot_ids"]) | set(state["closed_slot_ids"])
            wrapper._slot_usage_counts = {sid: 0 for sid in all_slot_ids}

    def generate(self, **kwargs: Any) -> Any:
        """Delegate generation to backbone; memory hook still fires."""
        return self.backbone.generate(**kwargs)

    # ------------------------------------------------------------------
    # Backbone config delegation

    @property
    def config(self) -> Any:
        return self.backbone.config
