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
    """Return the (hidden_size,) position-independent contribution of a slot.

    Uses only the global gate (ignores query_proj).  Used for overlap_loss
    where we need a fixed per-slot summary independent of input tokens.
    """
    gate = torch.sigmoid(block.gate_logits)
    return (gate.unsqueeze(-1) * block.memory).sum(0)


def _slot_contribution_tokens(
    block: SparseMemoryBlock, hidden_states: torch.Tensor,
) -> torch.Tensor:
    """Return per-token contribution (batch, seq_len, hidden_size) from a slot.

    If the block has a query_proj (query_dependent=True), the gate is
    content-dependent: each token gets its own gating pattern.  Otherwise
    falls back to the global position-independent gate.
    """
    if block.query_proj is not None:
        query_scores = block.query_proj(
            hidden_states.to(block.query_proj.weight.dtype)
        )
        gate = torch.sigmoid(block.gate_logits + query_scores)  # (B, T, mem)
    else:
        gate = torch.sigmoid(block.gate_logits)  # (mem,)
    return (gate @ block.memory).to(hidden_states.dtype)


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
        if cfg.casm_router_type == "similarity":
            from training.router_baseline import SimilarityRouter
            self.router = SimilarityRouter()
        else:
            self.router = CASMRouter(
                hidden_size=self._hidden_size,
                num_slots=len(self._active_slot_ids),
                router_hidden_size=cfg.casm_router_hidden_size,  # type: ignore[arg-type]
                temperature=cfg.casm_router_temperature,
            )

        # Routing decisions computed in forward(), consumed by layer hooks.
        self._routing_slot_ids: Optional[torch.Tensor] = None   # (B, top_k)
        self._routing_weights: Optional[torch.Tensor] = None    # (B, top_k)

        # Hook on the last N transformer layers (default: last layer only)
        num_injection = cfg.casm_num_injection_layers or 1
        self._num_injection_layers: int = min(num_injection, len(transformer_layers))
        self._hook_handles: list[Any] = []
        for layer in transformer_layers[-self._num_injection_layers:]:
            handle = layer.register_forward_hook(self._memory_hook)
            self._hook_handles.append(handle)

    # ------------------------------------------------------------------
    # Slot management

    def _create_slot(self) -> int:
        idx = self._next_slot_idx
        block = SparseMemoryBlock(
            memory_size=self._memory_size,
            hidden_size=self._hidden_size,
            query_dependent=True,
        )
        # Keep new slots on the same device as existing ones (e.g. CUDA).
        if self.slot_bank:
            existing_device = next(iter(self.slot_bank.values())).gate_logits.device
            block = block.to(existing_device)
        self.slot_bank[str(idx)] = block
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

        No-ops for SimilarityRouter (no trainable weights to expand).
        """
        if not hasattr(self.router, "net"):
            return
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
    # Forward hooks

    def _memory_hook(self, module: nn.Module, inputs: tuple, output: Any) -> Any:
        if self._routing_slot_ids is None:
            return output

        if isinstance(output, tuple):
            hidden = output[0]  # (batch, seq_len, hidden_size)
        else:
            hidden = output

        batch_size = hidden.shape[0]
        top_k = self._routing_slot_ids.shape[1]

        # Accumulate contributions as a list of full-batch tensors, then sum.
        # This avoids in-place indexed assignment on a non-requiring-grad buffer
        # (total_contrib[mask] +=) which breaks the autograd graph from the LM
        # loss back to the slot bank parameters.  torch.index_put (out-of-place)
        # correctly propagates gradients through the placed values.
        contrib_parts: list[torch.Tensor] = []

        for k in range(top_k):
            slot_ids_k = self._routing_slot_ids[:, k]  # (B,)
            weights_k = self._routing_weights[:, k]    # (B,)

            for sid in slot_ids_k.unique().tolist():
                key = str(int(sid))
                if key not in self.slot_bank or int(sid) in self._closed_slot_ids:
                    continue
                mask = (slot_ids_k == sid)  # (B,) bool
                if not mask.any():
                    continue
                # Per-token contribution for batch items routed to this slot
                slot_hidden = hidden[mask]  # (n, T, H)
                contrib = _slot_contribution_tokens(
                    self.slot_bank[key], slot_hidden,
                )  # (n, T, H)
                slot_weights = weights_k[mask]  # (n,)
                weighted = slot_weights.unsqueeze(-1).unsqueeze(-1) * contrib  # (n, T, H)

                # Expand (n, T, H) → (B, T, H) without in-place ops so that
                # autograd can trace gradients back through `weighted`.
                indices = mask.nonzero(as_tuple=False).view(-1)  # (n,)
                full = torch.zeros(
                    batch_size, hidden.shape[1], hidden.shape[2],
                    dtype=weighted.dtype, device=weighted.device,
                )
                full = torch.index_put(full, (indices,), weighted)  # out-of-place
                contrib_parts.append(full)

        if contrib_parts:
            total_contrib = (sum(contrib_parts) / self._num_injection_layers).to(hidden.dtype)  # type: ignore[arg-type]
        else:
            total_contrib = torch.zeros_like(hidden)

        if isinstance(output, tuple):
            return (hidden + total_contrib,) + output[1:]
        return output + total_contrib

    # ------------------------------------------------------------------
    # nn.Module interface

    def forward(self, input_ids: Optional[torch.Tensor] = None, **kwargs: Any) -> Any:
        """Forward pass with per-token routing-weighted memory injection.

        Computes the routing query from input token embeddings, selects the
        top-k slots, then stores the routing decision for the layer hooks.
        Each hook computes per-token contributions using the actual hidden
        states at that layer (content-dependent gating via query_proj).
        """
        if input_ids is not None and len(self._active_slot_ids) > 0:
            embeds = _get_input_embeddings(self.backbone, input_ids)  # (B, T, H)
            query = embeds.mean(dim=1)  # (B, H)

            top_k = min(self._casm_cfg.casm_top_k, len(self._active_slot_ids))  # type: ignore[arg-type]
            slot_ids, weights = self.router(query, top_k=top_k)  # (B, top_k)

            for idx in slot_ids.view(-1).tolist():
                if idx in self._slot_usage_counts:
                    self._slot_usage_counts[idx] += 1

            # Store routing decisions; hooks compute per-token contributions.
            self._routing_slot_ids = slot_ids
            self._routing_weights = weights
        else:
            self._routing_slot_ids = None
            self._routing_weights = None

        result = self.backbone(input_ids=input_ids, **kwargs)
        self._routing_slot_ids = None
        self._routing_weights = None
        return result

    # ------------------------------------------------------------------
    # CASM-specific helpers

    def casm_parameters(self):
        """Yield only the trainable CASM parameters (slot bank + router).

        SimilarityRouter has no trainable parameters, so router.parameters()
        is only called when the router is a torch.nn.Module.
        """
        yield from self.slot_bank.parameters()
        if hasattr(self.router, "parameters"):
            yield from self.router.parameters()

    def compute_sparsity_loss(self) -> torch.Tensor:
        """Sum of sparsity losses from all active slot blocks."""
        device = next(self.slot_bank.parameters()).device
        total = torch.zeros(1, device=device)
        for sid in self._active_slot_ids:
            key = str(sid)
            if key in self.slot_bank:
                total = total + self.slot_bank[key].sparsity_loss().to(device)
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
                contribs.append(_slot_contribution(self.slot_bank[key]).to(device))  # (H,)
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
        router_state = self.router.state_dict() if hasattr(self.router, "state_dict") else None
        state = {
            "slot_bank": {k: v.state_dict() for k, v in self.slot_bank.items()},
            "router": router_state,
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

        # Determine target device from the wrapper's existing slot bank.
        target_device = next(wrapper.slot_bank.parameters()).device

        # Create any slots present in the checkpoint but missing from the wrapper
        # (slots added via add_memory_slot() during the saved run).
        memory_size = state.get("memory_size", wrapper._memory_size)
        for key in state["slot_bank"]:
            if key not in wrapper.slot_bank:
                wrapper.slot_bank[key] = SparseMemoryBlock(
                    memory_size=memory_size,
                    hidden_size=wrapper._hidden_size,
                    query_dependent=True,
                ).to(target_device)

        # Load slot weights then move to target device (checkpoint was loaded
        # map_location="cpu", so tensors start on CPU).
        for key, sd in state["slot_bank"].items():
            if key in wrapper.slot_bank:
                wrapper.slot_bank[key].load_state_dict(sd, strict=False)
                wrapper.slot_bank[key] = wrapper.slot_bank[key].to(target_device)

        # Resize the router output layer to match the checkpoint before loading
        # its state dict (router may have grown via _expand_router during the run).
        # SimilarityRouter has no state dict — skip router restore entirely.
        router_state = state.get("router")
        if router_state is not None and hasattr(wrapper.router, "net"):
            checkpoint_num_slots = router_state["net.2.weight"].shape[0]
            if checkpoint_num_slots != wrapper.router.num_slots:
                old_layer = wrapper.router.net[2]
                new_layer = nn.Linear(
                    old_layer.in_features,
                    checkpoint_num_slots,
                    bias=(old_layer.bias is not None),
                )
                wrapper.router.net[2] = new_layer
                wrapper.router.num_slots = checkpoint_num_slots
            wrapper.router.load_state_dict(router_state)
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

    def generate(self, input_ids: Optional[torch.Tensor] = None, **kwargs: Any) -> Any:
        """Delegate generation to backbone with memory injection active.

        Computes routing from the prompt embeddings BEFORE calling
        backbone.generate(), so the forward hook sees a valid
        _routing_slot_ids tensor and injects the correct memory slot
        into every generated token's hidden states.
        """
        if input_ids is not None and len(self._active_slot_ids) > 0:
            with torch.no_grad():
                embeds = _get_input_embeddings(self.backbone, input_ids)  # (B, T, H)
                query = embeds.mean(dim=1)  # (B, H)
                top_k = min(self._casm_cfg.casm_top_k, len(self._active_slot_ids))  # type: ignore[arg-type]
                slot_ids, weights = self.router(query, top_k=top_k)
            self._routing_slot_ids = slot_ids
            self._routing_weights = weights
        try:
            result = self.backbone.generate(input_ids=input_ids, **kwargs)
        finally:
            self._routing_slot_ids = None
            self._routing_weights = None
        return result

    # ------------------------------------------------------------------
    # Backbone config delegation

    @property
    def config(self) -> Any:
        return self.backbone.config
