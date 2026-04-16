"""SMF (Sparse Memory Finetuning) model wrapper.

Wraps any causal LM backbone with a frozen backbone and one trainable
sparse memory module per selected layer.  Only the memory parameters
receive gradient updates.
"""

from __future__ import annotations

import math
import os
from typing import Any, Optional

import torch
import torch.nn as nn

from .train_config import TrainConfig


class SparseMemoryBlock(nn.Module):
    """Additive sparse memory injected after a single transformer layer.

    The block holds a bank of ``memory_size`` vectors of ``hidden_size``
    dimensions.  A query projection maps each token's hidden state to
    per-slot attention weights, making the memory contribution
    content-dependent rather than a position-independent bias.

    A global gate bias per slot is initialized so that
    ``sigmoid(bias) ≈ sparsity_ratio``, giving sparse activations from
    the start.  Sparsity is further encouraged by including
    ``sparsity_loss()`` in the training objective.
    """

    def __init__(
        self,
        memory_size: int,
        hidden_size: int,
        sparsity_ratio: float = 0.1,
        query_dependent: bool = True,
        memory_init_std: float = 0.02,
    ) -> None:
        super().__init__()
        self.memory = nn.Parameter(torch.empty(memory_size, hidden_size))
        nn.init.normal_(self.memory, std=memory_init_std)

        # Global gate bias: initialized so sigmoid(bias) ≈ sparsity_ratio,
        # giving a sparse starting point instead of all gates at 0.5.
        init_logit = math.log(sparsity_ratio / (1.0 - sparsity_ratio))
        self.gate_logits = nn.Parameter(torch.full((memory_size,), init_logit))

        # Query projection: maps token hidden states -> per-slot scores so
        # the memory contribution is content-dependent rather than a
        # position-independent bias.  Initialized to zero so the block
        # starts as a pure global-gate memory and becomes query-dependent
        # as it learns.  Set query_dependent=False (e.g. in CASM) when the
        # calling code computes the contribution itself.
        if query_dependent:
            self.query_proj: Optional[nn.Linear] = nn.Linear(hidden_size, memory_size, bias=False)
            nn.init.zeros_(self.query_proj.weight)
        else:
            self.query_proj = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: (batch, seq_len, hidden_size)
        if self.query_proj is not None:
            # query_scores: (batch, seq_len, memory_size) — per-token slot affinities
            query_scores = self.query_proj(hidden_states.to(self.query_proj.weight.dtype))
            # gate: (batch, seq_len, memory_size) — global bias + content signal
            gate = torch.sigmoid(self.gate_logits + query_scores)
        else:
            # Position-independent fallback: global gate broadcast over all tokens
            gate = torch.sigmoid(self.gate_logits)  # (memory_size,)
        # contribution: (batch, seq_len, hidden_size)
        contribution = gate @ self.memory
        return hidden_states + contribution.to(hidden_states.dtype)

    def sparsity_loss(self) -> torch.Tensor:
        """L1 penalty on the global gate activations (encourages sparsity)."""
        return torch.sigmoid(self.gate_logits).sum()


def _get_hidden_size(model: Any) -> int:
    """Inspect model.config for the hidden dimension."""
    cfg = getattr(model, "config", None)
    if cfg is not None:
        for attr in ("hidden_size", "n_embd", "d_model"):
            val = getattr(cfg, attr, None)
            if isinstance(val, int) and val > 0:
                return val
    raise ValueError(
        "Cannot determine hidden size from model config. "
        "Expected model.config.hidden_size / n_embd / d_model."
    )


def _get_transformer_layers(model: Any) -> nn.ModuleList:
    """Return the list of per-layer transformer blocks from a causal LM.

    Tries common attribute paths for GPT-2, LLaMA, and GPT-Neo style
    models.  Raises ``ValueError`` if none match.
    """
    for path in [
        ("transformer", "h"),       # GPT-2
        ("model", "layers"),        # LLaMA / Mistral
        ("transformer", "blocks"),  # GPT-Neo-X style
        ("model", "decoder", "layers"),  # BART-style encoder-decoder
    ]:
        obj: Any = model
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None:
            return obj  # type: ignore[return-value]
    raise ValueError(
        "Cannot find transformer layers in model. "
        "Expected one of: transformer.h, model.layers, transformer.blocks."
    )


class SMFModelWrapper(nn.Module):
    """Wraps a causal LM backbone for Sparse Memory Finetuning.

    - All backbone parameters are frozen immediately on construction.
    - One ``SparseMemoryBlock`` is attached to each layer listed in
      ``cfg.smf_update_layers`` via a forward hook.
    - Only the memory block parameters are trainable.
    - ``save_pretrained`` saves both the backbone and the memory state.
    """

    def __init__(self, backbone: nn.Module, cfg: TrainConfig) -> None:
        super().__init__()
        self.backbone = backbone
        self._smf_cfg = cfg

        # Freeze all backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad_(False)

        hidden_size = _get_hidden_size(self.backbone)
        transformer_layers = _get_transformer_layers(self.backbone)
        n_layers = len(transformer_layers)

        self.memory_blocks: nn.ModuleDict = nn.ModuleDict()
        self._hook_handles: list[Any] = []

        for raw_idx in cfg.smf_update_layers:  # type: ignore[union-attr]
            layer_idx = int(raw_idx)
            if layer_idx >= n_layers:
                raise ValueError(
                    f"smf_update_layers contains index {layer_idx} but the "
                    f"model only has {n_layers} layers."
                )
            block = SparseMemoryBlock(cfg.smf_memory_size, hidden_size, sparsity_ratio=cfg.smf_sparsity_ratio)  # type: ignore[arg-type]
            key = str(layer_idx)
            self.memory_blocks[key] = block

            # Hook: add the memory contribution after this transformer layer
            def _make_hook(mem_block: SparseMemoryBlock):
                def _hook(
                    module: nn.Module,
                    inputs: tuple,
                    output: Any,
                ) -> Any:
                    # GPT-2 blocks return (hidden_states, [present_kv, ...])
                    if isinstance(output, tuple):
                        hidden = output[0]
                        new_hidden = mem_block(hidden)
                        return (new_hidden,) + output[1:]
                    # Some models return only the tensor
                    return mem_block(output)

                return _hook

            handle = transformer_layers[layer_idx].register_forward_hook(
                _make_hook(block)
            )
            self._hook_handles.append(handle)

    # ------------------------------------------------------------------
    # nn.Module interface

    def forward(self, **kwargs: Any) -> Any:  # type: ignore[override]
        """Forward pass through backbone; hooks inject memory contributions."""
        return self.backbone(**kwargs)

    # ------------------------------------------------------------------
    # SMF-specific helpers

    def smf_parameters(self):
        """Yield only the trainable sparse memory parameters."""
        return self.memory_blocks.parameters()

    def compute_regularization_loss(self) -> torch.Tensor:
        """Return the summed sparsity penalty across all memory blocks."""
        device = next(self.memory_blocks.parameters()).device
        total = torch.zeros(1, device=device)
        for block in self.memory_blocks.values():
            total = total + block.sparsity_loss()
        return total.squeeze()

    # ------------------------------------------------------------------
    # Persistence: delegate to backbone and also save memory state

    def save_pretrained(self, path: str) -> None:
        """Save backbone weights and SMF memory state to *path*."""
        self.backbone.save_pretrained(path)
        memory_state = {k: v.state_dict() for k, v in self.memory_blocks.items()}
        torch.save(memory_state, os.path.join(path, "smf_memory.pt"))

    @staticmethod
    def load_memory_into(wrapper: "SMFModelWrapper", path: str) -> None:
        """Load previously-saved SMF memory state from a checkpoint dir."""
        memory_path = os.path.join(path, "smf_memory.pt")
        if not os.path.exists(memory_path):
            return
        state = torch.load(memory_path, map_location="cpu", weights_only=True)
        for key, block in wrapper.memory_blocks.items():
            if key in state:
                block.load_state_dict(state[key])

    # ------------------------------------------------------------------
    # Delegate attribute look-ups needed by the trainer

    def generate(self, **kwargs: Any) -> Any:
        """Delegate generation to backbone; hooks still inject memory."""
        return self.backbone.generate(**kwargs)

    @property
    def config(self) -> Any:
        """Expose backbone model config (needed by trainer for max_length)."""
        return self.backbone.config  # type: ignore[return-value]
