"""MLPRouter — lightweight trainable slot-selection router for CASM.

Architecture
------------
query text -> sentence-transformer encoder -> 384-d embedding
optionally: concatenate a one-hot period embedding (4-d) -> 388-d
-> MLP (Linear -> ReLU -> Dropout -> Linear) -> logits over num_slots

Training uses cross-entropy against ground-truth slot assignments.

Unlike SimilarityRouter, this module is a torch.nn.Module and participates
in gradient-based training.  See train_router.py for the training loop.

Usage:
    from training.router import MLPRouter

    router = MLPRouter(num_slots=100)
    logits = router.forward(["query text"], period_ids=[0])  # shape (1, 100)
    slot_id = router.route("query text", period_id=0)
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

PERIOD_MAP: dict[str, int] = {
    "2018": 0,
    "2020": 1,
    "2022": 2,
    "2024": 3,
}
N_PERIODS = len(PERIOD_MAP)


class MLPRouter(nn.Module):
    """MLP-based router that maps (query, period) -> slot logits.

    Parameters
    ----------
    input_dim:
        Dimensionality of the sentence-transformer embeddings.
    num_slots:
        Total number of slots to route between.  The output layer is
        resized dynamically via ``expand_to(new_num_slots)`` when new
        slots are added.
    hidden_dim:
        Hidden layer width.
    use_period:
        If True, concatenate a one-hot period embedding to the query
        embedding before passing through the MLP.  Increases input_dim
        by N_PERIODS.
    encoder_model:
        sentence-transformers model name for query encoding.
    dropout:
        Dropout probability in the MLP.
    """

    def __init__(
        self,
        num_slots: int,
        *,
        input_dim: int = 384,
        hidden_dim: int = 256,
        use_period: bool = True,
        encoder_model: str = "all-MiniLM-L6-v2",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for MLPRouter. "
                "Install it: uv add sentence-transformers"
            ) from exc

        self._encoder = SentenceTransformer(encoder_model)
        # Freeze the encoder — we only train the MLP head.
        for param in self._encoder.parameters():
            param.requires_grad_(False)

        self.use_period = use_period
        self._input_dim = input_dim + (N_PERIODS if use_period else 0)

        self.mlp = nn.Sequential(
            nn.Linear(self._input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_slots),
        )
        self._num_slots = num_slots

    # ------------------------------------------------------------------
    # Forward

    def _encode_queries(self, queries: list[str]) -> torch.Tensor:
        """Encode a list of query strings -> (N, input_dim) float32 tensor."""
        embeddings = self._encoder.encode(queries, convert_to_tensor=True)
        return embeddings.float()

    def forward(
        self,
        queries: list[str],
        period_ids: Optional[list[int]] = None,
    ) -> torch.Tensor:
        """Compute slot logits.

        Parameters
        ----------
        queries:
            List of N query strings.
        period_ids:
            Optional list of N integer period indices (0-3).  Required if
            use_period=True.

        Returns
        -------
        torch.Tensor
            Shape (N, num_slots) — raw logits (not softmaxed).
        """
        embeddings = self._encode_queries(queries)
        device = next(self.mlp.parameters()).device
        embeddings = embeddings.to(device)

        if self.use_period:
            if period_ids is None:
                raise ValueError("period_ids required when use_period=True")
            period_emb = torch.zeros(len(queries), N_PERIODS, device=device)
            for i, pid in enumerate(period_ids):
                period_emb[i, pid] = 1.0
            embeddings = torch.cat([embeddings, period_emb], dim=-1)

        return self.mlp(embeddings)

    def route(
        self,
        query: str,
        period_id: Optional[int] = None,
        *,
        period: Optional[str] = None,
    ) -> int:
        """Return the argmax slot for a single query.

        Parameters
        ----------
        query:
            Natural language query.
        period_id:
            Integer index (0-3) for the time period.
        period:
            Period name ("2018" etc.) — alternative to period_id.

        Returns
        -------
        int
            Slot index with the highest logit.
        """
        if period is not None and period_id is None:
            period_id = PERIOD_MAP[period]
        pids = [period_id] if self.use_period else None
        with torch.no_grad():
            logits = self.forward([query], period_ids=pids)
        return int(logits.argmax(dim=-1).item())

    def route_top_k(
        self,
        query: str,
        k: int,
        period_id: Optional[int] = None,
        *,
        period: Optional[str] = None,
    ) -> list[int]:
        """Return the top-k slot indices (highest logit first)."""
        if period is not None and period_id is None:
            period_id = PERIOD_MAP[period]
        pids = [period_id] if self.use_period else None
        with torch.no_grad():
            logits = self.forward([query], period_ids=pids)
        k = min(k, self._num_slots)
        top_k = torch.topk(logits[0], k).indices
        return top_k.tolist()

    # ------------------------------------------------------------------
    # Dynamic slot expansion

    def expand_to(self, new_num_slots: int) -> None:
        """Expand the output layer to accommodate more slots.

        Existing weights are preserved; new slot weights are randomly
        initialised.  Call this after adding a slot to the MemoryRegistry.

        Parameters
        ----------
        new_num_slots:
            The new total number of slots (must be > current num_slots).
        """
        if new_num_slots <= self._num_slots:
            return

        old_out: nn.Linear = self.mlp[-1]
        hidden_dim = old_out.in_features
        added = new_num_slots - self._num_slots

        new_out = nn.Linear(hidden_dim, new_num_slots)
        # Copy existing weights
        with torch.no_grad():
            new_out.weight[: self._num_slots] = old_out.weight
            new_out.bias[: self._num_slots] = old_out.bias
            # Initialise new rows near zero so they don't dominate immediately
            nn.init.normal_(new_out.weight[self._num_slots:], mean=0.0, std=0.02)
            nn.init.zeros_(new_out.bias[self._num_slots:])

        self.mlp[-1] = new_out
        self._num_slots = new_num_slots

    @property
    def num_slots(self) -> int:
        return self._num_slots
