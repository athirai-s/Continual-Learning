"""SimilarityRouter — zero-training baseline router for CASM.

Routes a query to the most similar memory slot using cosine similarity over
sentence embeddings.  Requires no training data; useful to verify the data
pipeline end-to-end and to establish a baseline routing_acc to beat.

Usage:
    router = SimilarityRouter()
    router.register_slot(slot_id=0, metadata={
        "entity": "Veldris Corp",
        "relation": "harbour_master",
        "period": "2018",
        "value": "Maren Holt",
    })
    best_slot = router.route("Who is the harbour master of Veldris Corp?", period="2018")
"""
from __future__ import annotations

from typing import Optional


class SimilarityRouter:
    """Cosine-similarity router over slot embeddings.

    Parameters
    ----------
    model_name:
        sentence-transformers model to use for encoding.  The default is a
        small, fast model suitable for CPU inference.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        # Lazy import so the module can be imported without the dependency
        # being present at test-collection time.
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for SimilarityRouter. "
                "Install it: uv add sentence-transformers"
            ) from exc

        self._encoder = SentenceTransformer(model_name)
        # slot_id -> numpy embedding vector
        self._slot_embeddings: dict[int, "np.ndarray"] = {}
        # slot_id -> raw metadata dict
        self._slot_metadata: dict[int, dict] = {}

    def register_slot(self, slot_id: int, metadata: dict) -> None:
        """Register a memory slot and pre-compute its embedding.

        Parameters
        ----------
        slot_id:
            Integer slot ID (must match the MemoryRegistry slot_id).
        metadata:
            Dict with at least "entity", "relation", and "period" keys.
            Optionally "value" for richer embedding text.
        """
        parts = [
            str(metadata.get("entity", "")),
            str(metadata.get("relation", "")).replace("_", " "),
            str(metadata.get("period", "")),
        ]
        if "value" in metadata:
            parts.append(str(metadata["value"]))
        text = " ".join(p for p in parts if p)
        import numpy as np
        emb = self._encoder.encode(text, normalize_embeddings=True)
        self._slot_embeddings[slot_id] = np.array(emb, dtype="float32")
        self._slot_metadata[slot_id] = dict(metadata)

    def route(self, query: str, period: Optional[str] = None) -> Optional[int]:
        """Return the slot_id most similar to *query*.

        Parameters
        ----------
        query:
            Natural language query string (e.g. the probe's prompt).
        period:
            Optional period hint appended to the query before encoding.

        Returns
        -------
        int or None
            The slot_id with the highest cosine similarity, or None if no
            slots have been registered.
        """
        if not self._slot_embeddings:
            return None

        import numpy as np

        query_text = f"{query} {period}" if period else query
        query_emb = np.array(
            self._encoder.encode(query_text, normalize_embeddings=True),
            dtype="float32",
        )

        best_id: Optional[int] = None
        best_score = -float("inf")
        for slot_id, slot_emb in self._slot_embeddings.items():
            score = float(np.dot(query_emb, slot_emb))
            if score > best_score:
                best_score = score
                best_id = slot_id
        return best_id

    def route_top_k(self, query: str, k: int, period: Optional[str] = None) -> list[int]:
        """Return the top-k slot_ids sorted by descending similarity.

        If fewer than k slots are registered, returns all of them.
        """
        if not self._slot_embeddings:
            return []

        import numpy as np

        query_text = f"{query} {period}" if period else query
        query_emb = np.array(
            self._encoder.encode(query_text, normalize_embeddings=True),
            dtype="float32",
        )

        scores = [
            (float(np.dot(query_emb, emb)), sid)
            for sid, emb in self._slot_embeddings.items()
        ]
        scores.sort(reverse=True)
        return [sid for _, sid in scores[:k]]

    # ------------------------------------------------------------------
    # Tensor interface — matches CASMRouter.forward so SimilarityRouter can
    # be used as a drop-in inside CASMModelWrapper.

    def __call__(
        self,
        query: "torch.Tensor",
        top_k: int = 1,
        **kwargs,
    ) -> "tuple[torch.Tensor, torch.Tensor]":
        """Tensor routing interface matching CASMRouter.forward.

        Zero-training baseline: cosine similarity between the query hidden
        state and fixed randomly-initialised slot prototype vectors.

        Args:
            query:  (B, H) float tensor — mean input token embeddings.
            top_k:  number of slots to select.

        Returns:
            slot_ids: (B, top_k) long tensor — slot indices.
            weights:  (B, top_k) float tensor — softmax-normalised scores.
        """
        import torch
        import torch.nn.functional as F

        device = query.device
        B, H = query.shape

        slot_ids_list = sorted(self._slot_embeddings.keys())
        n_slots = len(slot_ids_list)
        k = min(top_k, max(n_slots, 1))

        if n_slots == 0:
            ids = torch.zeros(B, k, dtype=torch.long, device=device)
            wts = torch.full((B, k), 1.0 / k, device=device)
            return ids, wts

        # Lazily initialise fixed random unit-vector prototypes in hidden-state
        # space. Seeded for reproducibility; never updated during training.
        if not hasattr(self, "_proto_tensors") or self._proto_tensors.shape != (n_slots, H):
            gen = torch.Generator().manual_seed(42)
            protos = torch.randn(n_slots, H, generator=gen)
            self._proto_tensors = F.normalize(protos, dim=-1)

        protos = self._proto_tensors.to(device=device, dtype=query.float().dtype)
        sims = F.normalize(query.float(), dim=-1) @ protos.T  # (B, S)

        top_vals, top_local = torch.topk(sims, k=k, dim=-1)  # (B, k)
        weights = F.softmax(top_vals, dim=-1)
        id_map = torch.tensor(slot_ids_list, dtype=torch.long, device=device)
        return id_map[top_local], weights

    # ------------------------------------------------------------------
    # Inspection helpers

    def registered_slots(self) -> list[int]:
        """Return sorted list of registered slot IDs."""
        return sorted(self._slot_embeddings.keys())

    def slot_metadata(self, slot_id: int) -> dict:
        """Return the metadata dict for a registered slot."""
        return dict(self._slot_metadata[slot_id])
