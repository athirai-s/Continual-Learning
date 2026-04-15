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
    # Inspection helpers

    def registered_slots(self) -> list[int]:
        """Return sorted list of registered slot IDs."""
        return sorted(self._slot_embeddings.keys())

    def slot_metadata(self, slot_id: int) -> dict:
        """Return the metadata dict for a registered slot."""
        return dict(self._slot_metadata[slot_id])
