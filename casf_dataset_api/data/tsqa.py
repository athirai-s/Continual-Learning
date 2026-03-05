import json
from typing import Optional
from datasets import load_dataset
from ..casf_types import Probe
from ..dataset import TemporalDataset


VALID_SPLITS = ["train", "val", "test"]


class TSQADataset(TemporalDataset):
    """
    TS-QA: Time-Sensitive QA dataset.

    Notes:
    - Not a KG triple dataset, so we set relation="qa" and subject=original_question (or question).
    - Perturbed examples (is_hard_negative=True) are contradiction-like stress tests,
      but TSQA does not provide explicit old->new answer chains, so previous_value remains None.
    """

    def __init__(self, source_filter: Optional[str] = None):
        self.source_filter = source_filter
        self.snapshot_id = None
        self._ds = None
        self._probes: dict[str, list[Probe]] = {}
        self._passages: dict[str, list[str]] = {}
        self._loaded_split: Optional[str] = None

    def load(self, split: str) -> None:
        if split == "validation":
            split = "val"
        if split not in VALID_SPLITS:
            raise ValueError(f"Unknown split {split!r}. Must be one of {VALID_SPLITS}")
        self._loaded_split = split

        if self._ds is None:
            self._ds = load_dataset("Catkamakura/ts-qa")

        # lazily create probes/passages when requested

    def _iter_rows(self, split: str):
        assert self._ds is not None
        hf_split = "validation" if split == "val" else split
        for ex in self._ds[hf_split]:
            if self.source_filter and ex.get("source") != self.source_filter:
                continue
            yield ex

    @staticmethod
    def _prompt(question: str, context: str) -> str:
        return f"Question: {question}\n\nContext:\n{context}\n\nAnswer:"

    @staticmethod
    def _parse_dimensions(dim_str) -> tuple[str, bool]:
        if not dim_str or not isinstance(dim_str, str):
            return ("UNK", False)
        try:
            obj = json.loads(dim_str)
            return (obj.get("question_type", "UNK"), bool(obj.get("has_critical_dimensions", False)))
        except Exception:
            return ("UNK", False)

    def _load_probes(self, split: str) -> None:
        probes: list[Probe] = []
        for ex in self._iter_rows(split):
            answers = ex.get("answers") or []
            gt = answers[0] if answers else ""

            qtype, has_crit = self._parse_dimensions(ex.get("dimensions"))
            is_hn = bool(ex.get("is_hard_negative", False))

            probes.append(
                Probe(
                    prompt=self._prompt(ex.get("question", ""), ex.get("context", "")),
                    ground_truth=gt,
                    relation="qa",
                    subject=ex.get("original_question") or ex.get("question", ""),
                    current_value=gt,
                    source="tsqa",
                    is_changed=is_hn,
                    timestamp=ex.get("question_timestamp"),
                    previous_value=None,
                    valid_from=ex.get("evidence_timestamp"),
                    valid_until=None,
                    metadata={
                        "id": ex.get("id"),
                        "source": ex.get("source"),
                        "is_hard_negative": is_hn,
                        "question_type": qtype,
                        "has_critical_dimensions": has_crit,
                        "question_timestamp": ex.get("question_timestamp"),
                        "evidence_timestamp": ex.get("evidence_timestamp"),
                    },
                )
            )

        self._probes[split] = probes

    def _load_passages(self, split: str) -> None:
        # For TSQA, "train passages" can be the contexts (evidence articles)
        self._passages[split] = [ex.get("context", "") for ex in self._iter_rows(split)]

    def get_probes(self, split: Optional[str] = None) -> list[Probe]:
        target = split or self._loaded_split
        if target is None:
            raise ValueError("No split loaded. Call load('train'|'validation'|'test') first.")
        if target not in self._probes:
            self._load_probes(target)
        return self._probes[target]

    def get_train_passages(self) -> list[str]:
        if self._loaded_split != "train":
            raise ValueError("Call load('train') before get_train_passages().")
        if "train" not in self._passages:
            self._load_passages("train")
        return self._passages["train"]

    def get_contradiction_pairs(self) -> list[tuple[Probe, Probe]]:
        # No explicit old/new fact chain in TSQA
        return []

if __name__ == "__main__":
    ds = TSQADataset()
    ds.load("train")
    probes = ds.get_probes()
    print("Loaded probes:", len(probes))
    print("First probe:", probes[0])

    # optional: show a few quick stats
    n_hard = sum(1 for p in probes if p.metadata.get("is_hard_negative"))
    print("Hard negatives:", n_hard)