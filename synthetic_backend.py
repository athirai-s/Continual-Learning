import json
from pathlib import Path
from typing import Optional

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from casf_dataset_api import Probe, TemporalDataset


SYNTHETIC_VOCAB_SIZE = 258
SYNTHETIC_MAX_LENGTH = 128


class SyntheticTokenizer:
    def __init__(self, vocab_size: int = SYNTHETIC_VOCAB_SIZE):
        self.vocab_size = vocab_size
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.pad_token_id = 0
        self.eos_token_id = 1

    def _encode(self, text: str, max_length: int) -> list[int]:
        token_ids = [2 + (byte % (self.vocab_size - 2)) for byte in text.encode("utf-8")]
        token_ids = token_ids[: max(0, max_length - 1)]
        token_ids.append(self.eos_token_id)
        return token_ids

    def __call__(
        self,
        text: str,
        truncation: bool = True,
        max_length: int = SYNTHETIC_MAX_LENGTH,
        padding: str | None = "max_length",
        return_tensors: Optional[str] = None,
    ) -> dict[str, torch.Tensor | list[list[int]]]:
        if not truncation:
            raise ValueError("SyntheticTokenizer expects truncation=True")

        token_ids = self._encode(text, max_length=max_length)
        attention_mask = [1] * len(token_ids)

        if padding == "max_length" and len(token_ids) < max_length:
            pad_len = max_length - len(token_ids)
            token_ids = token_ids + [self.pad_token_id] * pad_len
            attention_mask = attention_mask + [0] * pad_len

        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor([token_ids], dtype=torch.long),
                "attention_mask": torch.tensor([attention_mask], dtype=torch.long),
            }

        return {
            "input_ids": [token_ids],
            "attention_mask": [attention_mask],
        }

    def save_pretrained(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "synthetic_tokenizer.json", "w") as f:
            json.dump(
                {
                    "vocab_size": self.vocab_size,
                    "pad_token": self.pad_token,
                    "eos_token": self.eos_token,
                    "pad_token_id": self.pad_token_id,
                    "eos_token_id": self.eos_token_id,
                },
                f,
                indent=2,
            )


def build_synthetic_model(vocab_size: int = SYNTHETIC_VOCAB_SIZE) -> GPT2LMHeadModel:
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=SYNTHETIC_MAX_LENGTH,
        n_ctx=SYNTHETIC_MAX_LENGTH,
        n_embd=32,
        n_layer=2,
        n_head=4,
        bos_token_id=1,
        eos_token_id=1,
    )
    return GPT2LMHeadModel(config)


class SyntheticTemporalDataset(TemporalDataset):
    VALID_SPLITS = {"train", "changed", "unchanged"}

    def __init__(self):
        self.snapshot_id = "synthetic"
        self._loaded_split: Optional[str] = None
        self._train_passages = [
            "Alpha is associated with the new synthetic value.",
            "Beta remains associated with the steady synthetic value.",
            "Gamma switched from old to updated during this synthetic period.",
        ]
        self._probes = {
            "changed": [
                Probe(
                    prompt="Alpha relation ____.",
                    ground_truth="new value",
                    relation="relation",
                    subject="Alpha",
                    current_value="new value",
                    source="synthetic",
                    is_changed=True,
                    timestamp="synthetic_period",
                    previous_value="old value",
                    valid_from="synthetic_period",
                    metadata={"period": "synthetic_period"},
                ),
                Probe(
                    prompt="Gamma relation ____.",
                    ground_truth="updated value",
                    relation="relation",
                    subject="Gamma",
                    current_value="updated value",
                    source="synthetic",
                    is_changed=True,
                    timestamp="synthetic_period",
                    previous_value="stale value",
                    valid_from="synthetic_period",
                    metadata={"period": "synthetic_period"},
                ),
            ],
            "unchanged": [
                Probe(
                    prompt="Beta relation ____.",
                    ground_truth="steady value",
                    relation="relation",
                    subject="Beta",
                    current_value="steady value",
                    source="synthetic",
                    is_changed=False,
                    timestamp="synthetic_period",
                    previous_value=None,
                    valid_from="synthetic_period",
                    metadata={"period": "synthetic_period"},
                ),
                Probe(
                    prompt="Alpha relation ____.",
                    ground_truth="old value",
                    relation="relation",
                    subject="Alpha",
                    current_value="old value",
                    source="synthetic",
                    is_changed=False,
                    timestamp="prior_period",
                    previous_value=None,
                    valid_from="prior_period",
                    valid_until="synthetic_period",
                    metadata={"period": "prior_period"},
                ),
            ],
        }

    def load(self, split: str) -> None:
        if split not in self.VALID_SPLITS:
            raise ValueError(f"Unknown split {split!r}. Must be one of {sorted(self.VALID_SPLITS)}")
        self._loaded_split = split

    def get_probes(self, split: Optional[str] = None) -> list[Probe]:
        target = split or self._loaded_split
        if target is None or target == "train":
            raise ValueError("No probe split loaded. Call load('changed') or load('unchanged') first.")
        return self._probes[target]

    def get_train_passages(self) -> list[str]:
        return self._train_passages

    def get_contradiction_pairs(self) -> list[tuple[Probe, Probe]]:
        unchanged = {(probe.subject, probe.relation): probe for probe in self._probes["unchanged"]}
        return [
            (unchanged[(probe.subject, probe.relation)], probe)
            for probe in self._probes["changed"]
            if probe.is_contradiction and (probe.subject, probe.relation) in unchanged
        ]
