from abc import ABC, abstractmethod
from typing import Iterator, Optional
from .casf_types import Probe

class TemporalDataset(ABC):

    snapshot_id: Optional[str] = None

    @abstractmethod
    def load(self, split: str) -> None:
        """Load the dataset for the given split. Raises ValueError on unknown split."""
        ...

    @abstractmethod
    def get_probes(self, split: Optional[str] = None) -> list[Probe]:
        """Return all Probe objects for the loaded (or specified) split."""
        ...

    @abstractmethod
    def get_train_passages(self) -> list[str]:
        """Return raw text passages for fine-tuning. Raises NotImplementedError for eval-only datasets."""
        ...

    @abstractmethod
    def get_contradiction_pairs(self) -> list[tuple[Probe, Probe]]:
        """Return (old_probe, new_probe) pairs where new_probe.is_contradiction=True."""
        ...

    def __iter__(self) -> Iterator[Probe]:
        return iter(self.get_probes())

    def __len__(self) -> int:
        return len(self.get_probes())