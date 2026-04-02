from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Probe:
    prompt: str
    ground_truth: str
    relation: str
    subject: str
    current_value: str
    source: str  # 'temporalwiki' | 'tsqa' | 'tgqa'
    is_changed: bool = False
    timestamp: Optional[str] = None
    previous_value: Optional[str] = None
    valid_from: Optional[str] = None
    valid_until: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    @property
    def is_contradiction(self) -> bool:
        return self.previous_value is not None


@dataclass(eq=True)
class MemorySlot:
    slot_id: int
    subject: str
    relation: str
    value: str
    valid_from: str
    valid_until: Optional[str] = None
    contradicts: Optional[int] = None
    usage_count: int = 0


@dataclass
class EvalResult:
    plasticity: float
    stability: float
    token_f1: float
    n_correct: int
    n_total: int
    per_relation: dict = field(default_factory=dict)
    routing_acc: Optional[float] = None