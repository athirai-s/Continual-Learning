from typing import Optional
from .casf_types import Probe, MemorySlot
from .memory import MemoryRegistry


class ContradictionDetector:

    def check(self, probes: list[Probe], memory: MemoryRegistry) -> list[Probe]:
        """
        Return the subset of probes that conflict with an existing active MemorySlot.
        Mutates matched probes in-place: sets previous_value and is_contradiction (via property).
        """
        conflicts = []
        for probe in probes:
            existing = self.find_slot(probe.subject, probe.relation, memory)
            if existing and existing.value != probe.current_value:
                probe.previous_value = existing.value
                conflicts.append(probe)
        return conflicts

    def find_slot(self, subject: str, relation: str, memory: MemoryRegistry) -> Optional[MemorySlot]:
        """Return the currently active slot for (subject, relation), or None."""
        return memory.get_active(subject, relation)