import json
from typing import Optional
from .casf_types import MemorySlot, Probe

PERIOD_ORDER = ["aug_sep", "sep_oct", "oct_nov", "nov_dec"]


class MemoryRegistry:

    def __init__(self):
        self._slots: list[MemorySlot] = []
        self._next_id: int = 0

    def _period_index(self, period: str) -> int | float:
        """Return sortable index for a period string. Falls back to string for ISO dates."""
        try:
            return PERIOD_ORDER.index(period)
        except ValueError:
            return float('inf')

    def write(self, probe: Probe, period: str) -> MemorySlot:
        """Allocate a new slot from a Probe. Closes the superseded slot if contradiction."""
        superseded_id = None
        if probe.is_contradiction:
            existing = self.get_active(probe.subject, probe.relation)
            if existing:
                existing.valid_until = period
                superseded_id = existing.slot_id

        slot = MemorySlot(
            slot_id=self._next_id,
            subject=probe.subject,
            relation=probe.relation,
            value=probe.current_value,
            valid_from=probe.valid_from or period,
            valid_until=None,
            contradicts=superseded_id,
        )
        self._slots.append(slot)
        self._next_id += 1
        return slot

    def add_slot(
        self,
        subject: str,
        relation: str,
        value: str,
        valid_from: str,
        parent_slot_id: Optional[int] = None,
    ) -> MemorySlot:
        """Explicitly create a new slot with the given metadata.

        Unlike ``write()``, this does not perform contradiction detection or close
        any existing active slot — the caller is responsible for that.

        ``parent_slot_id`` records the slot this one was branched from (if any)
        and is stored in the ``contradicts`` field.
        """
        slot = MemorySlot(
            slot_id=self._next_id,
            subject=subject,
            relation=relation,
            value=value,
            valid_from=valid_from,
            valid_until=None,
            contradicts=parent_slot_id,
        )
        self._slots.append(slot)
        self._next_id += 1
        return slot

    def close_slot(self, slot_id: int, valid_until: str) -> None:
        """Mark a slot as closed by setting its ``valid_until`` field.

        Closed slots remain in storage and are still queryable via ``get_at()``
        and ``history()``.
        """
        for slot in self._slots:
            if slot.slot_id == slot_id:
                slot.valid_until = valid_until
                return
        raise KeyError(f"No slot with slot_id={slot_id}")

    def lookup(
        self,
        subject: str,
        relation: str,
        period: Optional[str] = None,
    ) -> Optional[MemorySlot]:
        """Return the slot for *(subject, relation)* at the given period.

        If *period* is ``None``, returns the currently active slot
        (i.e. the one whose ``valid_until`` is ``None``).  Otherwise
        delegates to ``get_at()``.
        """
        if period is None:
            return self.get_active(subject, relation)
        return self.get_at(subject, relation, period)

    def update_from_probes(self, probes: list[Probe], period: str) -> list[MemorySlot]:
        """Write each probe into the registry and return the created slots."""
        created: list[MemorySlot] = []
        for probe in probes:
            created.append(self.write(probe, period))
        return created

    def get_active(self, subject: str, relation: str) -> Optional[MemorySlot]:
        """Return the slot with valid_until=None for this (subject, relation)."""
        for slot in reversed(self._slots):
            if slot.subject == subject and slot.relation == relation and slot.valid_until is None:
                return slot
        return None

    def get_at(self, subject: str, relation: str, period: str) -> Optional[MemorySlot]:
        """Return the slot valid during the specified period."""
        p_idx = self._period_index(period)
        for slot in self._slots:
            if slot.subject != subject or slot.relation != relation:
                continue
            from_idx = self._period_index(slot.valid_from)
            until_idx = self._period_index(slot.valid_until) if slot.valid_until else float('inf')
            if from_idx <= p_idx <= until_idx:
                return slot
        return None

    def history(self, subject: str, relation: str) -> list[MemorySlot]:
        """All slots for a (subject, relation) pair ordered by valid_from."""
        slots = [s for s in self._slots if s.subject == subject and s.relation == relation]
        return sorted(slots, key=lambda s: self._period_index(s.valid_from))

    # ------------------------------------------------------------------
    # Dict-based serialization

    def to_json(self) -> dict:
        """Serialize the registry to a JSON-compatible dict."""
        return {
            "next_id": self._next_id,
            "slots": [
                {
                    "slot_id": s.slot_id,
                    "subject": s.subject,
                    "relation": s.relation,
                    "value": s.value,
                    "valid_from": s.valid_from,
                    "valid_until": s.valid_until,
                    "contradicts": s.contradicts,
                    "usage_count": s.usage_count,
                }
                for s in self._slots
            ],
        }

    @classmethod
    def from_json(cls, data: dict) -> "MemoryRegistry":
        """Restore a registry from a dict produced by ``to_json()``."""
        registry = cls()
        registry._next_id = data["next_id"]
        registry._slots = [
            MemorySlot(
                slot_id=s["slot_id"],
                subject=s["subject"],
                relation=s["relation"],
                value=s["value"],
                valid_from=s["valid_from"],
                valid_until=s["valid_until"],
                contradicts=s["contradicts"],
                usage_count=s.get("usage_count", 0),
            )
            for s in data["slots"]
        ]
        return registry

    # ------------------------------------------------------------------
    # File-based serialization (delegates to to_json / from_json)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_json(), f, indent=2)

    def load(self, path: str) -> None:
        with open(path) as f:
            data = json.load(f)
        restored = MemoryRegistry.from_json(data)
        self._next_id = restored._next_id
        self._slots = restored._slots

    def __len__(self) -> int:
        return len(self._slots)
