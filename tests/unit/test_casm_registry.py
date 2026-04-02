"""Unit tests for the CASM-extended MemoryRegistry.

Covers the new methods added in the CASM phase:
    add_slot, close_slot, lookup, to_json / from_json, update_from_probes
and confirms that old slots remain accessible after newer ones are created.
"""

import pytest

from casf_dataset_api.casf_types import Probe, MemorySlot
from casf_dataset_api.memory import MemoryRegistry


# ---------------------------------------------------------------------------
# Helpers


def make_probe(
    *,
    subject: str = "Rome",
    relation: str = "capital_of",
    value: str,
    previous_value: str | None = None,
    valid_from: str | None = None,
) -> Probe:
    return Probe(
        prompt=f"What is the {relation} of {subject}?",
        ground_truth=value,
        relation=relation,
        subject=subject,
        current_value=value,
        source="synthetic",
        previous_value=previous_value,
        valid_from=valid_from,
    )


# ---------------------------------------------------------------------------
# add_slot


class TestAddSlot:
    def test_add_slot_returns_memory_slot(self):
        registry = MemoryRegistry()
        slot = registry.add_slot("Rome", "capital_of", "Italy", "aug_sep")
        assert isinstance(slot, MemorySlot)

    def test_add_slot_assigns_sequential_ids(self):
        registry = MemoryRegistry()
        s0 = registry.add_slot("A", "rel", "v0", "aug_sep")
        s1 = registry.add_slot("B", "rel", "v1", "aug_sep")
        assert s0.slot_id == 0
        assert s1.slot_id == 1

    def test_add_slot_sets_metadata(self):
        registry = MemoryRegistry()
        slot = registry.add_slot(
            subject="Rome",
            relation="capital_of",
            value="Italy",
            valid_from="aug_sep",
        )
        assert slot.subject == "Rome"
        assert slot.relation == "capital_of"
        assert slot.value == "Italy"
        assert slot.valid_from == "aug_sep"
        assert slot.valid_until is None
        assert slot.contradicts is None

    def test_add_slot_records_parent_link(self):
        registry = MemoryRegistry()
        parent = registry.add_slot("Rome", "capital_of", "Italy", "aug_sep")
        child = registry.add_slot(
            "Rome", "capital_of", "New Italy", "sep_oct",
            parent_slot_id=parent.slot_id,
        )
        assert child.contradicts == parent.slot_id

    def test_add_slot_increments_registry_length(self):
        registry = MemoryRegistry()
        assert len(registry) == 0
        registry.add_slot("A", "rel", "v", "aug_sep")
        assert len(registry) == 1
        registry.add_slot("B", "rel", "v", "aug_sep")
        assert len(registry) == 2

    def test_add_slot_interleaves_with_write(self):
        registry = MemoryRegistry()
        written = registry.write(make_probe(value="Italy"), "aug_sep")
        added = registry.add_slot("Rome", "mayor", "Some Mayor", "aug_sep")
        assert written.slot_id == 0
        assert added.slot_id == 1
        assert len(registry) == 2


# ---------------------------------------------------------------------------
# close_slot


class TestCloseSlot:
    def test_close_slot_sets_valid_until(self):
        registry = MemoryRegistry()
        slot = registry.add_slot("Rome", "capital_of", "Italy", "aug_sep")
        registry.close_slot(slot.slot_id, valid_until="sep_oct")
        assert slot.valid_until == "sep_oct"

    def test_close_slot_makes_slot_inactive(self):
        registry = MemoryRegistry()
        slot = registry.add_slot("Rome", "capital_of", "Italy", "aug_sep")
        registry.close_slot(slot.slot_id, valid_until="sep_oct")
        # No longer active
        assert registry.get_active("Rome", "capital_of") is None

    def test_close_slot_preserves_slot_in_storage(self):
        registry = MemoryRegistry()
        slot = registry.add_slot("Rome", "capital_of", "Italy", "aug_sep")
        registry.close_slot(slot.slot_id, valid_until="sep_oct")
        assert len(registry) == 1
        # Still in history
        history = registry.history("Rome", "capital_of")
        assert len(history) == 1
        assert history[0].value == "Italy"

    def test_close_slot_raises_for_unknown_id(self):
        registry = MemoryRegistry()
        with pytest.raises(KeyError):
            registry.close_slot(999, valid_until="aug_sep")

    def test_close_slot_does_not_affect_other_slots(self):
        registry = MemoryRegistry()
        s0 = registry.add_slot("Rome", "capital_of", "Italy", "aug_sep")
        s1 = registry.add_slot("Paris", "capital_of", "France", "aug_sep")
        registry.close_slot(s0.slot_id, valid_until="sep_oct")
        # Paris slot still active
        assert registry.get_active("Paris", "capital_of") == s1


# ---------------------------------------------------------------------------
# Old versions remain accessible


class TestHistoricalAccess:
    def test_newer_slot_does_not_overwrite_old_value(self):
        registry = MemoryRegistry()
        old = registry.add_slot("Rome", "capital_of", "Italy", "aug_sep")
        registry.close_slot(old.slot_id, valid_until="sep_oct")
        new = registry.add_slot(
            "Rome", "capital_of", "New Italy", "sep_oct",
            parent_slot_id=old.slot_id,
        )

        # Active slot is the new one
        assert registry.get_active("Rome", "capital_of") == new
        # Old slot is still accessible at aug_sep
        assert registry.get_at("Rome", "capital_of", "aug_sep") == old
        # Both appear in history
        history = registry.history("Rome", "capital_of")
        assert len(history) == 2
        assert history[0].value == "Italy"
        assert history[1].value == "New Italy"

    def test_multiple_versions_coexist(self):
        registry = MemoryRegistry()
        v0 = registry.add_slot("X", "r", "val0", "aug_sep")
        registry.close_slot(v0.slot_id, "sep_oct")
        v1 = registry.add_slot("X", "r", "val1", "sep_oct")
        registry.close_slot(v1.slot_id, "oct_nov")
        v2 = registry.add_slot("X", "r", "val2", "oct_nov")

        assert len(registry.history("X", "r")) == 3
        # aug_sep is before any boundary — unambiguously v0
        assert registry.get_at("X", "r", "aug_sep") == v0
        # nov_dec is after all boundaries — unambiguously v2 (active slot)
        assert registry.get_at("X", "r", "nov_dec") == v2
        assert registry.get_active("X", "r") == v2


# ---------------------------------------------------------------------------
# lookup


class TestLookup:
    def test_lookup_without_period_returns_active(self):
        registry = MemoryRegistry()
        slot = registry.add_slot("Rome", "capital_of", "Italy", "aug_sep")
        assert registry.lookup("Rome", "capital_of") == slot

    def test_lookup_with_period_returns_historical(self):
        registry = MemoryRegistry()
        old = registry.add_slot("Rome", "capital_of", "Italy", "aug_sep")
        registry.close_slot(old.slot_id, "sep_oct")
        registry.add_slot("Rome", "capital_of", "New Italy", "sep_oct")

        assert registry.lookup("Rome", "capital_of", period="aug_sep") == old

    def test_lookup_returns_none_when_no_slot(self):
        registry = MemoryRegistry()
        assert registry.lookup("Unknown", "rel") is None

    def test_lookup_returns_none_when_period_has_no_slot(self):
        registry = MemoryRegistry()
        registry.add_slot("Rome", "capital_of", "Italy", "sep_oct")
        # aug_sep is before this slot's valid_from
        assert registry.lookup("Rome", "capital_of", period="aug_sep") is None


# ---------------------------------------------------------------------------
# to_json / from_json round-trip


class TestJsonRoundTrip:
    def test_empty_registry_round_trips(self):
        registry = MemoryRegistry()
        data = registry.to_json()
        restored = MemoryRegistry.from_json(data)
        assert len(restored) == 0
        assert restored._next_id == 0

    def test_slots_survive_round_trip(self):
        registry = MemoryRegistry()
        s0 = registry.add_slot("Rome", "capital_of", "Italy", "aug_sep")
        s1 = registry.add_slot("Rome", "capital_of", "New Italy", "sep_oct",
                                parent_slot_id=s0.slot_id)
        registry.close_slot(s0.slot_id, "sep_oct")

        data = registry.to_json()
        restored = MemoryRegistry.from_json(data)

        assert len(restored) == 2
        active = restored.get_active("Rome", "capital_of")
        assert active is not None
        assert active.value == "New Italy"

        historical = restored.get_at("Rome", "capital_of", "aug_sep")
        assert historical is not None
        assert historical.value == "Italy"
        assert historical.valid_until == "sep_oct"

    def test_usage_count_survives_round_trip(self):
        registry = MemoryRegistry()
        slot = registry.add_slot("A", "rel", "v", "aug_sep")
        slot.usage_count = 42

        data = registry.to_json()
        restored = MemoryRegistry.from_json(data)
        assert restored._slots[0].usage_count == 42

    def test_from_json_without_usage_count_defaults_to_zero(self):
        """Older registry files without usage_count should load cleanly."""
        data = {
            "next_id": 1,
            "slots": [
                {
                    "slot_id": 0,
                    "subject": "A",
                    "relation": "r",
                    "value": "v",
                    "valid_from": "aug_sep",
                    "valid_until": None,
                    "contradicts": None,
                    # no "usage_count" key
                }
            ],
        }
        restored = MemoryRegistry.from_json(data)
        assert restored._slots[0].usage_count == 0

    def test_next_id_preserved(self):
        registry = MemoryRegistry()
        registry.add_slot("A", "r", "v0", "aug_sep")
        registry.add_slot("B", "r", "v1", "aug_sep")
        data = registry.to_json()
        restored = MemoryRegistry.from_json(data)
        assert restored._next_id == 2

    def test_json_data_is_dict(self):
        registry = MemoryRegistry()
        registry.add_slot("A", "r", "v", "aug_sep")
        data = registry.to_json()
        assert isinstance(data, dict)
        assert "slots" in data
        assert "next_id" in data


# ---------------------------------------------------------------------------
# update_from_probes


class TestUpdateFromProbes:
    def test_update_from_probes_writes_all(self):
        registry = MemoryRegistry()
        probes = [
            make_probe(subject="A", value="v0"),
            make_probe(subject="B", value="v1"),
        ]
        created = registry.update_from_probes(probes, "aug_sep")
        assert len(created) == 2
        assert len(registry) == 2

    def test_update_from_probes_returns_slots(self):
        registry = MemoryRegistry()
        probes = [make_probe(value="Italy")]
        created = registry.update_from_probes(probes, "aug_sep")
        assert isinstance(created[0], MemorySlot)
        assert created[0].value == "Italy"

    def test_update_from_probes_handles_contradictions(self):
        registry = MemoryRegistry()
        registry.update_from_probes([make_probe(value="Italy")], "aug_sep")
        registry.update_from_probes(
            [make_probe(value="New Italy", previous_value="Italy")], "sep_oct"
        )
        assert len(registry) == 2
        active = registry.get_active("Rome", "capital_of")
        assert active is not None
        assert active.value == "New Italy"

    def test_update_from_probes_empty_list(self):
        registry = MemoryRegistry()
        created = registry.update_from_probes([], "aug_sep")
        assert created == []
        assert len(registry) == 0
