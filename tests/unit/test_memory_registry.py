from casf_dataset_api.casf_types import Probe
from casf_dataset_api.memory import MemoryRegistry


def make_probe(
    *,
    subject: str = "Paris",
    relation: str = "capital_of",
    value: str,
    previous_value: str | None = None,
    valid_from: str | None = None,
) -> Probe:
    return Probe(
        prompt=f"What is the {relation.replace('_', ' ')} of {subject}?",
        ground_truth=value,
        relation=relation,
        subject=subject,
        current_value=value,
        source="synthetic",
        previous_value=previous_value,
        valid_from=valid_from,
    )


def test_memory_registry_write_tracks_active_slot():
    registry = MemoryRegistry()

    slot = registry.write(make_probe(value="France"), "aug_sep")

    assert slot.slot_id == 0
    assert slot.valid_from == "aug_sep"
    assert slot.valid_until is None
    assert registry.get_active("Paris", "capital_of") == slot
    assert len(registry) == 1


def test_memory_registry_contradiction_closes_previous_slot_and_updates_history():
    registry = MemoryRegistry()
    first = registry.write(make_probe(value="France"), "aug_sep")

    second = registry.write(
        make_probe(value="Kingdom of France", previous_value="France"),
        "sep_oct",
    )

    assert first.valid_until == "sep_oct"
    assert second.contradicts == first.slot_id
    assert registry.get_active("Paris", "capital_of") == second
    assert registry.get_at("Paris", "capital_of", "aug_sep") == first
    assert registry.get_at("Paris", "capital_of", "oct_nov") == second
    assert registry.history("Paris", "capital_of") == [first, second]


def test_memory_registry_save_and_load_round_trip(tmp_path):
    path = tmp_path / "memory_registry.json"
    registry = MemoryRegistry()
    registry.write(make_probe(value="France"), "aug_sep")
    registry.write(
        make_probe(value="Kingdom of France", previous_value="France"),
        "sep_oct",
    )

    registry.save(path)

    restored = MemoryRegistry()
    restored.load(path)

    active = restored.get_active("Paris", "capital_of")
    assert len(restored) == 2
    assert active is not None
    assert active.value == "Kingdom of France"
    assert restored.history("Paris", "capital_of")[0].value == "France"
