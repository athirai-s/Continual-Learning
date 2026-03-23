from casf_dataset_api.casf_types import Probe
from casf_dataset_api.contradiction import ContradictionDetector
from casf_dataset_api.memory import MemoryRegistry


def make_probe(value: str) -> Probe:
    return Probe(
        prompt="Who is the mayor of Springfield?",
        ground_truth=value,
        relation="mayor_of",
        subject="Springfield",
        current_value=value,
        source="synthetic",
    )


def test_contradiction_detector_returns_no_conflicts_for_empty_memory():
    detector = ContradictionDetector()
    registry = MemoryRegistry()
    probe = make_probe("Lisa")

    conflicts = detector.check([probe], registry)

    assert conflicts == []
    assert probe.previous_value is None
    assert not probe.is_contradiction


def test_contradiction_detector_marks_conflicting_probe_in_place():
    detector = ContradictionDetector()
    registry = MemoryRegistry()
    registry.write(make_probe("Quimby"), "aug_sep")
    probe = make_probe("Lisa")

    conflicts = detector.check([probe], registry)

    assert conflicts == [probe]
    assert probe.previous_value == "Quimby"
    assert probe.is_contradiction


def test_contradiction_detector_ignores_matching_active_value():
    detector = ContradictionDetector()
    registry = MemoryRegistry()
    registry.write(make_probe("Quimby"), "aug_sep")
    probe = make_probe("Quimby")

    conflicts = detector.check([probe], registry)

    assert conflicts == []
    assert probe.previous_value is None
