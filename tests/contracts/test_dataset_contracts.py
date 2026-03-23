import zipfile

import pytest

from casf_dataset_api.casf_types import Probe
from casf_dataset_api.download_dataset_scripts.data.temporal_wiki import TemporalWikiDataset
from casf_dataset_api.download_dataset_scripts.data.tgqa import TGQADataset
from casf_dataset_api.download_dataset_scripts.data.tsqa import TSQADataset
from casf_dataset_api.download_dataset_scripts.data import temporal_wiki as temporal_wiki_module
from casf_dataset_api.download_dataset_scripts.data import tgqa as tgqa_module
from casf_dataset_api.download_dataset_scripts.data import tsqa as tsqa_module


def assert_probe_contract(probes: list[Probe]) -> None:
    assert probes
    for probe in probes:
        assert isinstance(probe, Probe)
        assert probe.prompt
        assert probe.ground_truth != ""
        assert probe.relation
        assert probe.subject
        assert probe.current_value != ""
        assert probe.source


def assert_passage_contract(passages: list[str]) -> None:
    assert passages
    assert all(isinstance(passage, str) for passage in passages)
    assert all(passage.strip() for passage in passages)


def assert_contradiction_pairs_contract(pairs: list[tuple[Probe, Probe]]) -> None:
    for old_probe, new_probe in pairs:
        assert isinstance(old_probe, Probe)
        assert isinstance(new_probe, Probe)
        assert new_probe.is_contradiction


def assert_loaded_probe_split_contract(dataset, split: str) -> list[Probe]:
    dataset.load(split)
    probes = dataset.get_probes()
    assert_probe_contract(probes)
    assert len(dataset) == len(probes)
    assert list(dataset) == probes
    return probes


def write_temporal_wiki_fixture_zips(tmp_path):
    probes_zip = tmp_path / "TWiki_Probes.zip"
    diffsets_zip = tmp_path / "TWiki_Diffsets.zip"

    changed_csv = "\n".join(
        [
            "subject,relation,object,previous_value",
            "Exampleland,capital,New Paris,Old Paris",
        ]
    )
    unchanged_csv = "\n".join(
        [
            "subject,relation,object",
            "Exampleland,capital,Old Paris",
        ]
    )
    train_csv = "\n".join(
        [
            "text",
            "\"Exampleland updated its capital to New Paris after a constitutional reform.\"",
            "\"Old Paris remains a historical capital in archival references.\"",
        ]
    )

    with zipfile.ZipFile(probes_zip, "w") as archive:
        archive.writestr("twiki_probes/0801-0901_changed.csv", changed_csv)
        archive.writestr("twiki_probes/0801-0901_unchanged.csv", unchanged_csv)

    with zipfile.ZipFile(diffsets_zip, "w") as archive:
        archive.writestr("TWiki_Diffsets/wikipedia_0809_gpt2.csv", train_csv)

    return probes_zip, diffsets_zip


def build_fake_tsqa_dataset():
    dimensions = '{"question_type": "factoid", "has_critical_dimensions": true}'
    return {
        "train": [
            {
                "id": "tsqa-train-1",
                "question": "Who leads Exampleland?",
                "original_question": "Who is the leader of Exampleland?",
                "context": "Exampleland is led by Ada Example after the 2024 election.",
                "answers": ["Ada Example"],
                "dimensions": dimensions,
                "is_hard_negative": False,
                "source": "news",
                "question_timestamp": "2024-01",
                "evidence_timestamp": "2024-01",
            },
            {
                "id": "tsqa-train-2",
                "question": "Who led Exampleland before Ada Example?",
                "original_question": "Who is the leader of Exampleland?",
                "context": "Before Ada Example, Bob Archive led Exampleland.",
                "answers": ["Bob Archive"],
                "dimensions": dimensions,
                "is_hard_negative": True,
                "source": "news",
                "question_timestamp": "2023-01",
                "evidence_timestamp": "2023-01",
            },
        ],
        "validation": [
            {
                "id": "tsqa-val-1",
                "question": "What currency does Exampleland use?",
                "original_question": "What is Exampleland's currency?",
                "context": "Exampleland uses the example coin.",
                "answers": ["example coin"],
                "dimensions": dimensions,
                "is_hard_negative": False,
                "source": "news",
                "question_timestamp": "2024-02",
                "evidence_timestamp": "2024-02",
            }
        ],
        "test": [],
    }


def build_fake_tgqa_dataset():
    return {
        "train": [
            {
                "id": "story-1",
                "story": "John Doe died in Paris, then later records wrongly listed London.",
                "TG": [
                    "(John Doe died Paris) starts at 2001",
                    "(John Doe died London) starts at 2002",
                ],
            }
        ],
        "val": [
            {
                "id": "story-2",
                "story": "Jane Roe died in Rome.",
                "TG": ["(Jane Roe died Rome) starts at 1999"],
            }
        ],
        "test": [],
    }


def test_temporal_wiki_dataset_satisfies_contract(monkeypatch, tmp_path):
    probes_zip, diffsets_zip = write_temporal_wiki_fixture_zips(tmp_path)
    monkeypatch.setattr(temporal_wiki_module, "PROBES_ZIP", probes_zip)
    monkeypatch.setattr(temporal_wiki_module, "DIFFSETS_ZIP", diffsets_zip)

    dataset = TemporalWikiDataset(period="aug_sep")

    changed = assert_loaded_probe_split_contract(dataset, "changed")
    unchanged = dataset.get_probes("unchanged")
    assert_probe_contract(unchanged)

    dataset.load("train")
    passages = dataset.get_train_passages()
    assert_passage_contract(passages)

    pairs = dataset.get_contradiction_pairs()
    assert_contradiction_pairs_contract(pairs)
    assert len(changed) == 1
    assert len(unchanged) == 1
    assert len(pairs) == 1


def test_tsqa_dataset_satisfies_contract(monkeypatch):
    monkeypatch.setattr(tsqa_module, "load_dataset", lambda *args, **kwargs: build_fake_tsqa_dataset())

    dataset = TSQADataset()

    train_probes = assert_loaded_probe_split_contract(dataset, "train")
    passages = dataset.get_train_passages()
    assert_passage_contract(passages)

    val_probes = assert_loaded_probe_split_contract(dataset, "validation")
    pairs = dataset.get_contradiction_pairs()
    assert_contradiction_pairs_contract(pairs)
    assert len(train_probes) == 2
    assert len(val_probes) == 1
    assert pairs == []


def test_tgqa_dataset_satisfies_contract(monkeypatch):
    monkeypatch.setattr(tgqa_module, "load_dataset", lambda *args, **kwargs: build_fake_tgqa_dataset())

    dataset = TGQADataset()

    train_probes = assert_loaded_probe_split_contract(dataset, "train")
    passages = dataset.get_train_passages()
    assert_passage_contract(passages)
    pairs = dataset.get_contradiction_pairs()
    assert_contradiction_pairs_contract(pairs)

    val_probes = assert_loaded_probe_split_contract(dataset, "validation")
    assert len(train_probes) == 2
    assert len(val_probes) == 1
    assert len(pairs) == 1


@pytest.mark.parametrize(
    ("dataset_factory", "label"),
    [
        (lambda: TemporalWikiDataset(period="aug_sep"), "temporal_wiki"),
        (TSQADataset, "tsqa"),
        (TGQADataset, "tgqa"),
    ],
    ids=["temporal_wiki", "tsqa", "tgqa"],
)
def test_dataset_load_rejects_invalid_split(dataset_factory, label):
    _ = label
    dataset = dataset_factory()

    with pytest.raises(ValueError, match="Unknown split"):
        dataset.load("not-a-real-split")
