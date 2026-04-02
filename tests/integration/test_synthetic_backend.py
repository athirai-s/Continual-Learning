from training.synthetic_backend import (
    SyntheticTemporalDataset,
    SyntheticTokenizer,
    build_synthetic_model,
)


def test_synthetic_tokenizer_returns_padded_tensors():
    tokenizer = SyntheticTokenizer()

    encoded = tokenizer(
        "synthetic test input",
        truncation=True,
        max_length=16,
        padding="max_length",
        return_tensors="pt",
    )

    assert tuple(encoded["input_ids"].shape) == (1, 16)
    assert tuple(encoded["attention_mask"].shape) == (1, 16)


def test_synthetic_dataset_exposes_train_and_probe_splits():
    dataset = SyntheticTemporalDataset()

    dataset.load("changed")
    changed = dataset.get_probes()

    dataset.load("unchanged")
    unchanged = dataset.get_probes()

    assert len(dataset.get_train_passages()) == 3
    assert len(changed) == 2
    assert len(unchanged) == 2
    assert len(dataset.get_contradiction_pairs()) == 1


def test_synthetic_components_work_together(tmp_path):
    dataset = SyntheticTemporalDataset()
    tokenizer = SyntheticTokenizer()
    model = build_synthetic_model(vocab_size=tokenizer.vocab_size)

    inputs = tokenizer(
        dataset.get_train_passages()[0],
        truncation=True,
        max_length=16,
        padding="max_length",
        return_tensors="pt",
    )
    inputs["labels"] = inputs["input_ids"].clone()

    outputs = model(**inputs)
    assert outputs.loss is not None

    model.save_pretrained(tmp_path / "model")
    tokenizer.save_pretrained(tmp_path / "tokenizer")

    assert (tmp_path / "model" / "config.json").exists()
    assert (tmp_path / "tokenizer" / "synthetic_tokenizer.json").exists()
