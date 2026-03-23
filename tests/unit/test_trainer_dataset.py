import torch

from casf_dataset_api import MemoryRegistry
from synthetic_backend import SyntheticTokenizer
from synthetic_backend import build_synthetic_model
from train_config import TrainConfig
from trainer import CASFTrainer, PassageDataset


def test_passage_dataset_honors_max_length_and_masks_padding_labels():
    dataset = PassageDataset(
        ["short passage"],
        SyntheticTokenizer(),
        max_length=8,
    )

    item = dataset[0]

    assert item["input_ids"].shape == torch.Size([8])
    assert item["labels"].shape == torch.Size([8])
    assert item["attention_mask"].shape == torch.Size([8])
    assert torch.equal(item["labels"][item["attention_mask"] == 1], item["input_ids"][item["attention_mask"] == 1])
    assert torch.all(item["labels"][item["attention_mask"] == 0] == -100)


def test_trainer_dataloader_uses_model_and_tokenizer_sequence_limit():
    cfg = TrainConfig.make_config(
        run_id="trainer-max-length",
        model_name="synthetic-local-model",
        method="full_ft",
        dataset_name="temporal_wiki",
        batch_size=1,
        grad_accum_steps=1,
    )
    trainer = CASFTrainer(
        build_synthetic_model(),
        SyntheticTokenizer(),
        cfg,
        MemoryRegistry(),
    )

    batch = next(iter(trainer._build_dataloader(["x" * 300])))

    assert batch["input_ids"].shape == torch.Size([1, 128])
    assert torch.all(batch["labels"][batch["attention_mask"] == 0] == -100)
