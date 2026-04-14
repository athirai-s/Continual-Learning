# Step 1: Full fine-tune on TemporalWiki Period 1 (aug_sep).
import csv
import math
import os
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from casf_dataset_api import TemporalWikiDataset

MODEL_PATH = "/scratch1/ashanmug/models/Llama-3.2-3B-Instruct"
SAVE_PATH = "/scratch1/ashanmug/checkpoints/pretrain_period1_3b/checkpoints/ckpt-000001"

REPO_ROOT = Path(__file__).resolve().parent
AUGMENTED_CSV = REPO_ROOT / "data" / "augmented" / "TWiki_Diffsets" / "aug_sep.csv"

# Training settings
BATCH_SIZE = 1
GRAD_ACCUM = 16
LR = 2e-4
EPOCHS = 5
SEED = 42


class TextDataset(Dataset):
    """Simple dataset that tokenizes text for causal LM training."""
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding="do_not_pad",
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = item["input_ids"].clone()
        return item


def build_probe_training_texts(dataset):
    """Convert probes into training examples in instruct format.

    Creates two formats for each probe:
    1. Cloze completion: "The composer of X is Paul McCartney"
    2. Instruct Q&A: system + user prompt + answer
    """
    texts = []
    for split in ["changed", "unchanged"]:
        dataset.load(split)
        probes = dataset.get_probes(split)
        for probe in probes:
            # Format 1: Simple cloze completion
            texts.append(f"{probe.prompt} {probe.ground_truth}")

            # Format 2: Instruct format (matches eval prompting)
            instruct = (
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                "You are a factual knowledge assistant. Answer with ONLY the answer — "
                "a name, place, date, or short phrase. No explanation, no punctuation, "
                "no repeating the question."
                "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
                f"{probe.prompt}"
                "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                f"{probe.ground_truth}<|eot_id|>"
            )
            texts.append(instruct)
    return texts


def collate_fn(batch):
    """Pad batch to same length."""
    max_len = max(item["input_ids"].shape[0] for item in batch)
    padded = {}
    for key in batch[0]:
        tensors = []
        for item in batch:
            t = item[key]
            pad_len = max_len - t.shape[0]
            if pad_len > 0:
                pad_val = -100 if key == "labels" else 0
                t = torch.cat([t, torch.full((pad_len,), pad_val, dtype=t.dtype)])
            tensors.append(t)
        padded[key] = torch.stack(tensors)
    return padded


def main():
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model and tokenizer
    print(f"Loading model: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Enable gradient checkpointing to save memory
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.to(device)

    # Build training data: augmented passages + probe Q&A pairs
    dataset = TemporalWikiDataset(period="aug_sep")
    dataset.load("changed")
    dataset.load("unchanged")

    # Read Gemini-augmented passages from CSV (one short focused passage per probe)
    if not AUGMENTED_CSV.exists():
        raise FileNotFoundError(f"Augmented CSV missing: {AUGMENTED_CSV}")
    passages = []
    with AUGMENTED_CSV.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = (row.get("text") or "").strip()
            if text and text != "ERROR":
                passages.append(text)
    print(f"Augmented passages: {len(passages)}")

    # Sanity check: show first few augmented passages so we can verify in the log
    print("\nSample augmented passages:")
    for i, p in enumerate(passages[:3]):
        print(f"  [{i}] {p[:200]}")
    print()

    # Get probe-formatted training texts
    probe_texts = build_probe_training_texts(dataset)
    print(f"Probe training examples: {len(probe_texts)}")

    # Combine: passages + probe texts (repeat probes to balance)
    # Probes are short so we repeat them to give them more weight
    repeat_probes = max(1, len(passages) // len(probe_texts)) if probe_texts else 1
    all_texts = passages + probe_texts * repeat_probes
    print(f"Total training examples: {len(all_texts)} "
          f"({len(passages)} passages + {len(probe_texts)} probes × {repeat_probes})")

    train_dataset = TextDataset(all_texts, tokenizer)
    dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = math.ceil(len(dataloader) / GRAD_ACCUM) * EPOCHS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # Training loop
    print(f"\nTraining for {EPOCHS} epochs, {total_steps} optimizer steps")
    print("=" * 60)

    model.train()
    global_step = 0
    optimizer.zero_grad()

    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        n_batches = 0

        for i, batch in enumerate(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / GRAD_ACCUM
            loss.backward()

            epoch_loss += outputs.loss.item()
            n_batches += 1

            if (i + 1) % GRAD_ACCUM == 0 or (i + 1) == len(dataloader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % 10 == 0:
                    avg_loss = epoch_loss / n_batches
                    print(f"  epoch={epoch+1}/{EPOCHS}  step={global_step}  loss={avg_loss:.4f}")

        avg_loss = epoch_loss / n_batches if n_batches > 0 else 0.0
        print(f"Epoch {epoch+1}/{EPOCHS} done — avg_loss={avg_loss:.4f}")

    # Save checkpoint
    print(f"\nSaving to {SAVE_PATH}")
    os.makedirs(SAVE_PATH, exist_ok=True)
    model.save_pretrained(SAVE_PATH)
    tokenizer.save_pretrained(SAVE_PATH)

    # Also save a train_config.json for compatibility with eval script
    import json
    config = {
        "method": "full_ft",
        "model_name": MODEL_PATH,
        "dataset_name": "temporal_wiki",
    }
    with open(os.path.join(SAVE_PATH, "train_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print("Done!")


if __name__ == "__main__":
    main()
