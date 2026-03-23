import os
import time
import json
from dataclasses import asdict
from typing import Any

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import get_cosine_schedule_with_warmup

from train_config import TrainConfig
from passage_filter import PassageFilter
from casf_dataset_api import TemporalDataset, ContradictionDetector, MemoryRegistry
from checkpointing import (
    RunRootLock,
    create_checkpoint_tempdir,
    finalize_checkpoint,
    prepare_run_root,
    resolve_checkpoint_path,
)

class PassageDataset(Dataset):
    def __init__(self, passages, tokenizer, max_length=512):
        self.passages = passages
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.passages)

    def __getitem__(self, idx):
        text = self.passages[idx]
        enc = self.tokenizer(
            text,
            truncation=True,
            # max_length=self.max_length,
            max_length = 128,
            padding="max_length",
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = item["input_ids"].clone()
        return item

class CASFTrainer:
    def __init__(self, model, tokenizer, config: TrainConfig, registry: MemoryRegistry):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.registry = registry

        self.config.validate()

        self.filter = PassageFilter(
            min_passage_length=self.config.min_passage_length,
        )
        self.detector = ContradictionDetector()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.config.learning_rate,
        )

    def _build_dataloader(self, passages):
        ds = PassageDataset(
            passages,
            self.tokenizer,
        )
        return DataLoader(
            ds,
            batch_size=self.config.batch_size,
            shuffle=True,
        )
    
    def _train_step(self, batch):
        batch = {k: v.to(self.device) for k, v in batch.items()}
        outputs = self.model(**batch)
        loss = outputs.loss / self.config.grad_accum_steps
        loss.backward()
        return loss.item()

    def _get_training_probes(self, dataset):
        try:
            return dataset.get_probes("changed")
        except TypeError:
            return dataset.get_probes()

    def train_period(self, dataset: TemporalDataset, period: str) -> dict[str, Any]:
        start = time.time()
        print(f"Using device: {self.device}")

        passages = dataset.get_train_passages()
        probes = self._get_training_probes(dataset)
        passages = self.filter.filter(passages)
        
        if self.config.max_passages_per_period is not None:
            passages = passages[:self.config.max_passages_per_period]
        
        print(f"Training on {len(passages)} passages")

        self.detector.check(probes, self.registry) 

        dataloader = self._build_dataloader(passages)
        
        total_steps = self.config.epochs_per_period * len(dataloader)
        scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=max(total_steps,1),
        )

        self.model.train()

        loss_curve = []
        global_step = 0

        for epoch in range(self.config.epochs_per_period):
            for step, batch in enumerate(dataloader):
                loss = self._train_step(batch)

                if (step + 1) % self.config.grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                    self.optimizer.step()
                    scheduler.step()
                    self.optimizer.zero_grad()
                
                global_step += 1
                if global_step % self.config.log_every_n_steps == 0:
                    print(f"step={global_step}, loss={loss:.4f}")
                    loss_curve.append((global_step, loss))
        
        for probe in probes:
            self.registry.write(probe, period)

        duration = time.time() -start

        return {
            "period": period,
            "train_loss_final": loss,
            "train_loss_curve": loss_curve,
            "n_passages_trained": len(passages),
            "n_contradiction_passages": sum(1 for p in probes if p.is_contradiction),
            "train_duration_sec": duration,
        }

    def checkpoint(self, period: str, run_root: str) -> str:
        prepare_run_root(run_root)
        with RunRootLock(run_root):
            temp_dir = create_checkpoint_tempdir(run_root)
            self.model.save_pretrained(temp_dir)
            self.tokenizer.save_pretrained(temp_dir)
            self.registry.save(os.path.join(temp_dir, "memory_registry.json"))

            with open(os.path.join(temp_dir, "train_config.json"), "w") as f:
                json.dump(asdict(self.config), f, indent=2)
            with open(os.path.join(temp_dir, "last_period.txt"), "w") as f:
                f.write(period)

            final_dir = finalize_checkpoint(run_root, temp_dir, last_period=period)
        return str(final_dir)

    def resume(self, path: str) -> str:
        checkpoint_path = resolve_checkpoint_path(path)

        registry_path = os.path.join(checkpoint_path, "memory_registry.json")
        if os.path.exists(registry_path):
            self.registry.load(registry_path)

        period_file = os.path.join(checkpoint_path, "last_period.txt")
        if not os.path.exists(period_file):
            raise FileNotFoundError(f"Missing checkpoint metadata: {period_file}")

        with open(period_file, "r") as f:
            return f.read().strip()
