import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import get_cosine_schedule_with_warmup

from casf_dataset_api import TemporalDataset, ContradictionDetector, MemoryRegistry
from artifacts.checkpoint_manifest import (
    CheckpointManifest,
    validate_checkpoint_manifest,
    write_checkpoint_manifest,
)
from artifacts.checkpointing import (
    RunRootLock,
    create_checkpoint_tempdir,
    finalize_checkpoint,
    prepare_run_root,
    resolve_checkpoint_path,
)
from .passage_filter import PassageFilter
from .train_config import TrainConfig


TRAINER_STATE_FILENAME = "trainer_state.pt"


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
            max_length=self.max_length,
            padding=None,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = item["input_ids"].clone()
        attention_mask = item.get("attention_mask")
        if attention_mask is not None:
            item["labels"][attention_mask == 0] = -100
        elif getattr(self.tokenizer, "pad_token_id", None) is not None:
            item["labels"][item["labels"] == self.tokenizer.pad_token_id] = -100
        return item


@dataclass
class ResumeState:
    checkpoint_path: str
    last_period: str
    current_unit: str
    completed_units: list[str]
    next_batch_index: int
    total_batches: int
    optimizer_steps_total: int
    total_optimizer_steps: int
    unit_snapshot: list[str]
    unit_completed: bool
    metadata_only: bool = False


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
            self._select_trainable_parameters(),
            lr=self.config.learning_rate,
        )
        self.scheduler = None
        self._completed_units: list[str] = []
        self._checkpoint_state: dict[str, Any] | None = None
        self._checkpoint_manifest: CheckpointManifest | None = None
        self._model_slot_to_registry_slot_id: dict[int, int] = {}

    def _resolve_max_length(self) -> int:
        candidates: list[int] = []
        tokenizer_max_length = getattr(self.tokenizer, "model_max_length", None)
        if isinstance(tokenizer_max_length, int) and 0 < tokenizer_max_length < 100_000:
            candidates.append(tokenizer_max_length)

        model_config = getattr(self.model, "config", None)
        for attr in ("n_positions", "max_position_embeddings", "n_ctx"):
            value = getattr(model_config, attr, None)
            if isinstance(value, int) and value > 0:
                candidates.append(value)

        if candidates:
            return min(candidates)
        return 512

    def _collate_batch(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        pad_token_id = getattr(self.tokenizer, "pad_token_id", 0) or 0
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [item["input_ids"] for item in batch],
            batch_first=True,
            padding_value=pad_token_id,
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            [item["attention_mask"] for item in batch],
            batch_first=True,
            padding_value=0,
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            [item["labels"] for item in batch],
            batch_first=True,
            padding_value=-100,
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def _build_dataloader(self, passages, start_batch_index: int = 0):
        start_index = start_batch_index * self.config.batch_size
        ds = PassageDataset(
            passages[start_index:],
            self.tokenizer,
            max_length=self._resolve_max_length(),
        )
        return DataLoader(
            ds,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=self._collate_batch,
        )
    
    def _select_trainable_parameters(self) -> list:
        """Return the parameter list the optimizer should update.

        For SMF, only the sparse memory parameters are trainable; the
        backbone is frozen.  For all other methods the full parameter set
        is used.
        """
        if self.config.method == "smf":
            from .smf_model import SMFModelWrapper
            if not isinstance(self.model, SMFModelWrapper):
                raise TypeError(
                    "method='smf' requires an SMFModelWrapper but got "
                    f"{type(self.model).__name__}"
                )
            params = list(self.model.smf_parameters())
            if not params:
                raise ValueError(
                    "SMFModelWrapper has no trainable memory parameters."
                )
            return params
        elif self.config.method == "casm":
            from .casm_model import CASMModelWrapper
            if not isinstance(self.model, CASMModelWrapper):
                raise TypeError(
                    "method='casm' requires a CASMModelWrapper but got "
                    f"{type(self.model).__name__}"
                )
            params = list(self.model.casm_parameters())
            if not params:
                raise ValueError(
                    "CASMModelWrapper has no trainable CASM parameters."
                )
            return params
        return list(self.model.parameters())

    def _rebuild_optimizer_for_casm(self) -> None:
        """Rebuild AdamW after router expansion, preserving state for existing params."""
        from .casm_model import CASMModelWrapper
        if not isinstance(self.model, CASMModelWrapper):
            return
        # Map old param tensor id -> optimizer state entry
        old_param_states: dict[int, dict] = {}
        old_state = self.optimizer.state_dict()
        flat_params = [p for g in self.optimizer.param_groups for p in g["params"]]
        for flat_idx, param in enumerate(flat_params):
            if flat_idx in old_state["state"]:
                old_param_states[id(param)] = old_state["state"][flat_idx]
        # Rebuild with full expanded param set
        new_params = list(self.model.casm_parameters())
        self.optimizer = torch.optim.AdamW(new_params, lr=self.config.learning_rate)
        # Re-inject preserved state for params that existed before expansion
        for group in self.optimizer.param_groups:
            for param in group["params"]:
                if id(param) in old_param_states:
                    self.optimizer.state[param] = old_param_states[id(param)]

    def _train_step(self, batch) -> float:
        batch = {k: v.to(self.device) for k, v in batch.items()}
        outputs = self.model(**batch)
        raw_loss = outputs.loss
        if (
            self.config.method == "smf"
            and self.config.smf_regularization_weight > 0
        ):
            from .smf_model import SMFModelWrapper
            if isinstance(self.model, SMFModelWrapper):
                reg = self.model.compute_regularization_loss()
                raw_loss = raw_loss + self.config.smf_regularization_weight * reg
        if (
            self.config.method == "casm"
            and (self.config.casm_sparsity_weight > 0 or self.config.casm_overlap_weight > 0)
        ):
            from .casm_model import CASMModelWrapper
            if isinstance(self.model, CASMModelWrapper):
                if self.config.casm_sparsity_weight > 0:
                    raw_loss = raw_loss + self.config.casm_sparsity_weight * self.model.compute_sparsity_loss()
                if self.config.casm_overlap_weight > 0:
                    raw_loss = raw_loss + self.config.casm_overlap_weight * self.model.compute_overlap_loss()
        (raw_loss / self.config.grad_accum_steps).backward()
        return float(raw_loss.item())

    def _get_training_probes(self, dataset):
        try:
            return dataset.get_probes("changed")
        except TypeError:
            return dataset.get_probes()

    def _build_scheduler(self, total_optimizer_steps: int):
        scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=max(total_optimizer_steps, 1),
        )
        self.scheduler = scheduler
        return scheduler

    def _capture_rng_state(self) -> dict[str, Any]:
        state: dict[str, Any] = {
            "python": random.getstate(),
            "torch": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
        return state

    def _restore_rng_state(self, state: dict[str, Any]) -> None:
        random.setstate(state["python"])
        torch.set_rng_state(state["torch"])
        if torch.cuda.is_available() and "cuda" in state:
            torch.cuda.set_rng_state_all(state["cuda"])

    def _move_optimizer_state_to_device(self) -> None:
        for state in self.optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(self.device)

    def _build_unit_snapshot(self, passages: list[str]) -> list[str]:
        snapshot: list[str] = []
        for _ in range(self.config.epochs_per_period):
            if len(passages) <= 1:
                snapshot.extend(passages)
                continue
            ordering = torch.randperm(len(passages)).tolist()
            snapshot.extend(passages[index] for index in ordering)
        return snapshot

    def _update_checkpoint_state(
        self,
        *,
        period: str,
        last_period: str,
        completed_units: list[str],
        unit_snapshot: list[str],
        next_batch_index: int,
        total_batches: int,
        optimizer_steps_total: int,
        total_optimizer_steps: int,
        unit_completed: bool,
    ) -> None:
        self._checkpoint_state = {
            "schema_version": 1,
            "last_period": last_period,
            "current_unit": period,
            "completed_units": completed_units,
            "next_batch_index": next_batch_index,
            "total_batches": total_batches,
            "optimizer_steps_total": optimizer_steps_total,
            "total_optimizer_steps": total_optimizer_steps,
            "unit_snapshot": unit_snapshot,
            "unit_completed": unit_completed,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (
                self.scheduler.state_dict() if self.scheduler is not None else None
            ),
            "rng_state": self._capture_rng_state(),
        }

    def train_period(
        self,
        dataset: TemporalDataset | None,
        period: str,
        checkpoint_hook: Callable[[str, int], None] | None = None,
        event_hook: Callable[[dict[str, Any]], None] | None = None,
        resume_state: ResumeState | None = None,
    ) -> dict[str, Any]:
        start = time.time()
        print(f"Using device: {self.device}")

        if dataset is None:
            raise ValueError("dataset is required to provide probes for the training unit")

        probes = self._get_training_probes(dataset)
        if resume_state is None:
            passages = dataset.get_train_passages()
            passages = self.filter.filter(passages)
            if self.config.max_passages_per_period is not None:
                passages = passages[:self.config.max_passages_per_period]
            unit_snapshot = self._build_unit_snapshot(passages)
            start_batch_index = 0
            optimizer_steps_total = 0
            scheduler = self._build_scheduler(
                math.ceil(math.ceil(len(unit_snapshot) / self.config.batch_size) / self.config.grad_accum_steps)
            )
            self.optimizer.zero_grad()
        else:
            unit_snapshot = list(resume_state.unit_snapshot)
            start_batch_index = resume_state.next_batch_index
            optimizer_steps_total = resume_state.optimizer_steps_total
            scheduler = self._build_scheduler(resume_state.total_optimizer_steps)
            scheduler_state = self._checkpoint_state["scheduler_state_dict"] if self._checkpoint_state else None
            if scheduler_state is not None:
                scheduler.load_state_dict(scheduler_state)
            self._restore_rng_state(self._checkpoint_state["rng_state"])
            self.optimizer.zero_grad()

        print(f"Training on {len(unit_snapshot)} ordered passages")

        _casm_pending_slot_links: list[tuple[int, object]] = []
        contradictions = self.detector.check(probes, self.registry)
        dataloader = self._build_dataloader(unit_snapshot, start_batch_index=start_batch_index)

        total_micro_steps = math.ceil(len(unit_snapshot) / self.config.batch_size)
        total_optimizer_steps = math.ceil(total_micro_steps / self.config.grad_accum_steps)

        if (
            self.config.method == "casm"
            and self.config.casm_branch_on_contradiction
            and contradictions
        ):
            from .casm_model import CASMModelWrapper
            if isinstance(self.model, CASMModelWrapper):
                for contradiction_probe in contradictions:
                    new_model_slot_id = self.model.add_memory_slot()
                    _casm_pending_slot_links.append((new_model_slot_id, contradiction_probe))
                self._rebuild_optimizer_for_casm()
                scheduler = self._build_scheduler(total_optimizer_steps)
                self.optimizer.zero_grad()

        self.model.train()

        loss_curve = []
        final_loss = 0.0
        micro_steps_total = start_batch_index
        total_tokens_trained = 0
        window_loss_total = 0.0
        window_micro_steps = 0
        window_tokens_total = 0
        window_data_wait_sec = 0.0
        window_forward_backward_sec = 0.0
        total_data_wait_sec = 0.0
        total_forward_backward_sec = 0.0
        total_optimizer_step_sec = 0.0
        next_batch_wait_start = time.perf_counter()

        self._update_checkpoint_state(
            period=period,
            last_period=resume_state.last_period if resume_state is not None else period,
            completed_units=list(self._completed_units),
            unit_snapshot=unit_snapshot,
            next_batch_index=start_batch_index,
            total_batches=total_micro_steps,
            optimizer_steps_total=optimizer_steps_total,
            total_optimizer_steps=total_optimizer_steps,
            unit_completed=False,
        )

        for batch in dataloader:
            batch_ready_time = time.perf_counter()
            data_wait_sec = batch_ready_time - next_batch_wait_start
            forward_backward_start = time.perf_counter()
            loss = self._train_step(batch)
            forward_backward_sec = time.perf_counter() - forward_backward_start
            batch_tokens = int(batch["attention_mask"].sum().item())
            micro_steps_total += 1
            total_tokens_trained += batch_tokens
            window_micro_steps += 1
            window_tokens_total += batch_tokens
            window_loss_total += loss
            window_data_wait_sec += data_wait_sec
            window_forward_backward_sec += forward_backward_sec

            is_window_boundary = window_micro_steps == self.config.grad_accum_steps
            is_last_micro_step = micro_steps_total == total_micro_steps
            if not (is_window_boundary or is_last_micro_step):
                next_batch_wait_start = time.perf_counter()
                continue

            optimizer_step_start = time.perf_counter()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()
            scheduler.step()
            self.optimizer.zero_grad()
            optimizer_step_sec = time.perf_counter() - optimizer_step_start
            optimizer_steps_total += 1
            total_data_wait_sec += window_data_wait_sec
            total_forward_backward_sec += window_forward_backward_sec
            total_optimizer_step_sec += optimizer_step_sec

            averaged_window_loss = window_loss_total / window_micro_steps
            final_loss = averaged_window_loss
            step_wall_sec = window_data_wait_sec + window_forward_backward_sec + optimizer_step_sec
            effective_tokens_per_sec = (
                window_tokens_total / step_wall_sec if step_wall_sec > 0 else 0.0
            )
            if event_hook is not None:
                event_hook(
                    {
                        "event_type": "train_step",
                        "unit": period,
                        "optimizer_step": optimizer_steps_total,
                        "total_optimizer_steps": total_optimizer_steps,
                        "micro_step": micro_steps_total,
                        "loss": averaged_window_loss,
                        "tokens_in_step": window_tokens_total,
                        "data_wait_sec": window_data_wait_sec,
                        "forward_backward_sec": window_forward_backward_sec,
                        "optimizer_step_sec": optimizer_step_sec,
                        "step_wall_sec": step_wall_sec,
                        "effective_tokens_per_sec": effective_tokens_per_sec,
                    }
                )
            if optimizer_steps_total % self.config.log_every_n_steps == 0:
                print(f"step={optimizer_steps_total}, loss={averaged_window_loss:.4f}")
                loss_curve.append((optimizer_steps_total, averaged_window_loss))

            self._update_checkpoint_state(
                period=period,
                last_period=period,
                completed_units=list(self._completed_units),
                unit_snapshot=unit_snapshot,
                next_batch_index=micro_steps_total,
                total_batches=total_micro_steps,
                optimizer_steps_total=optimizer_steps_total,
                total_optimizer_steps=total_optimizer_steps,
                unit_completed=False,
            )

            should_checkpoint = (
                checkpoint_hook is not None
                and self.config.checkpoint_every_n_optimizer_steps is not None
                and optimizer_steps_total % self.config.checkpoint_every_n_optimizer_steps == 0
                and optimizer_steps_total < total_optimizer_steps
            )
            if should_checkpoint:
                checkpoint_hook(period, optimizer_steps_total)

            window_loss_total = 0.0
            window_micro_steps = 0
            window_tokens_total = 0
            window_data_wait_sec = 0.0
            window_forward_backward_sec = 0.0
            next_batch_wait_start = time.perf_counter()
        
        probe_to_registry_slot_id: dict[int, int] = {}
        for probe in probes:
            reg_slot = self.registry.write(probe, period)
            probe_to_registry_slot_id[id(probe)] = reg_slot.slot_id

        if self.config.method == "casm":
            from .casm_model import CASMModelWrapper
            if isinstance(self.model, CASMModelWrapper):
                for model_slot_id, contradiction_probe in _casm_pending_slot_links:
                    reg_slot_id = probe_to_registry_slot_id.get(id(contradiction_probe))
                    if reg_slot_id is not None:
                        self._model_slot_to_registry_slot_id[model_slot_id] = reg_slot_id
                for model_slot_id, reg_slot_id in self._model_slot_to_registry_slot_id.items():
                    count = self.model._slot_usage_counts.get(model_slot_id, 0)
                    if count > 0:
                        for reg_slot in self.registry._slots:
                            if reg_slot.slot_id == reg_slot_id:
                                reg_slot.usage_count += count
                                break

        if period not in self._completed_units:
            self._completed_units.append(period)
        self._update_checkpoint_state(
            period=period,
            last_period=period,
            completed_units=list(self._completed_units),
            unit_snapshot=unit_snapshot,
            next_batch_index=total_micro_steps,
            total_batches=total_micro_steps,
            optimizer_steps_total=optimizer_steps_total,
            total_optimizer_steps=total_optimizer_steps,
            unit_completed=True,
        )

        duration = time.time() - start

        result = {
            "period": period,
            "train_loss_final": final_loss,
            "train_loss_curve": loss_curve,
            "n_passages_trained": len(unit_snapshot),
            "n_contradiction_passages": sum(1 for p in probes if p.is_contradiction),
            "train_duration_sec": duration,
            "micro_steps_total": micro_steps_total,
            "optimizer_steps_total": optimizer_steps_total,
        }
        if event_hook is not None:
            effective_tokens_per_sec = total_tokens_trained / duration if duration > 0 else 0.0
            event_hook(
                {
                    "event_type": "period_end",
                    "unit": period,
                    "train_loss_final": result["train_loss_final"],
                    "n_passages_trained": result["n_passages_trained"],
                    "n_contradiction_passages": result["n_contradiction_passages"],
                    "train_duration_sec": result["train_duration_sec"],
                    "micro_steps_total": result["micro_steps_total"],
                    "optimizer_steps_total": result["optimizer_steps_total"],
                    "tokens_trained_total": total_tokens_trained,
                    "data_wait_sec_total": total_data_wait_sec,
                    "forward_backward_sec_total": total_forward_backward_sec,
                    "optimizer_step_sec_total": total_optimizer_step_sec,
                    "effective_tokens_per_sec": effective_tokens_per_sec,
                }
            )
        return result

    def checkpoint(
        self,
        period: str,
        run_root: str,
        *,
        manifest_metadata: dict[str, Any] | None = None,
        lock_run_root: bool = True,
    ) -> str:
        prepare_run_root(run_root)
        if lock_run_root:
            with RunRootLock(run_root):
                return self._checkpoint_unlocked(period, run_root, manifest_metadata=manifest_metadata)
        return self._checkpoint_unlocked(period, run_root, manifest_metadata=manifest_metadata)

    def _checkpoint_unlocked(
        self,
        period: str,
        run_root: str,
        *,
        manifest_metadata: dict[str, Any] | None = None,
    ) -> str:
        temp_dir = create_checkpoint_tempdir(run_root)
        self.model.save_pretrained(temp_dir)
        self.tokenizer.save_pretrained(temp_dir)
        self.registry.save(os.path.join(temp_dir, "memory_registry.json"))

        with open(os.path.join(temp_dir, "train_config.json"), "w") as f:
            json.dump(asdict(self.config), f, indent=2)
        with open(os.path.join(temp_dir, "last_period.txt"), "w") as f:
            f.write(period)
        if self._checkpoint_state is not None:
            checkpoint_state = dict(self._checkpoint_state)
            checkpoint_state["rng_state"] = self._capture_rng_state()
            torch.save(
                checkpoint_state,
                os.path.join(temp_dir, TRAINER_STATE_FILENAME),
            )
        if manifest_metadata is not None:
            write_checkpoint_manifest(
                temp_dir,
                model_name=manifest_metadata["model_name"],
                training_plan=manifest_metadata["training_plan"],
                resume_compatibility=manifest_metadata["resume_compatibility"],
                dataset_identity=manifest_metadata["dataset_identity"],
            )

        final_dir = finalize_checkpoint(run_root, temp_dir, last_period=period)
        return str(final_dir)

    def resume(self, path: str) -> ResumeState:
        checkpoint_path = resolve_checkpoint_path(path)

        registry_path = os.path.join(checkpoint_path, "memory_registry.json")
        if os.path.exists(registry_path):
            self.registry.load(registry_path)

        period_file = os.path.join(checkpoint_path, "last_period.txt")
        if not os.path.exists(period_file):
            raise FileNotFoundError(f"Missing checkpoint metadata: {period_file}")

        with open(period_file, "r") as f:
            last_period = f.read().strip()

        trainer_state_path = os.path.join(checkpoint_path, TRAINER_STATE_FILENAME)
        if not os.path.exists(trainer_state_path):
            return ResumeState(
                checkpoint_path=str(checkpoint_path),
                last_period=last_period,
                current_unit=last_period,
                completed_units=[last_period],
                next_batch_index=0,
                total_batches=0,
                optimizer_steps_total=0,
                total_optimizer_steps=0,
                unit_snapshot=[],
                unit_completed=True,
                metadata_only=True,
            )

        self._checkpoint_manifest = validate_checkpoint_manifest(checkpoint_path)
        trainer_state = torch.load(trainer_state_path, map_location="cpu", weights_only=False)
        self.optimizer.load_state_dict(trainer_state["optimizer_state_dict"])
        self._move_optimizer_state_to_device()
        self._checkpoint_state = trainer_state
        self._completed_units = list(trainer_state["completed_units"])

        return ResumeState(
            checkpoint_path=str(checkpoint_path),
            last_period=last_period,
            current_unit=trainer_state["current_unit"],
            completed_units=list(trainer_state["completed_units"]),
            next_batch_index=trainer_state["next_batch_index"],
            total_batches=trainer_state["total_batches"],
            optimizer_steps_total=trainer_state["optimizer_steps_total"],
            total_optimizer_steps=trainer_state["total_optimizer_steps"],
            unit_snapshot=list(trainer_state["unit_snapshot"]),
            unit_completed=bool(trainer_state["unit_completed"]),
            metadata_only=False,
        )
