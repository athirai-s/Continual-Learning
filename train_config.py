from dataclasses import dataclass, asdict
from typing import Optional, Any
import json


@dataclass
class TrainConfig:
    model_name: str
    method: str
    dataset_name: str = "temporal_wiki"
    precision: str = "bfloat16"

    learning_rate: float = 2e-5
    batch_size: int = 8
    grad_accum_steps: int = 4
    epochs_per_period: int = 1
    grad_clip: float = 1.0
    warmup_steps: int = 100

    min_passage_length: int = 100
    deduplicate: bool = True
    weighted_sampling: bool = False
    max_passages_per_period: Optional[int] = None

    shuffle_by_relation: bool = True
    contradiction_batch_frac: float = 0.25

    run_id: str = "debug_run"
    log_every_n_steps: int = 100
    eval_after_each_period: bool = True

    checkpoint_dir: str = "checkpoints"
    checkpoint_every_n_optimizer_steps: Optional[int] = None
    seed: int = 0

    def validate(self) -> None:
        if self.method not in {"full_ft", "lora", "smf"}:
            raise ValueError(f"Unsupported method: {self.method}")
        if self.precision not in {"bfloat16", "float16", "int8"}:
            raise ValueError(f"Unsupported precision: {self.precision}")
        if self.dataset_name not in {"temporal_wiki", "tsqa", "tgqa"}:
            raise ValueError(f"Unsupported dataset_name: {self.dataset_name}")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if self.grad_accum_steps <= 0:
            raise ValueError("grad_accum_steps must be > 0")
        if self.epochs_per_period <= 0:
            raise ValueError("epochs_per_period must be > 0")
        if self.min_passage_length < 0:
            raise ValueError("min_passage_length must be >= 0")
        if not (0.0 <= self.contradiction_batch_frac <= 1.0):
            raise ValueError("contradiction_batch_frac must be between 0 and 1")
        if self.log_every_n_steps <= 0:
            raise ValueError("log_every_n_steps must be > 0")
        if (
            self.checkpoint_every_n_optimizer_steps is not None
            and self.checkpoint_every_n_optimizer_steps <= 0
        ):
            raise ValueError("checkpoint_every_n_optimizer_steps must be > 0")
        if self.seed < 0:
            raise ValueError("seed must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @staticmethod
    def make_config(
        run_id: str,
        model_name: str,
        method: str = "full_ft",
        dataset_name: str = "temporal_wiki",
        batch_size: int = 8,
        grad_accum_steps: int = 4,
        max_passages_per_period: int | None = None,
        log_every_n_steps: int = 100,
        checkpoint_every_n_optimizer_steps: int | None = None,
        seed: int = 0,
    ) -> "TrainConfig":
        cfg = TrainConfig(
            run_id=run_id,
            model_name=model_name,
            method=method,
            dataset_name=dataset_name,
            batch_size=batch_size,
            grad_accum_steps=grad_accum_steps,
            max_passages_per_period=max_passages_per_period,
            log_every_n_steps=log_every_n_steps,
            checkpoint_every_n_optimizer_steps=checkpoint_every_n_optimizer_steps,
            seed=seed,
        )
        cfg.validate()
        return cfg
