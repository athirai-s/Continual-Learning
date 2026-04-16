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

    min_passage_length: int = 0
    deduplicate: bool = True
    weighted_sampling: bool = False
    max_passages_per_period: Optional[int] = None
    dataset_fraction: Optional[float] = None  # fraction of data to use (0, 1]; None = all

    shuffle_by_relation: bool = True
    contradiction_batch_frac: float = 0.25

    run_id: str = "debug_run"
    log_every_n_steps: int = 100
    eval_after_each_period: bool = True
    capture_activations: bool = False

    checkpoint_dir: str = "checkpoints"
    checkpoint_every_n_optimizer_steps: Optional[int] = None
    seed: int = 0

    # LoRA-specific fields
    lora_r: Optional[int] = None
    lora_alpha: Optional[int] = None
    lora_dropout: float = 0.05
    lora_target_modules: Optional[list] = None

    # SMF-specific fields
    smf_memory_size: Optional[int] = None
    smf_sparsity_ratio: Optional[float] = None
    smf_update_layers: Optional[list] = None
    smf_regularization_weight: float = 0.01
    smf_freeze_backbone: bool = True
    smf_learning_rate: Optional[float] = None

    # CASM-specific fields
    casm_num_slots: Optional[int] = None
    casm_router_hidden_size: Optional[int] = None
    casm_top_k: Optional[int] = None
    casm_router_temperature: float = 1.0
    casm_sparsity_weight: float = 0.0
    casm_overlap_weight: float = 0.0
    casm_branch_on_contradiction: bool = True
    casm_memory_size: Optional[int] = None
    casm_num_injection_layers: Optional[int] = None
    casm_router_type: str = "mlp"  # 'mlp' = CASMRouter (learned), 'similarity' = SimilarityRouter (cosine, no training)

    def validate(self) -> None:
        if self.method not in {"full_ft", "lora", "smf", "casm"}:
            raise ValueError(f"Unsupported method: {self.method}")
        if self.precision not in {"bfloat16", "float16", "int8"}:
            raise ValueError(f"Unsupported precision: {self.precision}")
        if self.dataset_name not in {"temporal_wiki", "tsqa", "tgqa", "synthetic"}:
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
        if self.dataset_fraction is not None and not (0.0 < self.dataset_fraction <= 1.0):
            raise ValueError("dataset_fraction must be in (0, 1]")

        if self.method == "lora":
            self._validate_lora()
        elif self.method == "smf":
            self._validate_smf()
        elif self.method == "casm":
            self._validate_casm()

    def _validate_lora(self) -> None:
        if self.lora_r is None or self.lora_r < 1:
            raise ValueError("lora_r must be >= 1 for method='lora'")
        if self.lora_alpha is None or self.lora_alpha < 1:
            raise ValueError("lora_alpha must be >= 1 for method='lora'")
        if not (0.0 <= self.lora_dropout < 1.0):
            raise ValueError("lora_dropout must be in [0, 1) for method='lora'")
        if self.lora_target_modules is not None:
            if not self.lora_target_modules:
                raise ValueError(
                    "lora_target_modules must be non-empty when provided for method='lora'"
                )
            if any(
                not isinstance(module, str) or not module.strip()
                for module in self.lora_target_modules
            ):
                raise ValueError(
                    "lora_target_modules must contain non-empty strings for method='lora'"
                )

    def _validate_smf(self) -> None:
        if self.smf_memory_size is None or self.smf_memory_size <= 0:
            raise ValueError("smf_memory_size must be > 0 for method='smf'")
        if self.smf_sparsity_ratio is None or not (0 < self.smf_sparsity_ratio <= 1):
            raise ValueError("smf_sparsity_ratio must be in (0, 1] for method='smf'")
        if not self.smf_update_layers:
            raise ValueError("smf_update_layers must be non-empty for method='smf'")
        if self.smf_regularization_weight < 0:
            raise ValueError("smf_regularization_weight must be >= 0")
        if not self.smf_freeze_backbone:
            raise ValueError("smf_freeze_backbone must be True for method='smf'")

    def _validate_casm(self) -> None:
        if self.casm_num_slots is None or self.casm_num_slots < 1:
            raise ValueError("casm_num_slots must be >= 1 for method='casm'")
        if self.casm_top_k is None or self.casm_top_k < 1:
            raise ValueError("casm_top_k must be >= 1 for method='casm'")
        if self.casm_top_k > self.casm_num_slots:
            raise ValueError("casm_top_k must be <= casm_num_slots")
        if self.casm_router_hidden_size is None or self.casm_router_hidden_size <= 0:
            raise ValueError("casm_router_hidden_size must be > 0 for method='casm'")
        if self.casm_sparsity_weight < 0:
            raise ValueError("casm_sparsity_weight must be >= 0")
        if self.casm_overlap_weight < 0:
            raise ValueError("casm_overlap_weight must be >= 0")
        if self.casm_memory_size is not None and self.casm_memory_size <= 0:
            raise ValueError("casm_memory_size must be > 0")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TrainConfig":
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)

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
