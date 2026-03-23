from dataclasses import dataclass

from train_config import TrainConfig


DEFAULT_TEMPORAL_WIKI_PLAN = [
    "aug_sep",
    "sep_oct",
    "oct_nov",
    "nov_dec",
]


@dataclass(frozen=True)
class TrainingPlan:
    dataset_name: str
    units: list[str]


def build_training_plan(cfg: TrainConfig, units: list[str] | None = None) -> TrainingPlan:
    if units is not None:
        if not units:
            raise ValueError("TrainingPlan units must not be empty")
        return TrainingPlan(dataset_name=cfg.dataset_name, units=list(units))

    if cfg.dataset_name == "temporal_wiki":
        return TrainingPlan(
            dataset_name=cfg.dataset_name,
            units=list(DEFAULT_TEMPORAL_WIKI_PLAN),
        )

    return TrainingPlan(
        dataset_name=cfg.dataset_name,
        units=[cfg.dataset_name],
    )
