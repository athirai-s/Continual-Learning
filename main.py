import argparse

from train_config import TrainConfig
from train_runner import run_mode


DEFAULT_MODEL_NAME = "meta-llama/Llama-3.2-3B"
DEFAULT_DATASET_NAME = "temporal_wiki"
DEFAULT_RUN_ID = "debug_run"
DEFAULT_CHECKPOINT_DIR = "checkpoints"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run continual-learning training.")
    parser.add_argument("--mode", choices=["real", "synthetic"], default="real")
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    return parser


def make_config(args: argparse.Namespace) -> TrainConfig:
    return TrainConfig.make_config(
        run_id=args.run_id,
        model_name=args.model_name,
        method="full_ft",
        dataset_name=args.dataset_name,
        batch_size=1,
        grad_accum_steps=1,
        max_passages_per_period=20,
        log_every_n_steps=1,
    )


def main() -> int:
    args = build_parser().parse_args()
    cfg = make_config(args)
    run_mode(args.mode, cfg, checkpoint_dir=args.checkpoint_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
