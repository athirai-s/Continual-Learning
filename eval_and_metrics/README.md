# eval_and_metrics — PR 6: Evaluation and Metrics

Per-period evaluation for SMF and CASM continual-learning runs.

## Files

| File | Purpose |
|---|---|
| `evaluation.py` | Core metric computation — plasticity, stability, contradiction_acc, routing_acc, perplexity |
| `metric_helpers.py` | Pure-Python utilities: token_f1, retention, forgetting, BWT, version coverage |
| `evaluation_runner.py` | Thin adapter called by `train_runner.run_training()` when `cfg.eval_after_each_period=True` |
| `report.py` | CLI: print a per-period table / JSON / CSV for a completed run |
| `compare_runs.py` | CLI: side-by-side metric comparison across full_ft / lora / smf / casm runs |
| `tests/` | Unit tests for all of the above |

## Metrics by method

| Metric key | full_ft | lora | smf | casm |
|---|---|---|---|---|
| `plasticity` / `{method}/plasticity` | ✓ | ✓ | ✓ | ✓ |
| `stability` / `{method}/stability` | ✓ | ✓ | ✓ | ✓ |
| `{method}/perplexity_changed` | ✓ | ✓ | ✓ | ✓ |
| `smf/active_params`, `smf/sparsity` | — | — | ✓ | — |
| `casm/contradiction_acc` | — | — | — | ✓ |
| `casm/routing_acc` | — | — | — | ✓ |

**Plasticity** — fraction of *changed* probes answered correctly after training.  
**Stability** — fraction of *unchanged* probes still answered correctly (resistance to forgetting).  
**Contradiction accuracy** — fraction of contradiction probes where the model returns the *new* answer (CASM branch correctness).  
**Routing accuracy** — fraction of probes routed to the slot that owns the correct version (CASM router quality).

## How it fits into the runner

`train_runner.py` already calls `run_period_evaluation` when `cfg.eval_after_each_period=True`:

```python
# train_runner.py (existing code — no changes needed)
from .evaluation_runner import run_period_evaluation

if cfg.eval_after_each_period:
    eval_dataset = dataset_factory(unit, cfg)
    result["evaluation"] = run_period_evaluation(
        model=trainer.model,
        tokenizer=trainer.tokenizer,
        dataset=eval_dataset,
        cfg=cfg,
        unit=period_name,
        run_root=run_root,
    )
```

The one addition needed: pass `registry=trainer.registry` so CASM metrics can access slot state:

```python
result["evaluation"] = run_period_evaluation(
    model=trainer.model,
    tokenizer=trainer.tokenizer,
    dataset=eval_dataset,
    cfg=cfg,
    unit=period_name,
    run_root=run_root,
    registry=trainer.registry,   # ← add this
)
```

## CLI usage

```bash
# Per-run period table
python eval_and_metrics/report.py --run-root checkpoints/my_run_id

# JSON output
python eval_and_metrics/report.py --run-root checkpoints/my_run_id --format json

# Compare full_ft vs smf vs casm
python eval_and_metrics/compare_runs.py \
    --runs checkpoints/run_full_ft checkpoints/run_smf checkpoints/run_casm \
    --metric casm/plasticity smf/plasticity plasticity train_loss_final
```

## Running the tests

```bash
# From repo root
pytest eval_and_metrics/tests/ -v
```
