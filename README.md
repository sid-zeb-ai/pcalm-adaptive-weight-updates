# Adaptive inference budgets for PC-ALM

Research code and reports building on **Augmented Lagrangian Predictive Coding** (Seely and
Gould, arXiv 2605.31022; official implementation github.com/SakanaAI/pc-alm, MIT, vendored here
unchanged under `pcalm/`).

Question: Algorithm 1 of the paper updates the weights after a fixed `T = 2L` inference cycles.
Can the update instead be triggered by the multipliers and residuals already in the loop, and
does that save compute or improve accuracy?

## Findings so far

- **Phase 1, global trigger** (`pcalm_adaptive`): a batch-level stop rule on the composite credit
  `lambda_i + rho r_i` fires at a fixed step of about 0.85 L / sqrt(alpha eta_h) regardless of
  data or training state. Matched accuracy with 11 to 22% fewer steps; no accuracy gain.
  Report: `reports/adaptive-budget-phase1/`.
- **Phase 2, per-layer freeze-and-fire** (`pcalm_layerwise`): each layer fires on its own credit
  stability, snapshots the credit for its weight update, and stops taking inference steps.
  Matches `T = 2L` accuracy over 1 and 5 epochs with 49% (L = 64) and 56% (L = 32) of the
  active layer-cycles. Fire times are linear in layer index with slope 1.75 cycles/layer against
  the paper's wave prediction of 2.03. Report: `reports/adaptive-budget-phase2/`.

Compute in phase 2 is counted as active layer-cycles; a dense JAX implementation masks frozen
layers. The `sparse-implementation` branch realises the saving in wall-clock with real
conditionals and adds a wavefront gate that also skips layers before the credit wave arrives.

## Layout

- `pcalm/` model, data, inference (`run_pcalm`, `run_pcalm_adaptive`, `run_pcalm_layerwise`), training.
- `scripts/run_node.py` fixed entry point; `configs/run.yaml` is the only file experiment
  branches change; `scripts/make_node_config.py` writes it.
- `tests/` 32 tests including exact-reproduction checks against the paper's algorithm.
- `docs/superpowers/specs/` design documents for phases 1 and 2.
- `reports/` reports, figures with their scripts, and `data/all-nodes-aggregate.csv`
  (one row per experiment node, mean and std over 3 seeds).

Branches named `orx/...` are the experiment tree nodes managed by OpenResearch: each holds the
exact code and config a run used. `experiments-phase1-2` is the phase-2 winner's code plus the
reports; `sparse-implementation` continues from it.

## Running

```sh
uv sync --extra test && uv run pytest -q
uv run python scripts/run_node.py --config configs/run.yaml --data-dir <dir with MNIST/raw, FashionMNIST/raw> --output-root results
```
