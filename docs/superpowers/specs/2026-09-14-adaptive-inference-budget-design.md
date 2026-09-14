# Adaptive inference budget for PC-ALM: design

Date: 2026-09-14
Project: Augmented Lagrangian Predictive Coding (arXiv 2605.31022, Seely and Gould, Sakana AI)
Status: approved in chat, phase 1 (global adaptive budget)

## 1. Problem

Algorithm 1 of the paper runs exactly `T` primal-dual inference cycles per batch and then
updates the weights with the gradient of the augmented Lagrangian at the final activities and
multipliers. The weight gradient at layer `i` is driven by the composite credit

    g_i = lambda_i + rho * r_i,        r_i = h_i - sigma(W_i h_{i-1}).

The paper sets `T = L` or `T = 2L`. Appendix D shows why `2L` works: credit travels as a damped
wave with group velocity `sqrt(alpha * eta_h)` layers per step, so the wavefront reaches the
input layer at `t_infl ~= L / sqrt(alpha * eta_h)`, which is about `2L` at `eta_h ~= 1/4`,
`alpha = 1`. The fixed budget is therefore a proxy for "the credit wave has crossed the network".

Goal: replace the fixed budget with a trigger computed from the quantities already present in
the inference loop (`r_i`, `lambda_i`, `g_i`) and test whether that yields

- compute savings: fewer mean inference steps per batch at matched test accuracy, and
- accuracy gains: higher test accuracy than `T = L` / `T = 2L` at the same or smaller step cap,
  especially in deep narrow cells where PC underperforms BP.

Phase 1 (this spec) is a global, batch-level trigger. Phase 2 (later spec) is a per-layer
trigger where each layer updates its weights when its own credit settles.

## 2. Code base

The official JAX implementation (github.com/SakanaAI/pc-alm, MIT) is vendored into this
repository unchanged as the baseline. Its model, data, optimiser, and training code are not
modified. The new method is added alongside `pc` and `pcalm` so the paper's fixed-budget
baselines remain reproducible from the same code.

Files touched by phase 1:

- `pcalm/inference.py`: new `run_pcalm_adaptive` built on `jax.lax.while_loop`; a new
  `Schedule.family == "pcalm_adaptive"` branch in `infer_for_schedule`; the function returns the
  step count in addition to activities and multipliers.
- `pcalm/config.py`: new `MethodConfig` fields `tau`, `patience`, `t_min`, `t_max`,
  `arrival_frac`.
- `pcalm/training.py`: record the per-batch step count; emit a per-run summary line on stdout;
  optional per-step diagnostics (`delta_t`, per-layer `||r_i||`, `||lambda_i||`, `||g_i||`) for
  a single diagnostic batch, written to `diag_trace.csv`.
- `train.py`, `scripts/run_headline_grid.py`: CLI flags for the new fields and for the
  `pcalm_adaptive` method.
- `tests/test_adaptive.py`: unit tests (section 5).

Data lives outside the repository at a fixed absolute path and is passed with `--data-dir`,
because runs are built from the committed source archive.

## 3. The trigger

After each primal-dual cycle `t` (primal step, then dual step), compute over the batch

    delta_t = sum_i ||g_i^(t) - g_i^(t-1)||_F / ( sum_i ||g_i^(t)||_F + eps )

with `eps = 1e-12`. Stop and take the final primal step when all of the following hold:

1. `t >= t_min` (default 2, so the first PC-like step never triggers);
2. arrival guard: `||g_1^(t)|| >= arrival_frac * ||g_{L-1}^(t)||` with `arrival_frac = 0.1`,
   i.e. the input-side credit has reached a usable fraction of the output-side credit, so an
   untouched network cannot look converged;
3. `delta_t < tau` for `patience` consecutive cycles (default `patience = 3`), which filters
   the oscillatory transients caused by the complex eigenvalues of the iteration matrix.

If the criterion never fires, stop at `t_max` (default `3L`, deliberately above the paper's
`2L` so the adaptive method is allowed to spend more when it decides to).

The loop then performs the paper's final primal step and forms the weight gradient exactly as
`run_pcalm` does, with `weight_credit_timing = "pre_dual_energy"`.

Hyperparameters swept in phase 1: `tau in {0.01, 0.03, 0.1}`. All others fixed as above. One
setting must work across depths; per-cell tuning of `tau` is a negative result, not an option.

Batch semantics: the stop decision is shared by the whole batch because the weight update
averages over the batch. Per-sample stopping is deferred to phase 2.

## 4. Experiments

All runs: Fashion-MNIST, ReLU residual MLP, one epoch over 60k training samples, batch 64,
Adam with the paper's learning-rate rule, the paper's frozen per-cell `eta_h` from
`configs/eta_best_by_cell.csv`, `rho = 1`, `alpha = 1`, three seeds (0, 1, 2).

Cells (width N, depth L): (32, 16), (32, 32), (32, 64), (8, 64). The last two are the deep
narrow regime where PC fails.

Methods per cell:

- BP (reference).
- PC, `T = 2L`.
- PC-ALM, `T = L`.
- PC-ALM, `T = 2L`.
- PC-ALM adaptive, `tau in {0.01, 0.03, 0.1}`, `t_max = 3L`.

Metrics per run, printed as one summary line and written to `summary.json`:

- final test accuracy, final train accuracy;
- gradient cosine to BP on a diagnostic batch;
- mean, median, min, max inference steps per batch (fixed methods report `T`);
- fraction of batches that hit `t_max`.

Figures:

1. Test accuracy versus mean inference steps per batch, one panel per cell, fixed-budget
   methods as anchor points, adaptive runs as points labelled by `tau`, error bars over seeds.
2. For one batch of the (32, 64) cell at initialisation: `delta_t` and per-layer `||g_i||`
   versus `t`, with the predicted inflection `L / sqrt(alpha * eta_h)` marked and the trigger
   step marked.

Decision rule for promoting to phase 2: an adaptive setting counts as a success if, across
cells, its accuracy is within one seed standard deviation of PC-ALM `T = 2L` with fewer mean
steps, or exceeds it at any mean step count. If no single `tau` satisfies this in all four
cells, report that and reconsider the criterion before building the per-layer trigger.

## 5. Testing

- With `tau = 0` (never fires) and `t_max = T`, the adaptive gradient equals `run_pcalm` at
  budget `T` to `atol = 1e-5`.
- Per-sample activities at the stop step are batch-size invariant, mirroring the existing
  `test_inference_is_per_sample_batch_invariant`.
- With `tau` large and `t_min = 2`, the loop stops at exactly `t_min + patience - 1`... only if
  the arrival guard passes; a test constructs a case where the guard blocks stopping and checks
  the step count exceeds `t_min`.
- The reported step count is an integer in `[t_min, t_max]`.

## 6. Experiment tree

Baseline node: vendored code, run command fixed to the uv-based entry point with absolute
`--data-dir`. Round 1 children under the baseline are the per-cell method variants, siblings in
one bush. Phase 2 will branch from the best phase-1 node.

## 7. Out of scope for phase 1

Per-layer or per-sample triggers, changes to `rho` or `alpha` schedules, other datasets,
multi-epoch training, GPU backends.
