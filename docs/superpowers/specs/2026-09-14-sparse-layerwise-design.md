# Sparse layer-wise PC-ALM: real layer skipping and a wavefront gate

Date: 2026-09-14. Status: approved in chat. Round 5 of the experiment tree, child of the phase-2
winner (freeze-and-fire, tau = 0.1, N = 32, L = 64). GitHub branch `sparse-implementation`.

## 1. Purpose

Phase 2 counted compute as *active layer-cycles* because the dense JAX implementation masks
frozen layers with `jnp.where` and still executes their matmuls. This round (a) makes the skip
real, so wall-clock drops, and (b) adds a second, exact gate that also skips a layer *before* the
credit wave reaches it, where its gradient is identically zero. It then measures wall-clock and
executed layer-cycles against the dense fixed `T = 2L` schedule, and tests the scaling
prediction at L = 128.

## 2. Layer-local primal step

For hidden layer `i` (0-indexed in the `free` list, `i = 0..L-2`, batch size `B`), the only terms
of the shifted augmented Lagrangian that depend on `h_i` are

    E_i(h_i) = (rho / 2B) ||r_i + lambda_i / rho||^2  +  (rho / 2B) ||r_{i+1}(h_i) + lambda_{i+1} / rho||^2

for `i < L-2`, and for the last hidden layer `i = L-2` the second term is replaced by the
supervised loss `(1/2B) ||y - pred_L(h_{L-2})||^2`. `r_i = h_i - pred_i(h_{i-1})` uses the
*previous cycle's* `h_{i-1}` (Jacobi update, exactly as the dense `jax.grad` of the full energy
computes all layer gradients from the same old activities). `grad_{h_i} E_i` equals the
corresponding block of the dense gradient; a unit test asserts this to 1e-6.

The primal update is `h_i <- h_i - eta_h * B * grad E_i` (same effective step as `_solve_inner`).

## 3. Gates, implemented with `lax.cond`

Each layer's primal step is wrapped in `jax.lax.cond(run_i, step, identity)`. XLA executes only
the taken branch, so a skipped layer performs no matmuls (verified by timing, section 6).
`run_i` is:

- gate `freeze`:     `run_i = not fired_i`
- gate `wavefront`:  `run_i = not fired_i and not (||g_i^prev|| == 0 and ||g_{i+1}^prev|| == 0)`

where `g^prev` is the credit from the end of the previous cycle (for `i = L-2` the downstream
credit is the supervised error, which is nonzero from cycle 1, so the last hidden layer always
runs until it fires). Before the wave reaches layer `i`, `h_i` equals its forward value and
`lambda_i = 0`, so `g_i` and `g_{i+1}` are exactly zero and `grad E_i` is exactly zero: skipping
the step is exact, not an approximation. The dual step of a skipped layer is also skipped: for a
fired layer in freeze mode that is the phase-2 semantics; for a pre-arrival layer `r_i = 0` so the
step is a no-op.

Firing rule, patience, `t_min`, `t_max`, `eps_arrive`, the frozen credit `G_i`, and the weight
gradient via `duals_eff_i = G_i - rho r_i(final)` are all identical to `run_pcalm_layerwise`
(mode `freeze`). The trigger statistic is computed for all layers every cycle (cheap norms) so
that streak bookkeeping matches the dense version exactly.

## 4. Accounting

`info["executed_layer_cycles"]` counts (layer, cycle) pairs whose `lax.cond` took the compute
branch. `info["active_layer_cycles"]` keeps the phase-2 definition (not fired) so the two reports
stay comparable. Training logs and RESULT lines add `executed_layer_cycles_frac` (normalised by
`(L-1) * 2L`) beside `active_layer_cycles_frac`.

Predictions to test (slope `s ≈ 1.75` cycles/layer, settling width `w ≈ 7`):
- freeze gate:     executed fraction ≈ s/4 + w/(2L)      -> 0.49 at L = 64, 0.47 at L = 128
- wavefront gate:  executed fraction ≈ (s-1)/4 + w/(2L)  -> 0.24 at L = 64, 0.22 at L = 128

## 5. Exactness tests (`tests/test_sparse.py`)

- Per-layer `grad E_i` equals the dense gradient block for random activities and duals, all
  three activations, 1e-6.
- `run_pcalm_layerwise_sparse(gate="freeze")` reproduces `run_pcalm_layerwise(mode="freeze")`:
  final activities, `duals_eff`, steps, fire times, and `method_grad` to 1e-5, on depth 10 and
  depth 4 cases, tanh and relu.
- `gate="wavefront"` reproduces the same dense reference to the same tolerance, and
  `executed_layer_cycles < active_layer_cycles`.
- `tau = 0` (never fires), `t_max = T`: both gates reproduce `run_pcalm(budget = T)`.
- Batch-size invariance of per-sample activities.

## 6. Timing harness (`pcalm/timing.py`, run through `scripts/run_node.py` with `kind: timing`)

At initialisation (representative: fire times do not depend on training state), batch 64,
Fashion-MNIST input dimension, the paper's `eta_h` for the cell (fallback: the depth table),
for each configuration below, JIT-compile, warm up 3 calls, then time 20 calls with
`block_until_ready` and report the median and interquartile range in ms per batch:

- dense PC-ALM fixed `T = 2L` (`run_pcalm`)
- dense layer-wise freeze (`run_pcalm_layerwise`, masked)
- sparse freeze gate
- sparse wavefront gate

Cells: L = 64 at widths 32, 128, 512; and N = 32 at L = 128. Also record steps, active and
executed layer-cycle fractions, and the ratio of wall-clock to the dense fixed schedule. Output:
`TIMING` lines per configuration, a final `RESULT` line, and `timing.csv` in the node's output
folder. Note in the report that CPU (Apple M4 Max) was measured and that GPU behaviour differs
because small kernels are launch-bound.

## 7. Experiment nodes (round 5, siblings under the phase-2 winner)

- This node: implementation + timing harness; its run is the timing sweep.
- Sparse wavefront, N = 32, L = 64, 1 epoch, 3 seeds: accuracy must equal the dense phase-2
  winner within seed noise (it is the same algorithm) and the executed fraction is measured over
  real training.
- L = 128 cell, N = 32, 1 epoch, 3 seeds: BP, PC-ALM `T = 2L`, sparse wavefront. Tests the
  scaling prediction and whether accuracy still matches at the paper's deepest depth.

## 8. Out of scope

GPU timing, per-sample gating, changes to the trigger itself, Gauss-Seidel (sequential) layer
updates.
