# Phase 2 design: per-layer freeze-and-fire for PC-ALM

Date: 2026-09-14. Status: DRAFT for discussion, not yet approved. Builds on the phase-1 report
(`../adaptive-budget-phase1/adaptive-budget-phase1-report.md`).

## 1. What phase 1 tells us to build

Phase 1 showed that a global trigger on the composite credit `g_i = lambda_i + rho r_i` fires at a
fixed step of about 0.85 L / sqrt(alpha eta_h) regardless of batch or training state, so the only
global gain is a constant 15 to 20% budget reduction. The trace of `||g_i||` against inference
cycle (phase-1 Figure 2b) shows where the rest of the compute goes: the credit wave travels one
layer per 1/sqrt(alpha eta_h) cycles from the output side to the input side; a layer's credit
rises over about 20 cycles and then stays flat while the wave continues toward the input. At
N = 32, L = 64 the output-side layer is flat from cycle 10 to cycle 190, but the loop keeps
recomputing its primal and dual step for all of them.

Phase 2 makes the stopping decision per layer. Each layer watches only its own credit, and when
that credit has arrived and stabilised the layer (a) fixes the credit it will hand to its weight
update and (b) stops participating in inference. The global loop ends when every layer has fired,
or at a hard cap.

## 2. Algorithm (freeze-and-fire)

State per layer `i = 1..L-1`: activities `h_i`, multipliers `lambda_i`, previous credit
`g_i^prev`, patience counter `s_i`, flag `fired_i`, frozen credit `G_i`.

Each inference cycle `t`:

1. For every layer with `fired_i = False`: primal step `h_i <- h_i - eta_h grad_{h_i} L_rho`.
   Layers with `fired_i = True` keep `h_i` fixed.
2. For every layer: recompute `r_i = h_i - sigma(W_i h_{i-1})` (needed by neighbours) and
   `g_i = lambda_i + rho r_i`.
3. Per-layer statistic `d_i = ||g_i - g_i^prev|| / (||g_i|| + eps)`; `s_i <- s_i + 1` if
   `d_i < tau` else `0`.
4. Fire layer `i` when `not fired_i`, `t >= t_min`, `||g_i|| > eps_arrive`, `s_i >= patience`:
   set `fired_i = True`, `G_i = g_i`.
5. Dual step for unfired layers only: `lambda_i <- lambda_i + alpha r_i`.
6. Stop when all `fired_i`, or `t = t_max`. Unfired layers at `t_max` take `G_i = g_i`.

Weight update: `grad_{W_i}` of the augmented Lagrangian evaluated with `G_i` in place of
`lambda_i + rho r_i`, i.e. exactly the paper's update with each layer's credit frozen at its own
fire time. The output layer `W_L` uses the supervised error at the final `h_{L-1}`.

Locality. Every quantity in steps 3 to 5 is local to layer `i`. The arrival guard is now purely
local: before the wave reaches layer `i`, `h_i` equals its forward value and `lambda_i = 0`, so
`g_i` is exactly zero, and the relative-change statistic is undefined or large while `g_i` is
growing. `eps_arrive` is only a numerical floor. The one non-local element left in the whole
scheme is the global cap `t_max`.

## 3. Why freezing rather than only firing

Two variants are worth separating because they answer different questions:

- **fire-only** (ablation): layer `i` snapshots `G_i` at its fire time but keeps running its
  primal and dual steps. Tests whether the early snapshot is a good gradient; saves no inference
  compute.
- **freeze-and-fire** (main): as in section 2. Saves the frozen layer's primal gradient (the
  `W_{i+1}^T` matmul and the update) and dual step for the rest of the loop. The `W_i h_{i-1}`
  matmul is still needed while `h_{i-1}` moves, so a frozen layer costs about half of an active
  one. A useful side effect: the reflected wave seen after 2L in phase 1 cannot form, because the
  layers it would travel through are frozen.

## 4. Compute accounting

Wall-clock on a dense JAX implementation will not drop, because masked layers are still computed.
The primary compute metric is therefore **active layer-cycles**: the sum over cycles of the number
of unfired layers, normalised by `(L-1) * 2L` (the cost of the paper's fixed schedule). Under the
linear wave picture the wavefront crosses layer `i` at cycle `(L - i) / sqrt(alpha eta_h)`, so
active layer-cycles form a triangle of area about half the rectangle: the expected saving is
about 50% of the 2L schedule, versus 15 to 20% for the global trigger. Wall-clock is reported too,
as a secondary number, together with a count of the matmuls a sparse implementation would skip.

## 5. Risks and how the experiments address them

1. **Freezing perturbs neighbours.** Layer `i-1` receives its credit through
   `W_i^T diag(sigma') (rho r_i + lambda_i)`. Once layer `i` is frozen, `lambda_i` stops
   integrating and only `rho r_i` (through the moving `h_{i-1}`) still responds. If that starves
   layer `i-1`, accuracy drops in the deep cells. The fire-only ablation isolates this: if
   fire-only matches the baseline but freeze-and-fire does not, freezing is the culprit and the
   fallback is to freeze `h_i` but keep integrating `lambda_i`.
2. **Snapshot too early.** The credit plateau after arrival is what the weight update gets; the
   later overshoot is discarded. Phase 1 suggests the overshoot does not help accuracy, but only
   at one epoch. Compare against PC-ALM at `T = 2L` and at the phase-1 constant `T = 1.75L`.
3. **Patience and tau.** Reuse phase-1 values `patience = 3`, `tau in {0.03, 0.1}`; a per-layer
   statistic is not diluted by settled layers, so tau should matter more here than it did in
   phase 1. `t_min = 2`, `t_max = 3L`.
4. **Batch semantics.** As in phase 1, the fire decision is shared across the batch.

## 6. Experiments (round 3 of the tree, under the phase-1 winner)

Cells (32, 32) and (32, 64), 3 seeds, one epoch, plus the (32, 64) cell for 5 epochs if the
round-2 multi-epoch check shows the alignment gap matters.

Nodes per cell:
- freeze-and-fire, `tau = 0.1`; freeze-and-fire, `tau = 0.03`
- fire-only, `tau = 0.1` (ablation)
- freeze `h` only, keep `lambda` integrating, `tau = 0.1` (fallback variant, run only if freeze-and-fire loses accuracy)
- baselines already exist: BP, PC-ALM `T = 2L`, adaptive global `tau = 0.1`

Metrics per run: test accuracy, gradient cosine to BP, mean inference cycles per batch, active
layer-cycles as a fraction of `(L-1) * 2L`, and the per-layer mean fire time (to compare with the
wave prediction `(L - i) / sqrt(alpha eta_h)`).

Figures: accuracy against active layer-cycles for all methods (the phase-1 figure with the new
x-axis); per-layer fire time against layer index with the wave prediction as a line.

## 7. Implementation notes

- New family `pcalm_layerwise` in `pcalm/inference.py`, sharing `_solve_inner`'s energy
  gradient but masking the per-layer update with `fired_i`. The gradient is still computed for
  all layers on a dense implementation; the mask only zeroes the update. Correct compute
  accounting comes from the counter, not from wall-clock.
- The dual step is masked the same way. `G_i` is carried in the loop state and returned in place
  of `duals` for the weight gradient (the `al_energy_shifted` interface takes `duals`; pass
  `G_i - rho r_i(final)` so the shifted energy reproduces `G_i`, or add a direct credit path).
- Unit tests: with `tau = 0` (never fires) the result equals `run_pcalm(budget = t_max)`; with
  all layers forced to fire at `t = 1` the weight gradient equals the PC-ALM gradient at
  `T = 1`; fire times are non-increasing in layer index on a linear network at initialisation.
