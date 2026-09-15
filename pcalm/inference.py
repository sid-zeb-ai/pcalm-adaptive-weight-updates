from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .model import Params, block_pred, forward


@dataclass(frozen=True)
class Schedule:
    family: str
    budget: int
    alpha: float = 0.0
    inner_steps: int = 1
    weight_credit_timing: str = "pre_dual_energy"
    # Adaptive-budget fields (family == "pcalm_adaptive"); ignored otherwise.
    tau: float = 0.0
    patience: int = 3
    t_min: int = 2
    t_max: int = 0
    arrival_frac: float = 0.1
    criterion: str = "sum"
    # Per-layer freeze-and-fire fields (family == "pcalm_layerwise"); ignored otherwise.
    eps_arrive: float = 1e-3
    mode: str = "freeze"


ADAPTIVE_EPS = 1e-12
ADAPTIVE_CRITERIA = ("sum", "max")
LAYERWISE_MODES = ("freeze", "fire_only", "freeze_h")


def supervised_loss(params: Params, scales, skips, x, y, free, phi) -> jax.Array:
    y_pred = block_pred(params[-1], scales[-1], skips[-1], free[-1], phi, is_first=False)
    return 0.5 * jnp.mean(jnp.sum((y_pred - y) ** 2, axis=-1))


def bp_loss(params: Params, scales, skips, x, y, phi) -> jax.Array:
    y_pred = forward(params, scales, skips, x, phi)[-1]
    return 0.5 * jnp.mean(jnp.sum((y_pred - y) ** 2, axis=-1))


def free_init(params: Params, scales, skips, x, phi) -> list[jax.Array]:
    return forward(params, scales, skips, x, phi)[:-1]


def constraint_residuals(params: Params, scales, skips, x, free, phi) -> list[jax.Array]:
    """Hidden model-edge constraints only; the label residual is not constrained."""
    residuals = []
    for layer_ix, z_l in enumerate(free):
        z_prev = x if layer_ix == 0 else free[layer_ix - 1]
        pred = block_pred(
            params[layer_ix],
            scales[layer_ix],
            skips[layer_ix],
            z_prev,
            phi,
            is_first=(layer_ix == 0),
        )
        residuals.append(z_l - pred)
    return residuals


def zero_duals_like(residuals: list[jax.Array]) -> list[jax.Array]:
    return [jnp.zeros_like(c) for c in residuals]


def al_energy_shifted(params: Params, scales, skips, x, y, free, duals, rho: float, phi) -> jax.Array:
    residuals = constraint_residuals(params, scales, skips, x, free, phi)
    total = supervised_loss(params, scales, skips, x, y, free, phi)
    batch_size = x.shape[0]
    for residual, dual in zip(residuals, duals):
        shifted = residual + dual / rho
        total = total + 0.5 * rho * jnp.sum(shifted * shifted) / batch_size
    return total


def run_pc(params: Params, scales, skips, x, y, *, state_lr: float, rho: float, steps: int, phi):
    free0 = free_init(params, scales, skips, x, phi)
    duals0 = zero_duals_like(constraint_residuals(params, scales, skips, x, free0, phi))
    return _solve_inner(params, scales, skips, x, y, free0, duals0, state_lr, rho, steps, phi), duals0


def run_pcalm(
    params: Params,
    scales,
    skips,
    x,
    y,
    *,
    state_lr: float,
    rho: float,
    alpha: float,
    budget: int,
    inner_steps: int,
    weight_credit_timing: str,
    phi,
):
    if budget < 1:
        raise ValueError("PC-ALM budget must be at least 1")
    if inner_steps < 1:
        raise ValueError("PC-ALM inner_steps must be at least 1")
    if weight_credit_timing not in {"pre_dual_energy", "post_dual_energy"}:
        raise ValueError("weight_credit_timing must be pre_dual_energy or post_dual_energy")

    free = free_init(params, scales, skips, x, phi)
    duals = zero_duals_like(constraint_residuals(params, scales, skips, x, free, phi))

    def outer(carry, _):
        free_c, duals_c = carry
        free_c = _solve_inner(params, scales, skips, x, y, free_c, duals_c, state_lr, rho, inner_steps, phi)
        residuals = constraint_residuals(params, scales, skips, x, free_c, phi)
        duals_next = [lam + alpha * r for lam, r in zip(duals_c, residuals)]
        return (free_c, duals_next), None

    if budget > 1:
        (free, duals_before), _ = jax.lax.scan(outer, (free, duals), xs=None, length=budget - 1)
    else:
        duals_before = duals

    free = _solve_inner(params, scales, skips, x, y, free, duals_before, state_lr, rho, inner_steps, phi)
    residuals = constraint_residuals(params, scales, skips, x, free, phi)
    duals_after = [lam + alpha * r for lam, r in zip(duals_before, residuals)]
    duals_weight = duals_before if weight_credit_timing == "pre_dual_energy" else duals_after
    return free, duals_weight


def _fro(a: jax.Array) -> jax.Array:
    return jnp.sqrt(jnp.sum(a * a))


def _adaptive_cycle(params, scales, skips, x, y, free, duals, g_prev, *, state_lr, rho, alpha, inner_steps, phi, criterion="sum"):
    """One primal step followed by the stop statistics for the credit the weight update would see.

    Returns the updated activities, the residuals at those activities, the composite credit
    g_i = lambda_i + rho r_i (with lambda_i *before* this cycle's dual step, matching the
    `pre_dual_energy` weight-credit timing), and the batch relative change delta_t of g:
      criterion="sum":  sum_i ||g_i - g_i^prev|| / (sum_i ||g_i|| + eps)
      criterion="max":  max_i ||g_i - g_i^prev|| / (max_i ||g_i|| + eps)
    The "max" form is sensitive to a single layer that is still moving (e.g. the input side
    while the credit wave is still arriving), which the layer-summed form averages away.
    """
    if criterion not in ADAPTIVE_CRITERIA:
        raise ValueError(f"criterion must be one of {ADAPTIVE_CRITERIA}, got {criterion!r}")
    free = _solve_inner(params, scales, skips, x, y, free, duals, state_lr, rho, inner_steps, phi)
    residuals = constraint_residuals(params, scales, skips, x, free, phi)
    credit = [lam + rho * r for lam, r in zip(duals, residuals)]
    change = jnp.stack([_fro(g - gp) for g, gp in zip(credit, g_prev)])
    size = jnp.stack([_fro(g) for g in credit])
    if criterion == "sum":
        delta = jnp.sum(change) / (jnp.sum(size) + ADAPTIVE_EPS)
    else:
        delta = jnp.max(change) / (jnp.max(size) + ADAPTIVE_EPS)
    return free, residuals, credit, delta


def run_pcalm_adaptive(
    params: Params,
    scales,
    skips,
    x,
    y,
    *,
    state_lr: float,
    rho: float,
    alpha: float,
    tau: float,
    patience: int,
    t_min: int,
    t_max: int,
    arrival_frac: float,
    inner_steps: int,
    phi,
    criterion: str = "sum",
):
    """PC-ALM whose inference budget is chosen per batch by a credit-stability trigger.

    Cycle t = 1, 2, ...: primal step, then compute delta_t from the composite credit. Stop
    (skipping the dual step, so this cycle is the paper's "final primal step") when
      t >= t_min, the input-side credit has arrived (||g_1|| >= arrival_frac * ||g_{L-1}||),
      and delta_t < tau for `patience` consecutive cycles;
    or unconditionally at t == t_max. With tau = 0 this reproduces `run_pcalm(budget=t_max)`
    under `weight_credit_timing="pre_dual_energy"`. Returns (free, duals, steps).
    """
    if t_max < 1:
        raise ValueError("PC-ALM adaptive t_max must be at least 1")
    if t_min < 1:
        raise ValueError("PC-ALM adaptive t_min must be at least 1")
    if patience < 1:
        raise ValueError("PC-ALM adaptive patience must be at least 1")

    free0 = free_init(params, scales, skips, x, phi)
    duals0 = zero_duals_like(constraint_residuals(params, scales, skips, x, free0, phi))
    g0 = zero_duals_like(duals0)
    t0 = jnp.asarray(0, dtype=jnp.int32)
    carry0 = (free0, duals0, g0, t0, jnp.asarray(0, dtype=jnp.int32), jnp.asarray(False))

    def cond(carry):
        return jnp.logical_not(carry[5])

    def body(carry):
        free_c, duals_c, g_prev, t, streak, _ = carry
        t = t + 1
        free_c, residuals, credit, delta = _adaptive_cycle(
            params, scales, skips, x, y, free_c, duals_c, g_prev,
            state_lr=state_lr, rho=rho, alpha=alpha, inner_steps=inner_steps, phi=phi, criterion=criterion,
        )
        streak = jnp.where(delta < tau, streak + 1, 0)
        arrived = _fro(credit[0]) >= arrival_frac * _fro(credit[-1])
        fire = (t >= t_min) & arrived & (streak >= patience)
        done = fire | (t >= t_max)
        # On the stopping cycle the dual step is skipped: this cycle is the final primal step.
        duals_next = [jnp.where(done, lam, lam + alpha * r) for lam, r in zip(duals_c, residuals)]
        return (free_c, duals_next, credit, t, streak, done)

    free, duals, _, steps, _, _ = jax.lax.while_loop(cond, body, carry0)
    return free, duals, steps


def trace_pcalm_adaptive(
    params: Params,
    scales,
    skips,
    x,
    y,
    *,
    state_lr: float,
    rho: float,
    alpha: float,
    t_max: int,
    inner_steps: int,
    phi,
):
    """Run exactly t_max primal-dual cycles and record per-cycle diagnostics.

    Returns a dict of arrays indexed by cycle t = 1..t_max: `delta` (T,) for the "sum"
    criterion, `delta_max` (T,) for the "max" criterion, and per-layer Frobenius norms
    `credit_norm`, `residual_norm`, `dual_norm`, each (T, L-1). The dual norm is that of
    lambda *before* the cycle's dual step, matching the weight-credit timing. Used for the
    diagnostic trace only; the trigger itself lives in `run_pcalm_adaptive`.
    """
    free0 = free_init(params, scales, skips, x, phi)
    duals0 = zero_duals_like(constraint_residuals(params, scales, skips, x, free0, phi))
    g0 = zero_duals_like(duals0)

    def step(carry, _):
        free_c, duals_c, g_prev = carry
        free_c, residuals, credit, delta = _adaptive_cycle(
            params, scales, skips, x, y, free_c, duals_c, g_prev,
            state_lr=state_lr, rho=rho, alpha=alpha, inner_steps=inner_steps, phi=phi,
        )
        change = jnp.stack([_fro(g - gp) for g, gp in zip(credit, g_prev)])
        size = jnp.stack([_fro(g) for g in credit])
        out = {
            "delta": delta,
            "delta_max": jnp.max(change) / (jnp.max(size) + ADAPTIVE_EPS),
            "credit_norm": size,
            "residual_norm": jnp.stack([_fro(r) for r in residuals]),
            "dual_norm": jnp.stack([_fro(lam) for lam in duals_c]),
        }
        duals_next = [lam + alpha * r for lam, r in zip(duals_c, residuals)]
        return (free_c, duals_next, credit), out

    _, trace = jax.lax.scan(step, (free0, duals0, g0), xs=None, length=t_max)
    return trace


def run_pcalm_layerwise(
    params: Params,
    scales,
    skips,
    x,
    y,
    *,
    state_lr: float,
    rho: float,
    alpha: float,
    tau: float,
    patience: int,
    t_min: int,
    t_max: int,
    eps_arrive: float,
    mode: str,
    inner_steps: int,
    phi,
):
    """Per-layer freeze-and-fire PC-ALM (phase-2 design).

    Each of the L-1 hidden layers watches its own credit `g_i = lambda_i + rho r_i` and fires
    independently once that credit has arrived and stabilised (relative change < tau for
    `patience` consecutive cycles, and t >= t_min). A fired layer's credit is frozen at `G_i`
    for the weight update. In mode "freeze" and "freeze_h" a fired layer's primal step is
    skipped; "freeze_h" still applies the dual step to fired layers, "freeze" does not;
    "fire_only" never skips the primal or dual step (an ablation that saves no compute but still
    freezes the credit used for the weight gradient).

    Returns `(free, duals_eff, steps, info)` where `duals_eff_i = G_i - rho * r_i(final)` so that
    `al_energy_shifted(..., free, duals_eff, rho, ...)` has weight-gradient contribution
    `-(rho * r_i + duals_eff_i) = -G_i` in place of `lambda_i + rho r_i`, and `info` carries
    `fire_times` (int32, shape (L-1,)) and `active_layer_cycles` (int32 scalar).
    """
    if mode not in LAYERWISE_MODES:
        raise ValueError(f"mode must be one of {LAYERWISE_MODES}, got {mode!r}")
    if t_max < 1:
        raise ValueError("PC-ALM layerwise t_max must be at least 1")
    if t_min < 1:
        raise ValueError("PC-ALM layerwise t_min must be at least 1")
    if patience < 1:
        raise ValueError("PC-ALM layerwise patience must be at least 1")

    free0 = free_init(params, scales, skips, x, phi)
    duals0 = zero_duals_like(constraint_residuals(params, scales, skips, x, free0, phi))
    n_layers = len(free0)
    fired0 = [jnp.asarray(False) for _ in range(n_layers)]
    streak0 = [jnp.asarray(0, dtype=jnp.int32) for _ in range(n_layers)]
    g_prev0 = zero_duals_like(duals0)
    G0 = zero_duals_like(duals0)
    fire_time0 = [jnp.asarray(t_max, dtype=jnp.int32) for _ in range(n_layers)]
    t0 = jnp.asarray(0, dtype=jnp.int32)
    active0 = jnp.asarray(0, dtype=jnp.int32)
    done0 = jnp.asarray(False)

    carry0 = (free0, duals0, g_prev0, G0, fired0, streak0, fire_time0, t0, active0, done0)

    def cond(carry):
        return jnp.logical_not(carry[-1])

    def primal_step(free_, duals_, fired):
        """One gradient step of the shifted AL energy w.r.t. all free activities, matching
        `_solve_inner`'s effective learning rate; a fired layer's update is masked to zero in
        modes that skip the primal step (dense compute, `jnp.where` masking only)."""

        def energy(free_c):
            return al_energy_shifted(params, scales, skips, x, y, free_c, duals_, rho, phi)

        grads = jax.grad(energy)(free_)
        effective_lr = state_lr * free_[0].shape[0]
        skip_primal = mode in ("freeze", "freeze_h")
        new_free = []
        for z, g, is_fired in zip(free_, grads, fired):
            updated = z - effective_lr * g
            masked = jnp.where(is_fired & skip_primal, z, updated) if skip_primal else updated
            new_free.append(masked)
        return new_free

    def body(carry):
        free_c, duals_c, g_prev, G_c, fired, streak, fire_time, t, active, _ = carry
        t = t + 1

        free_next = free_c
        for _ in range(inner_steps):
            free_next = primal_step(free_next, duals_c, fired)

        residuals = constraint_residuals(params, scales, skips, x, free_next, phi)
        credit = [lam + rho * r for lam, r in zip(duals_c, residuals)]

        change = [_fro(g - gp) for g, gp in zip(credit, g_prev)]
        size = [_fro(g) for g in credit]
        d = [c / (s + ADAPTIVE_EPS) for c, s in zip(change, size)]
        streak = [jnp.where(dd < tau, s + 1, 0) for dd, s in zip(d, streak)]

        # Layers whose primal step ran this cycle: those not already fired at the start of it.
        skip_primal = mode in ("freeze", "freeze_h")
        if skip_primal:
            active = active + sum(jnp.where(f, 0, 1) for f in fired)
        else:
            active = active + n_layers

        fire_now = [
            (~f) & (t >= t_min) & (s_fro >= eps_arrive) & (s_streak >= patience)
            for f, s_fro, s_streak in zip(fired, size, streak)
        ]
        G_c = [jnp.where(fn, g, G) for fn, g, G in zip(fire_now, credit, G_c)]
        fire_time = [jnp.where(fn, t, ft) for fn, ft in zip(fire_now, fire_time)]
        fired = [f | fn for f, fn in zip(fired, fire_now)]

        done = jnp.stack(fired).all() | (t >= t_max)
        # Layers still unfired at the stopping cycle take their current credit.
        G_c = [jnp.where(done & ~f, g, G) for f, g, G in zip(fired, credit, G_c)]
        fire_time = [jnp.where(done & ~f, t, ft) for f, ft in zip(fired, fire_time)]

        # Dual step for layers that are not (fired and mode == "freeze"); skipped entirely on the
        # done cycle (the final cycle is a primal step only, matching run_pcalm_adaptive).
        skip_dual_when_fired = mode == "freeze"
        duals_next = []
        for lam, r, f in zip(duals_c, residuals, fired):
            stepped = lam + alpha * r
            kept = jnp.where(f & skip_dual_when_fired, lam, stepped)
            duals_next.append(jnp.where(done, lam, kept))

        return (free_next, duals_next, credit, G_c, fired, streak, fire_time, t, active, done)

    free, duals_before, credit_final, G_final, fired_final, _, fire_time_final, steps, active_final, _ = (
        jax.lax.while_loop(cond, body, carry0)
    )

    residuals_final = constraint_residuals(params, scales, skips, x, free, phi)
    duals_eff = [g - rho * r for g, r in zip(G_final, residuals_final)]
    info = {
        "fire_times": jnp.stack(fire_time_final).astype(jnp.int32),
        "active_layer_cycles": active_final.astype(jnp.int32),
    }
    return free, duals_eff, steps, info


def _no_layerwise_info(n_layers: int, steps) -> dict[str, jax.Array]:
    return {
        "fire_times": jnp.zeros((n_layers,), dtype=jnp.int32),
        "active_layer_cycles": (jnp.asarray(n_layers, dtype=jnp.int32) * steps).astype(jnp.int32),
    }


def infer_for_schedule(params: Params, scales, skips, x, y, schedule: Schedule, *, state_lr: float, rho: float, phi):
    """Returns (free, duals, steps, info); `steps` is the number of primal steps actually taken.

    `info` carries `fire_times` (int32, shape (L-1,)) and `active_layer_cycles` (int32 scalar);
    for families other than `pcalm_layerwise`, `fire_times` is all zeros and
    `active_layer_cycles = (L - 1) * steps` (every layer is active every cycle).
    """
    n_layers = len(params) - 1
    if schedule.family == "pc":
        free, duals = run_pc(params, scales, skips, x, y, state_lr=state_lr, rho=rho, steps=schedule.budget, phi=phi)
        steps = jnp.asarray(schedule.budget, dtype=jnp.int32)
        return free, duals, steps, _no_layerwise_info(n_layers, steps)
    if schedule.family == "pcalm":
        free, duals = run_pcalm(
            params,
            scales,
            skips,
            x,
            y,
            state_lr=state_lr,
            rho=rho,
            alpha=schedule.alpha,
            budget=schedule.budget,
            inner_steps=schedule.inner_steps,
            weight_credit_timing=schedule.weight_credit_timing,
            phi=phi,
        )
        steps = jnp.asarray(schedule.budget, dtype=jnp.int32)
        return free, duals, steps, _no_layerwise_info(n_layers, steps)
    if schedule.family == "pcalm_adaptive":
        t_max = schedule.t_max if schedule.t_max > 0 else schedule.budget
        free, duals, steps = run_pcalm_adaptive(
            params,
            scales,
            skips,
            x,
            y,
            state_lr=state_lr,
            rho=rho,
            alpha=schedule.alpha,
            tau=schedule.tau,
            patience=schedule.patience,
            t_min=schedule.t_min,
            t_max=t_max,
            arrival_frac=schedule.arrival_frac,
            inner_steps=schedule.inner_steps,
            phi=phi,
            criterion=schedule.criterion,
        )
        return free, duals, steps, _no_layerwise_info(n_layers, steps)
    if schedule.family == "pcalm_layerwise":
        return run_pcalm_layerwise(
            params,
            scales,
            skips,
            x,
            y,
            state_lr=state_lr,
            rho=rho,
            alpha=schedule.alpha,
            tau=schedule.tau,
            patience=schedule.patience,
            t_min=schedule.t_min,
            t_max=schedule.t_max if schedule.t_max > 0 else schedule.budget,
            eps_arrive=schedule.eps_arrive,
            mode=schedule.mode,
            inner_steps=schedule.inner_steps,
            phi=phi,
        )
    raise ValueError(f"unknown schedule family: {schedule.family}")


def method_grad_and_steps(params: Params, scales, skips, x, y, schedule: Schedule, *, state_lr: float, rho: float, phi):
    if schedule.family == "bp":
        grads = jax.grad(lambda p: bp_loss(p, scales, skips, x, y, phi))(params)
        steps = jnp.asarray(0, dtype=jnp.int32)
        return grads, steps, _no_layerwise_info(len(params) - 1, jnp.asarray(0, dtype=jnp.int32))
    free, duals, steps, info = infer_for_schedule(params, scales, skips, x, y, schedule, state_lr=state_lr, rho=rho, phi=phi)
    free = jax.tree_util.tree_map(jax.lax.stop_gradient, free)
    duals = jax.tree_util.tree_map(jax.lax.stop_gradient, duals)
    grads = jax.grad(lambda p: al_energy_shifted(p, scales, skips, x, y, free, duals, rho, phi))(params)
    return grads, steps, info


def method_grad(params: Params, scales, skips, x, y, schedule: Schedule, *, state_lr: float, rho: float, phi):
    grads, _, _ = method_grad_and_steps(params, scales, skips, x, y, schedule, state_lr=state_lr, rho=rho, phi=phi)
    return grads


def _solve_inner(params: Params, scales, skips, x, y, free, duals, state_lr: float, rho: float, steps: int, phi):
    def energy(free_):
        return al_energy_shifted(params, scales, skips, x, y, free_, duals, rho, phi)

    grad_free = jax.grad(energy)

    # `al_energy_shifted` is a *mean over the batch* (the constraint terms divide
    # by batch_size and the supervised loss is a mean), so its gradient w.r.t. a
    # single sample's activity is 1/batch_size of the per-sample gradient. The
    # paper's activity step is the per-sample eta_h = 1/lambda_max; we recover it
    # by scaling the step by the runtime batch size (an effective step of
    # `state_lr * batch_size` on the batch-mean energy). Without this, finite-T PC / PC-ALM
    # inference runs batch_size-times too slowly and never nears its fixed point.
    effective_lr = state_lr * free[0].shape[0]

    def step(free_, _):
        grads = grad_free(free_)
        return [z - effective_lr * g for z, g in zip(free_, grads)], None

    if steps <= 0:
        return free
    free, _ = jax.lax.scan(step, free, xs=None, length=steps)
    return free
