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


ADAPTIVE_EPS = 1e-12
ADAPTIVE_CRITERIA = ("sum", "max")


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


def infer_for_schedule(params: Params, scales, skips, x, y, schedule: Schedule, *, state_lr: float, rho: float, phi):
    """Returns (free, duals, steps); `steps` is the number of primal steps actually taken."""
    if schedule.family == "pc":
        free, duals = run_pc(params, scales, skips, x, y, state_lr=state_lr, rho=rho, steps=schedule.budget, phi=phi)
        return free, duals, jnp.asarray(schedule.budget, dtype=jnp.int32)
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
        return free, duals, jnp.asarray(schedule.budget, dtype=jnp.int32)
    if schedule.family == "pcalm_adaptive":
        t_max = schedule.t_max if schedule.t_max > 0 else schedule.budget
        return run_pcalm_adaptive(
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
    raise ValueError(f"unknown schedule family: {schedule.family}")


def method_grad_and_steps(params: Params, scales, skips, x, y, schedule: Schedule, *, state_lr: float, rho: float, phi):
    if schedule.family == "bp":
        grads = jax.grad(lambda p: bp_loss(p, scales, skips, x, y, phi))(params)
        return grads, jnp.asarray(0, dtype=jnp.int32)
    free, duals, steps = infer_for_schedule(params, scales, skips, x, y, schedule, state_lr=state_lr, rho=rho, phi=phi)
    free = jax.tree_util.tree_map(jax.lax.stop_gradient, free)
    duals = jax.tree_util.tree_map(jax.lax.stop_gradient, duals)
    grads = jax.grad(lambda p: al_energy_shifted(p, scales, skips, x, y, free, duals, rho, phi))(params)
    return grads, steps


def method_grad(params: Params, scales, skips, x, y, schedule: Schedule, *, state_lr: float, rho: float, phi):
    grads, _ = method_grad_and_steps(params, scales, skips, x, y, schedule, state_lr=state_lr, rho=rho, phi=phi)
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
