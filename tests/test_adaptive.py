from __future__ import annotations

import jax
import jax.numpy as jnp

from pcalm.inference import (
    Schedule,
    method_grad,
    method_grad_and_steps,
    run_pcalm,
    run_pcalm_adaptive,
    trace_pcalm_adaptive,
)
from pcalm.model import activation_fn, init_params, model_scales, skip_mask


def small_case(depth: int = 6, width: int = 5, n: int = 7):
    params = init_params(jax.random.PRNGKey(0), depth=depth, width=width, input_dim=3, output_dim=2)
    scales = model_scales(width=width, depth=depth, input_dim=3)
    skips = skip_mask(depth)
    phi = activation_fn("tanh")
    x = jax.random.normal(jax.random.PRNGKey(1), (n, 3))
    y = jax.nn.one_hot(jnp.arange(n) % 2, 2)
    return params, scales, skips, phi, x, y


COMMON = dict(state_lr=0.1, rho=1.0, alpha=1.0, inner_steps=1)


def test_tau_zero_reproduces_fixed_budget_pcalm():
    params, scales, skips, phi, x, y = small_case()
    T = 9
    free_ref, duals_ref = run_pcalm(
        params, scales, skips, x, y, budget=T, weight_credit_timing="pre_dual_energy", phi=phi, **COMMON
    )
    free_ad, duals_ad, steps = run_pcalm_adaptive(
        params, scales, skips, x, y, tau=0.0, patience=1, t_min=1, t_max=T, arrival_frac=0.0, phi=phi, **COMMON
    )
    assert int(steps) == T
    for a, b in zip(free_ref, free_ad):
        assert jnp.allclose(a, b, atol=1e-5, rtol=1e-5)
    for a, b in zip(duals_ref, duals_ad):
        assert jnp.allclose(a, b, atol=1e-5, rtol=1e-5)

    g_ref = method_grad(params, scales, skips, x, y, Schedule(family="pcalm", budget=T, alpha=1.0), state_lr=0.1, rho=1.0, phi=phi)
    g_ad, steps_ad = method_grad_and_steps(
        params, scales, skips, x, y,
        Schedule(family="pcalm_adaptive", budget=T, alpha=1.0, tau=0.0, patience=1, t_min=1, t_max=T, arrival_frac=0.0),
        state_lr=0.1, rho=1.0, phi=phi,
    )
    assert int(steps_ad) == T
    for a, b in zip(g_ref, g_ad):
        assert jnp.allclose(a, b, atol=1e-5, rtol=1e-5)


def test_large_tau_stops_at_t_min_plus_patience():
    params, scales, skips, phi, x, y = small_case()
    # tau above any possible relative change (delta_t <= ~2), no arrival guard:
    # streak reaches `patience` at t = patience, so the stop is at max(t_min, patience).
    _, _, steps = run_pcalm_adaptive(
        params, scales, skips, x, y, tau=10.0, patience=3, t_min=5, t_max=50, arrival_frac=0.0, phi=phi, **COMMON
    )
    assert int(steps) == 5
    _, _, steps = run_pcalm_adaptive(
        params, scales, skips, x, y, tau=10.0, patience=3, t_min=1, t_max=50, arrival_frac=0.0, phi=phi, **COMMON
    )
    assert int(steps) == 3


def test_arrival_guard_delays_stop():
    params, scales, skips, phi, x, y = small_case(depth=10)
    kw = dict(tau=10.0, patience=1, t_min=1, t_max=60, phi=phi, **COMMON)
    _, _, steps_free = run_pcalm_adaptive(params, scales, skips, x, y, arrival_frac=0.0, **kw)
    _, _, steps_guarded = run_pcalm_adaptive(params, scales, skips, x, y, arrival_frac=0.5, **kw)
    assert int(steps_free) == 1
    assert int(steps_guarded) > int(steps_free)
    assert int(steps_guarded) <= 60


def test_stop_respects_cap_and_bounds():
    params, scales, skips, phi, x, y = small_case()
    _, _, steps = run_pcalm_adaptive(
        params, scales, skips, x, y, tau=1e-9, patience=3, t_min=2, t_max=7, arrival_frac=0.1, phi=phi, **COMMON
    )
    assert 2 <= int(steps) <= 7


def test_stop_step_is_batch_size_invariant_per_sample():
    params, scales, skips, phi, x, y = small_case()
    kw = dict(tau=0.0, patience=1, t_min=1, t_max=4, arrival_frac=0.0, phi=phi, **COMMON)
    free_single, _, s1 = run_pcalm_adaptive(params, scales, skips, x[:1], y[:1], **kw)
    free_batch, _, s2 = run_pcalm_adaptive(params, scales, skips, x, y, **kw)
    assert int(s1) == int(s2)
    for a, b in zip(free_single, free_batch):
        assert jnp.allclose(a[0], b[0], atol=1e-5, rtol=1e-5)


def test_trace_matches_loop_statistics():
    params, scales, skips, phi, x, y = small_case()
    T = 6
    trace = trace_pcalm_adaptive(params, scales, skips, x, y, t_max=T, phi=phi, **COMMON)
    assert trace["delta"].shape == (T,)
    assert trace["credit_norm"].shape == (T, len(params) - 1)
    # First cycle: g_prev = 0 so delta_1 == 1 exactly (up to eps).
    assert jnp.allclose(trace["delta"][0], 1.0, atol=1e-5)
    # Dual before the first cycle's dual step is zero.
    assert jnp.allclose(trace["dual_norm"][0], 0.0)
    # Final activities of the trace equal the fixed-budget run at T (both do T primal steps).
    free_ref, _ = run_pcalm(
        params, scales, skips, x, y, budget=T, weight_credit_timing="pre_dual_energy", phi=phi, **COMMON
    )
    free_ad, _, _ = run_pcalm_adaptive(
        params, scales, skips, x, y, tau=0.0, patience=1, t_min=1, t_max=T, arrival_frac=0.0, phi=phi, **COMMON
    )
    for a, b in zip(free_ref, free_ad):
        assert jnp.allclose(a, b, atol=1e-5)
