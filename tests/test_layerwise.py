from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from pcalm.inference import (
    LAYERWISE_MODES,
    Schedule,
    al_energy_shifted,
    constraint_residuals,
    method_grad,
    method_grad_and_steps,
    run_pcalm,
    run_pcalm_layerwise,
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


@pytest.mark.parametrize("mode", LAYERWISE_MODES)
def test_tau_zero_reproduces_fixed_budget_pcalm(mode):
    params, scales, skips, phi, x, y = small_case()
    T = 9
    free_ref, duals_ref = run_pcalm(
        params, scales, skips, x, y, budget=T, weight_credit_timing="pre_dual_energy", phi=phi, **COMMON
    )
    free_lw, duals_eff, steps, info = run_pcalm_layerwise(
        params, scales, skips, x, y, tau=0.0, patience=1, t_min=1, t_max=T, eps_arrive=0.0, mode=mode, phi=phi, **COMMON
    )
    assert int(steps) == T
    for a, b in zip(free_ref, free_lw):
        assert jnp.allclose(a, b, atol=1e-5, rtol=1e-5)

    n_layers = len(params) - 1
    assert int(info["active_layer_cycles"]) == n_layers * T
    assert jnp.all(info["fire_times"] == T)

    g_ref = method_grad(params, scales, skips, x, y, Schedule(family="pcalm", budget=T, alpha=1.0), state_lr=0.1, rho=1.0, phi=phi)
    g_lw, steps_lw, info_lw = method_grad_and_steps(
        params, scales, skips, x, y,
        Schedule(family="pcalm_layerwise", budget=T, alpha=1.0, tau=0.0, patience=1, t_min=1, t_max=T, eps_arrive=0.0, mode=mode),
        state_lr=0.1, rho=1.0, phi=phi,
    )
    assert int(steps_lw) == T
    for a, b in zip(g_ref, g_lw):
        assert jnp.allclose(a, b, atol=1e-5, rtol=1e-5)


def test_forced_fire_at_t1_matches_pcalm_budget1():
    params, scales, skips, phi, x, y = small_case()
    free_lw, duals_eff, steps, info = run_pcalm_layerwise(
        params, scales, skips, x, y, tau=10.0, patience=1, t_min=1, t_max=60, eps_arrive=0.0, mode="freeze",
        phi=phi, **COMMON,
    )
    assert int(steps) == 1
    assert jnp.all(info["fire_times"] == 1)

    g_ref = method_grad(params, scales, skips, x, y, Schedule(family="pcalm", budget=1, alpha=1.0), state_lr=0.1, rho=1.0, phi=phi)
    g_lw, steps_lw, _ = method_grad_and_steps(
        params, scales, skips, x, y,
        Schedule(family="pcalm_layerwise", budget=1, alpha=1.0, tau=10.0, patience=1, t_min=1, t_max=60, eps_arrive=0.0, mode="freeze"),
        state_lr=0.1, rho=1.0, phi=phi,
    )
    assert int(steps_lw) == 1
    for a, b in zip(g_ref, g_lw):
        assert jnp.allclose(a, b, atol=1e-5, rtol=1e-5)


def test_credit_identity_and_gradient_matches_al_energy_shifted():
    params, scales, skips, phi, x, y = small_case(depth=10)
    free, duals_eff, steps, info = run_pcalm_layerwise(
        params, scales, skips, x, y, tau=0.1, patience=2, t_min=2, t_max=60, eps_arrive=1e-3, mode="freeze",
        phi=phi, **COMMON,
    )
    g_direct = jax.grad(lambda p: al_energy_shifted(p, scales, skips, x, y, free, duals_eff, 1.0, phi))(params)
    g_mg, steps_mg, _ = method_grad_and_steps(
        params, scales, skips, x, y,
        Schedule(family="pcalm_layerwise", budget=60, alpha=1.0, tau=0.1, patience=2, t_min=2, t_max=60, eps_arrive=1e-3, mode="freeze"),
        state_lr=0.1, rho=1.0, phi=phi,
    )
    assert int(steps_mg) == int(steps)
    for a, b in zip(g_direct, g_mg):
        assert jnp.allclose(a, b, atol=1e-6, rtol=1e-6)


def test_fire_order_and_savings():
    params, scales, skips, phi, x, y = small_case(depth=10)
    n_layers = len(params) - 1
    free, duals_eff, steps, info = run_pcalm_layerwise(
        params, scales, skips, x, y, tau=0.1, patience=3, t_min=2, t_max=80, eps_arrive=1e-3, mode="freeze",
        phi=phi, **COMMON,
    )
    fire_times = info["fire_times"]
    assert int(fire_times[-1]) <= int(fire_times[0])
    assert jnp.all(fire_times >= 1) and jnp.all(fire_times <= 80)
    assert int(info["active_layer_cycles"]) < n_layers * int(steps)


def test_fire_only_mode_saves_no_compute():
    params, scales, skips, phi, x, y = small_case(depth=10)
    n_layers = len(params) - 1
    free, duals_eff, steps, info = run_pcalm_layerwise(
        params, scales, skips, x, y, tau=0.1, patience=3, t_min=2, t_max=80, eps_arrive=1e-3, mode="fire_only",
        phi=phi, **COMMON,
    )
    assert int(info["active_layer_cycles"]) == n_layers * int(steps)


def test_per_sample_batch_invariance():
    params, scales, skips, phi, x, y = small_case()
    kw = dict(tau=0.0, patience=1, t_min=1, t_max=8, eps_arrive=0.0, mode="freeze", phi=phi, **COMMON)
    free_single, _, s1, _ = run_pcalm_layerwise(params, scales, skips, x[:1], y[:1], **kw)
    free_batch, _, s2, _ = run_pcalm_layerwise(params, scales, skips, x, y, **kw)
    assert int(s1) == int(s2)
    for a, b in zip(free_single, free_batch):
        assert jnp.allclose(a[0], b[0], atol=1e-5, rtol=1e-5)


def test_invalid_mode_rejected():
    params, scales, skips, phi, x, y = small_case()
    with pytest.raises(ValueError):
        run_pcalm_layerwise(
            params, scales, skips, x, y, tau=0.1, patience=1, t_min=1, t_max=3, eps_arrive=0.0, mode="bogus",
            phi=phi, **COMMON,
        )
