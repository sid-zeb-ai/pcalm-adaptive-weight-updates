from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from pcalm.inference import (
    SPARSE_GATES,
    Schedule,
    _layer_local_energy,
    al_energy_shifted,
    free_init,
    method_grad,
    method_grad_and_steps,
    run_pcalm,
    run_pcalm_layerwise,
    run_pcalm_layerwise_sparse,
    zero_duals_like,
)
from pcalm.model import activation_fn, init_params, model_scales, skip_mask


def small_case(depth: int = 6, width: int = 5, n: int = 7, activation: str = "tanh"):
    params = init_params(jax.random.PRNGKey(0), depth=depth, width=width, input_dim=3, output_dim=2)
    scales = model_scales(width=width, depth=depth, input_dim=3)
    skips = skip_mask(depth)
    phi = activation_fn(activation)
    x = jax.random.normal(jax.random.PRNGKey(1), (n, 3))
    y = jax.nn.one_hot(jnp.arange(n) % 2, 2)
    return params, scales, skips, phi, x, y


COMMON = dict(state_lr=0.1, rho=1.0, alpha=1.0, inner_steps=1)


# --- Section 5, bullet 1: per-layer grad E_i equals the dense gradient block. ---


@pytest.mark.parametrize("activation", ["tanh", "relu", "linear"])
def test_layer_local_energy_grad_matches_dense_block(activation):
    params, scales, skips, phi, x, y = small_case(depth=8, activation=activation)
    free = free_init(params, scales, skips, x, phi)
    keys = jax.random.split(jax.random.PRNGKey(2), len(free))
    free = [f + 0.1 * jax.random.normal(k, f.shape) for f, k in zip(free, keys)]
    duals_keys = jax.random.split(jax.random.PRNGKey(3), len(free))
    duals = [0.05 * jax.random.normal(k, f.shape) for f, k in zip(free, duals_keys)]
    rho = 1.3

    dense_grad = jax.grad(lambda fr: al_energy_shifted(params, scales, skips, x, y, fr, duals, rho, phi))(free)

    for layer_ix in range(len(free)):
        g_local = jax.grad(
            lambda h, layer_ix=layer_ix: _layer_local_energy(
                params, scales, skips, x, y, free, duals, layer_ix, h, rho, phi
            )
        )(free[layer_ix])
        assert jnp.allclose(g_local, dense_grad[layer_ix], atol=1e-6, rtol=1e-6)


# --- Section 5, bullet 2 & 3: sparse gates reproduce dense run_pcalm_layerwise(mode="freeze"). ---


@pytest.mark.parametrize("activation", ["tanh", "relu"])
@pytest.mark.parametrize("depth", [10, 4])
@pytest.mark.parametrize("gate", SPARSE_GATES)
def test_sparse_reproduces_dense_layerwise_freeze(activation, depth, gate):
    params, scales, skips, phi, x, y = small_case(depth=depth, activation=activation)
    kw = dict(tau=0.1, patience=3, t_min=2, t_max=60, eps_arrive=1e-3, **COMMON)

    free_ref, duals_ref, steps_ref, info_ref = run_pcalm_layerwise(
        params, scales, skips, x, y, mode="freeze", phi=phi, **kw
    )
    free_s, duals_s, steps_s, info_s = run_pcalm_layerwise_sparse(
        params, scales, skips, x, y, gate=gate, phi=phi, **kw
    )

    assert int(steps_s) == int(steps_ref)
    for a, b in zip(free_ref, free_s):
        assert jnp.allclose(a, b, atol=1e-5, rtol=1e-5)
    for a, b in zip(duals_ref, duals_s):
        assert jnp.allclose(a, b, atol=1e-5, rtol=1e-5)
    assert jnp.all(info_ref["fire_times"] == info_s["fire_times"])

    g_ref, steps_g_ref, _ = method_grad_and_steps(
        params, scales, skips, x, y,
        Schedule(family="pcalm_layerwise", alpha=1.0, budget=60, mode="freeze", **{k: v for k, v in kw.items() if k not in COMMON}),
        state_lr=0.1, rho=1.0, phi=phi,
    )
    g_s, steps_g_s, _ = method_grad_and_steps(
        params, scales, skips, x, y,
        Schedule(family="pcalm_layerwise_sparse", alpha=1.0, budget=60, gate=gate, **{k: v for k, v in kw.items() if k not in COMMON}),
        state_lr=0.1, rho=1.0, phi=phi,
    )
    assert int(steps_g_s) == int(steps_g_ref)
    for a, b in zip(g_ref, g_s):
        assert jnp.allclose(a, b, atol=1e-5, rtol=1e-5)

    if gate == "wavefront":
        assert int(info_s["executed_layer_cycles"]) < int(info_s["active_layer_cycles"])


def test_wavefront_executed_strictly_less_than_active_when_staggered():
    params, scales, skips, phi, x, y = small_case(depth=10)
    free, duals_eff, steps, info = run_pcalm_layerwise_sparse(
        params, scales, skips, x, y, tau=0.1, patience=3, t_min=2, t_max=80, eps_arrive=1e-3, gate="wavefront",
        phi=phi, **COMMON,
    )
    assert int(info["executed_layer_cycles"]) < int(info["active_layer_cycles"])


# --- Section 5, bullet 4: tau=0, t_max=T reproduces run_pcalm(budget=T) for both gates. ---


@pytest.mark.parametrize("gate", SPARSE_GATES)
def test_tau_zero_reproduces_fixed_budget_pcalm(gate):
    params, scales, skips, phi, x, y = small_case()
    T = 9
    free_ref, duals_ref = run_pcalm(
        params, scales, skips, x, y, budget=T, weight_credit_timing="pre_dual_energy", phi=phi, **COMMON
    )
    free_s, duals_s, steps_s, info_s = run_pcalm_layerwise_sparse(
        params, scales, skips, x, y, tau=0.0, patience=1, t_min=1, t_max=T, eps_arrive=0.0, gate=gate, phi=phi,
        **COMMON,
    )
    assert int(steps_s) == T
    for a, b in zip(free_ref, free_s):
        assert jnp.allclose(a, b, atol=1e-5, rtol=1e-5)
    assert jnp.all(info_s["fire_times"] == T)

    n_layers = len(params) - 1
    assert int(info_s["active_layer_cycles"]) == n_layers * T
    if gate == "freeze":
        # No layer ever fires before t_max here, so freeze runs every layer every cycle.
        assert int(info_s["executed_layer_cycles"]) == n_layers * T
    else:
        # wavefront additionally (and exactly) skips layer i in cycle 1 while g_i^prev and
        # g_{i+1}^prev are still both zero (the credit has not reached it yet); the resulting
        # trajectory is identical to the dense reference (checked above) because grad E_i is
        # exactly zero there, so this is strictly fewer executed cycles, not an approximation.
        assert int(info_s["executed_layer_cycles"]) <= n_layers * T


# --- Section 5, bullet 5: batch-size invariance of per-sample activities. ---


@pytest.mark.parametrize("gate", SPARSE_GATES)
def test_per_sample_batch_invariance(gate):
    params, scales, skips, phi, x, y = small_case()
    kw = dict(tau=0.0, patience=1, t_min=1, t_max=8, eps_arrive=0.0, gate=gate, phi=phi, **COMMON)
    free_single, _, s1, _ = run_pcalm_layerwise_sparse(params, scales, skips, x[:1], y[:1], **kw)
    free_batch, _, s2, _ = run_pcalm_layerwise_sparse(params, scales, skips, x, y, **kw)
    assert int(s1) == int(s2)
    for a, b in zip(free_single, free_batch):
        assert jnp.allclose(a[0], b[0], atol=1e-5, rtol=1e-5)


def test_invalid_gate_rejected():
    params, scales, skips, phi, x, y = small_case()
    with pytest.raises(ValueError):
        run_pcalm_layerwise_sparse(
            params, scales, skips, x, y, tau=0.1, patience=1, t_min=1, t_max=3, eps_arrive=0.0, gate="bogus",
            phi=phi, **COMMON,
        )


# --- active_layer_cycles pinned exactly from fire_times for a staggered-fire run. ---
# Phase-2 definition: a layer is "active" (not yet fired) in cycles 1..fire_time_i, so
# active_layer_cycles = sum_i fire_time_i.


@pytest.mark.parametrize("run_kind", ["dense_layerwise", "sparse_freeze"])
def test_active_layer_cycles_matches_sum_of_fire_times(run_kind):
    params, scales, skips, phi, x, y = small_case(depth=10)
    kw = dict(tau=0.1, patience=3, t_min=2, t_max=80, eps_arrive=1e-3, phi=phi, **COMMON)
    if run_kind == "dense_layerwise":
        _, _, steps, info = run_pcalm_layerwise(params, scales, skips, x, y, mode="freeze", **kw)
    else:
        _, _, steps, info = run_pcalm_layerwise_sparse(params, scales, skips, x, y, gate="freeze", **kw)

    fire_times = info["fire_times"]
    # Confirm the run actually staggers fire times (otherwise this test would not discriminate).
    assert len(set(int(t) for t in fire_times)) > 1

    expected_active = int(jnp.sum(fire_times))
    assert int(info["active_layer_cycles"]) == expected_active
