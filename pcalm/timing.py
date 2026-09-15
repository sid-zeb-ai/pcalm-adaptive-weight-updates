"""Wall-clock timing harness for dense vs. sparse (really-skipping) PC-ALM (spec section 6).

Compares, at initialisation (fire times do not depend on training state) with random
Fashion-MNIST-shaped inputs:
  - dense PC-ALM fixed T = 2L                (`run_pcalm`)
  - dense layer-wise freeze, masked compute  (`run_pcalm_layerwise`, mode="freeze")
  - sparse freeze gate                       (`run_pcalm_layerwise_sparse`, gate="freeze")
  - sparse wavefront gate                    (`run_pcalm_layerwise_sparse`, gate="wavefront")

Called from `scripts/run_node.py` when the node config has a top-level `kind: timing`.
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .inference import run_pcalm, run_pcalm_layerwise, run_pcalm_layerwise_sparse
from .model import activation_fn, init_params, model_scales, skip_mask

ROOT = Path(__file__).resolve().parents[1]
ETA_TABLE = ROOT / "configs" / "eta_best_by_cell.csv"
ETA_BY_DEPTH = ROOT / "configs" / "eta_by_depth.csv"

INPUT_DIM = 784
OUTPUT_DIM = 10
BATCH_SIZE = 64
DEFAULT_REPEATS = 20
WARMUP_CALLS = 3
ACTIVATION = "relu"
DATASET_FOR_ETA = "fashion_mnist"

ALPHA = 1.0
RHO = 1.0
TAU = 0.1
PATIENCE = 3
T_MIN = 2
EPS_ARRIVE = 1e-3


def lookup_state_lr(dataset: str, activation: str, width: int, depth: int) -> float:
    """The paper's eta_h for this cell; fallback to the depth table, then a flat default.

    Duplicated (rather than imported) from `scripts/run_node.py` so this module has no
    dependency on the `scripts/` package.
    """
    if ETA_TABLE.exists():
        with ETA_TABLE.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if (row["dataset"], row["activation"], int(row["N"]), int(row["L"])) == (
                    dataset,
                    activation,
                    width,
                    depth,
                ):
                    return float(row["eta_best_1_over_lambda_median"])
    if ETA_BY_DEPTH.exists():
        with ETA_BY_DEPTH.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if int(row["L"]) == depth:
                    return float(row["state_lr"])
    return 0.25


def _make_random_batch(input_dim: int, output_dim: int, batch_size: int, seed: int):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((batch_size, input_dim)).astype(np.float32)
    labels = rng.integers(0, output_dim, size=(batch_size,))
    y = np.eye(output_dim, dtype=np.float32)[labels]
    return jnp.asarray(x), jnp.asarray(y)


def _frac(count: int, n_hidden_layers: int, two_l: int) -> float:
    denom = n_hidden_layers * two_l
    return float(count) / denom if denom > 0 else 0.0


def _time_calls(call, repeats: int) -> np.ndarray:
    # Warm up (also triggers compilation).
    for _ in range(WARMUP_CALLS):
        jax.block_until_ready(call())
    times_ms = np.empty(repeats, dtype=np.float64)
    for i in range(repeats):
        start = time.perf_counter()
        jax.block_until_ready(call())
        times_ms[i] = (time.perf_counter() - start) * 1e3
    return times_ms


def _median_iqr(values: np.ndarray) -> tuple[float, float]:
    q1, med, q3 = np.percentile(values, [25, 50, 75])
    return float(med), float(q3 - q1)


def _cell_configs(width: int, depth: int, state_lr: float, scales, skips, phi):
    two_l = 2 * depth
    t_max = 3 * depth
    n_hidden_layers = depth - 1
    common_gate = dict(
        state_lr=state_lr, rho=RHO, alpha=ALPHA, tau=TAU, patience=PATIENCE, t_min=T_MIN,
        t_max=t_max, eps_arrive=EPS_ARRIVE, inner_steps=1,
    )

    def dense_fixed(params, x, y):
        return run_pcalm(
            params, scales, skips, x, y,
            state_lr=state_lr, rho=RHO, alpha=ALPHA, budget=two_l, inner_steps=1,
            weight_credit_timing="pre_dual_energy", phi=phi,
        )

    def dense_layerwise(params, x, y):
        return run_pcalm_layerwise(params, scales, skips, x, y, mode="freeze", phi=phi, **common_gate)

    def sparse_freeze(params, x, y):
        return run_pcalm_layerwise_sparse(params, scales, skips, x, y, gate="freeze", phi=phi, **common_gate)

    def sparse_wavefront(params, x, y):
        return run_pcalm_layerwise_sparse(params, scales, skips, x, y, gate="wavefront", phi=phi, **common_gate)

    return [
        ("dense_fixed_2L", dense_fixed),
        ("dense_layerwise_freeze", dense_layerwise),
        ("sparse_freeze", sparse_freeze),
        ("sparse_wavefront", sparse_wavefront),
    ], two_l, t_max, n_hidden_layers


def run_timing(raw: dict[str, Any], output_root: str, data_dir: str) -> None:
    """Entry point invoked by `scripts/run_node.py` for a node config with `kind: timing`."""
    tag = str(raw.get("tag", "timing"))
    timing_cfg = dict(raw.get("timing", {}) or {})
    cells = list(timing_cfg.get("cells", []))
    repeats = int(timing_cfg.get("repeats", DEFAULT_REPEATS))
    if not cells:
        raise ValueError("timing.cells must list at least one {width, depth} cell")

    out_dir = Path(output_root) / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"NODE kind=timing tag={tag} n_cells={len(cells)} repeats={repeats}", flush=True)

    csv_rows: list[dict[str, Any]] = []
    for cell in cells:
        width = int(cell["width"])
        depth = int(cell["depth"])
        state_lr = lookup_state_lr(DATASET_FOR_ETA, ACTIVATION, width, depth)
        phi = activation_fn(ACTIVATION)
        scales = model_scales(width=width, depth=depth, input_dim=INPUT_DIM)
        skips = skip_mask(depth)
        params = init_params(
            jax.random.PRNGKey(0), depth=depth, width=width, input_dim=INPUT_DIM, output_dim=OUTPUT_DIM,
        )
        x, y = _make_random_batch(INPUT_DIM, OUTPUT_DIM, BATCH_SIZE, seed=0)

        configs, two_l, t_max, n_hidden_layers = _cell_configs(width, depth, state_lr, scales, skips, phi)
        cell_name = f"N{width}_L{depth}"

        dense_median_ms = None
        for impl_name, fn in configs:
            jitted = jax.jit(fn)

            def call(jitted=jitted):
                return jitted(params, x, y)

            times_ms = _time_calls(call, repeats)
            median_ms, iqr_ms = _median_iqr(times_ms)
            if impl_name == "dense_fixed_2L":
                dense_median_ms = median_ms

            out = call()
            if impl_name == "dense_fixed_2L":
                free, duals = out
                steps = two_l
                active_frac = _frac(n_hidden_layers * steps, n_hidden_layers, two_l)
                executed_frac = active_frac
            else:
                free, duals, steps_arr, info = out
                steps = int(steps_arr)
                active_frac = _frac(int(info["active_layer_cycles"]), n_hidden_layers, two_l)
                executed_frac = _frac(int(info["executed_layer_cycles"]), n_hidden_layers, two_l)

            ratio = median_ms / dense_median_ms if dense_median_ms else 1.0
            print(
                f"TIMING cell={cell_name} impl={impl_name} ms_median={median_ms:.4f} ms_iqr={iqr_ms:.4f} "
                f"steps={steps} active_frac={active_frac:.4f} executed_frac={executed_frac:.4f} "
                f"ratio_to_dense_2L={ratio:.4f}",
                flush=True,
            )
            csv_rows.append(
                {
                    "cell": cell_name,
                    "width": width,
                    "depth": depth,
                    "impl": impl_name,
                    "ms_median": median_ms,
                    "ms_iqr": iqr_ms,
                    "steps": steps,
                    "active_frac": active_frac,
                    "executed_frac": executed_frac,
                    "ratio_to_dense_2L": ratio,
                }
            )

    csv_path = out_dir / "timing.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    with (out_dir / "timing.json").open("w", encoding="utf-8") as f:
        json.dump(csv_rows, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"RESULT kind=timing n_cells={len(cells)}", flush=True)
