from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .config import ExperimentConfig
from .data import load_dataset
from .inference import Schedule, bp_loss, method_grad, method_grad_and_steps, trace_pcalm_adaptive
from .metrics import mse_ce_accuracy, tree_cos
from .model import activation_fn, init_params, logits, model_scales, skip_mask
from .optim import adam_apply, adam_init


def train_one(config: ExperimentConfig, *, data_dir: str | Path = "data") -> dict[str, Any]:
    model = config.model
    method = config.method
    training = config.training
    learning_rate = adam_learning_rate(model.width, model.depth, training.eta0, training.gamma0, training.learning_rate)
    phi = activation_fn(model.activation)
    scales = model_scales(model.width, model.depth, model.input_dim)
    skips = skip_mask(model.depth)
    x_train, y_train, x_test, y_test = load_dataset(
        config.dataset,
        train_subset=training.train_subset,
        test_subset=training.test_subset,
        seed=training.seed,
        data_dir=data_dir,
        input_dim=model.input_dim,
        output_dim=model.output_dim,
    )

    key = jax.random.PRNGKey(training.seed)
    params = init_params(
        key,
        depth=model.depth,
        width=model.width,
        input_dim=model.input_dim,
        output_dim=model.output_dim,
        dtype=jnp.float32,
    )
    opt_state = adam_init(params)
    schedule = Schedule(
        family=method.name,
        budget=method.budget,
        alpha=method.alpha,
        inner_steps=method.inner_steps,
        weight_credit_timing=method.weight_credit_timing,
        tau=method.tau,
        patience=method.patience,
        t_min=method.t_min,
        t_max=method.t_max,
        arrival_frac=method.arrival_frac,
        criterion=method.criterion,
    )
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"CONFIG {json.dumps(asdict(config), sort_keys=True)}", flush=True)

    update = make_update_fn(schedule, scales, skips, phi, method.state_lr, method.rho, learning_rate)
    eval_batch = make_eval_fn(scales, skips, phi)
    diag_batch = make_diag_fn(schedule, scales, skips, phi, method.state_lr, method.rho)
    diag_n = min(training.batch_size, x_train.shape[0])
    x_diag = jnp.asarray(x_train[:diag_n])
    y_diag = jnp.asarray(y_train[:diag_n])
    trace_fn = make_trace_fn(schedule, scales, skips, phi, method.state_lr, method.rho)
    if trace_fn is not None:
        write_trace(trace_fn(params, x_diag, y_diag), output_dir / "diag_trace_init.csv")

    rows = []
    steps_per_batch: list[int] = []
    step = 0
    log_every = 100
    for epoch in range(training.epochs):
        for batch_idx in batch_order(x_train.shape[0], training.batch_size, training.seed + epoch, training.drop_last):
            xb = jnp.asarray(x_train[batch_idx])
            yb = jnp.asarray(y_train[batch_idx])
            params, opt_state, inf_steps = update(params, opt_state, xb, yb)
            steps_per_batch.append(int(inf_steps))
            step += 1
            if step % log_every == 0:
                recent = steps_per_batch[-log_every:]
                print(
                    f"PROGRESS epoch={epoch + 1} step={step} mean_inf_steps_recent={float(np.mean(recent)):.2f}",
                    flush=True,
                )
        train_mse, train_ce, train_acc = evaluate(params, x_train, y_train, training.batch_size, eval_batch)
        test_mse, test_ce, test_acc = evaluate(params, x_test, y_test, training.batch_size, eval_batch)
        rows.append(
            {
                "epoch": epoch + 1,
                "step": step,
                "train_mse": train_mse,
                "train_ce": train_ce,
                "train_acc": train_acc,
                "test_mse": test_mse,
                "test_ce": test_ce,
                "test_acc": test_acc,
            }
        )

    diag = diag_batch(params, x_diag, y_diag)
    if trace_fn is not None:
        write_trace(trace_fn(params, x_diag, y_diag), output_dir / "diag_trace_final.csv")
    steps_arr = np.asarray(steps_per_batch, dtype=np.float64)
    cap = (method.t_max if method.t_max > 0 else method.budget) if method.name == "pcalm_adaptive" else method.budget
    final = {
        "dataset": config.dataset,
        "method": method.name,
        "width": model.width,
        "depth": model.depth,
        "activation": model.activation,
        "seed": training.seed,
        "budget": method.budget if method.name != "bp" else 0,
        "alpha": method.alpha if method.name == "pcalm" else 0.0,
        "state_lr": method.state_lr,
        "rho": method.rho,
        "learning_rate": learning_rate,
        "eta0": training.eta0,
        "gamma0": training.gamma0,
        "epochs": training.epochs,
        "batch_size": training.batch_size,
        "train_subset": training.train_subset,
        "test_subset": training.test_subset,
        "steps": step,
        "final_train_acc": rows[-1]["train_acc"],
        "final_test_acc": rows[-1]["test_acc"],
        "final_train_mse": rows[-1]["train_mse"],
        "final_test_mse": rows[-1]["test_mse"],
        "grad_cos_to_bp": float(diag["grad_cos_to_bp"]),
        "tau": method.tau if method.name == "pcalm_adaptive" else 0.0,
        "patience": method.patience,
        "t_min": method.t_min,
        "t_max": cap if method.name != "bp" else 0,
        "arrival_frac": method.arrival_frac,
        "criterion": method.criterion if method.name == "pcalm_adaptive" else "",
        "mean_inf_steps": float(steps_arr.mean()) if steps_arr.size else 0.0,
        "median_inf_steps": float(np.median(steps_arr)) if steps_arr.size else 0.0,
        "min_inf_steps": int(steps_arr.min()) if steps_arr.size else 0,
        "max_inf_steps": int(steps_arr.max()) if steps_arr.size else 0,
        "frac_at_cap": float(np.mean(steps_arr >= cap)) if (steps_arr.size and method.name != "bp") else 0.0,
    }
    write_csv(rows, output_dir / "metrics.csv")
    write_json(final, output_dir / "summary.json")
    write_json(asdict(config), output_dir / "config.json")
    if steps_per_batch:
        np.savetxt(output_dir / "inf_steps_per_batch.csv", steps_arr, fmt="%d", header="inf_steps", comments="")
    print(
        "RESULT "
        + " ".join(
            f"{k}={final[k]}"
            for k in (
                "method", "dataset", "activation", "width", "depth", "seed", "tau", "criterion", "t_max",
                "final_test_acc", "final_train_acc", "grad_cos_to_bp",
                "mean_inf_steps", "median_inf_steps", "min_inf_steps", "max_inf_steps", "frac_at_cap",
            )
        ),
        flush=True,
    )
    return final


def adam_learning_rate(width: int, depth: int, eta0: float, gamma0: float, explicit_lr: float | None) -> float:
    if gamma0 != 1.0:
        raise ValueError("This reference implementation requires gamma0=1 (fixed model parameterization).")
    if explicit_lr is not None:
        return float(explicit_lr)
    return float(eta0 * (gamma0**2) * math.sqrt(width / depth))


def make_update_fn(schedule: Schedule, scales, skips, phi, state_lr: float, rho: float, learning_rate: float):
    @jax.jit
    def update(params, opt_state, x, y):
        grads, steps = method_grad_and_steps(params, scales, skips, x, y, schedule, state_lr=state_lr, rho=rho, phi=phi)
        params, opt_state = adam_apply(params, grads, opt_state, learning_rate)
        return params, opt_state, steps

    return update


def make_trace_fn(schedule: Schedule, scales, skips, phi, state_lr: float, rho: float):
    """Per-cycle diagnostics of the adaptive trigger statistics; None for BP/PC."""
    if schedule.family not in {"pcalm", "pcalm_adaptive"}:
        return None
    t_max = schedule.t_max if (schedule.family == "pcalm_adaptive" and schedule.t_max > 0) else schedule.budget

    @jax.jit
    def trace(params, x, y):
        return trace_pcalm_adaptive(
            params, scales, skips, x, y,
            state_lr=state_lr, rho=rho, alpha=schedule.alpha, t_max=t_max,
            inner_steps=schedule.inner_steps, phi=phi,
        )

    return trace


def write_trace(trace: dict[str, jax.Array], path: Path) -> None:
    delta = np.asarray(trace["delta"], dtype=np.float64)
    delta_max = np.asarray(trace["delta_max"], dtype=np.float64)
    credit = np.asarray(trace["credit_norm"], dtype=np.float64)
    residual = np.asarray(trace["residual_norm"], dtype=np.float64)
    dual = np.asarray(trace["dual_norm"], dtype=np.float64)
    n_layers = credit.shape[1]
    header = (
        ["t", "delta", "delta_max"]
        + [f"credit_l{i + 1}" for i in range(n_layers)]
        + [f"residual_l{i + 1}" for i in range(n_layers)]
        + [f"dual_l{i + 1}" for i in range(n_layers)]
    )
    t = np.arange(1, delta.shape[0] + 1, dtype=np.float64)[:, None]
    table = np.concatenate([t, delta[:, None], delta_max[:, None], credit, residual, dual], axis=1)
    np.savetxt(path, table, delimiter=",", header=",".join(header), comments="", fmt="%.6g")


def make_eval_fn(scales, skips, phi):
    @jax.jit
    def eval_batch(params, x, y):
        return mse_ce_accuracy(logits(params, scales, skips, x, phi), y)

    return eval_batch


def make_diag_fn(schedule: Schedule, scales, skips, phi, state_lr: float, rho: float):
    @jax.jit
    def diag(params, x, y):
        bp_grads = jax.grad(lambda p: bp_loss(p, scales, skips, x, y, phi))(params)
        method_grads = method_grad(params, scales, skips, x, y, schedule, state_lr=state_lr, rho=rho, phi=phi)
        return {"grad_cos_to_bp": tree_cos(method_grads, bp_grads)}

    return diag


def evaluate(params, X: np.ndarray, Y: np.ndarray, batch_size: int, eval_batch) -> tuple[float, float, float]:
    totals = np.zeros(3, dtype=np.float64)
    count = 0
    for start in range(0, X.shape[0], batch_size):
        stop = min(start + batch_size, X.shape[0])
        x = jnp.asarray(X[start:stop])
        y = jnp.asarray(Y[start:stop])
        mse, ce, acc = eval_batch(params, x, y)
        n = stop - start
        totals += np.array([float(mse), float(ce), float(acc)]) * n
        count += n
    return tuple((totals / max(count, 1)).tolist())


def batch_order(n: int, batch_size: int, seed: int, drop_last: bool) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    usable = (n // batch_size) * batch_size if drop_last else n
    perm = perm[:usable]
    return [perm[start : start + batch_size] for start in range(0, usable, batch_size)]


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(value: Any, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")
