"""Adaptive-trigger diagnostics at initialization, L=64, adaptive-sum tau=0.03.

Regenerate: uv run --no-project --with matplotlib --with numpy python trigger-trace.py
"""

import csv
import json
import math
import os
import sys

from orx_figstyle import BASELINE, PALETTE, TEXT, family, figure_grid, panel_labels, save, use_style

TAG = "fashion_relu_n32_l64_adaptive_sum_tau0.03_tmax3L"
RUN_DIR = f"/Users/sidvivek/.local/share/openresearch/files/augmented-lagrangian-predictive-coding/runs/{TAG}"
TRACE_PATH = os.path.join(RUN_DIR, "seed0", "diag_trace_init.csv")
AGG_PATH = os.path.join(RUN_DIR, "aggregate.json")

L = 64
TAUS = [0.01, 0.03, 0.1]
CREDIT_LAYERS = [1, 16, 32, 48, 63]
ALPHA = 1.0


def load_trace(path):
    rows = []
    with open(path) as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append({k: float(v) for k, v in row.items()})
    return rows


def main():
    if not os.path.exists(TRACE_PATH) or not os.path.exists(AGG_PATH):
        print(
            f"trigger-trace.py: skipping — missing data for {TAG} "
            f"(runs still finishing: {TRACE_PATH} or {AGG_PATH} not found)",
            file=sys.stderr,
        )
        return

    with open(AGG_PATH) as handle:
        agg = json.load(handle)
    eta_h = agg["state_lr"]

    rows = load_trace(TRACE_PATH)
    t = [row["t"] for row in rows]

    use_style()
    fig, axes = figure_grid(1, 2, width=TEXT, ratio=0.42)
    ax_a, ax_b = axes

    t_2L = 2 * L
    t_star = L / math.sqrt(ALPHA * eta_h)

    # Panel (a): sum and max criteria vs t, log-y, with tau thresholds and
    # vertical reference lines at t=2L and the predicted inflection.
    delta = [row["delta"] for row in rows]
    delta_max = [row["delta_max"] for row in rows]
    (line_sum,) = ax_a.plot(t, delta, color=PALETTE["blue"], label="sum criterion")
    (line_max,) = ax_a.plot(t, delta_max, color=PALETTE["orange"], label="max criterion")
    ax_a.set_yscale("log")

    for tau in TAUS:
        ax_a.axhline(tau, color=BASELINE, linestyle=":", linewidth=0.7, zorder=0)
        ax_a.annotate(
            f"τ={tau}",
            xy=(1.0, tau), xycoords=("axes fraction", "data"),
            xytext=(2, 0), textcoords="offset points",
            ha="left", va="center", fontsize=6, color=BASELINE, clip_on=False,
        )

    for ax in (ax_a, ax_b):
        ax.axvline(t_2L, color=BASELINE, linestyle="-", linewidth=0.8, zorder=0)
        ax.axvline(t_star, color=BASELINE, linestyle="--", linewidth=0.8, zorder=0)
        ax.set_xlabel("inference cycle t")

    # t=2L and the predicted inflection often land close together; stack their
    # labels at different heights, one left- and one right-aligned, so they
    # never overlap regardless of spacing.
    ax_a.text(
        t_2L, 1.14, "t=2L", transform=ax_a.get_xaxis_transform(),
        ha="right", va="bottom", fontsize=6, color=BASELINE, clip_on=False,
    )
    ax_a.text(
        t_star, 1.0, r"t=L/$\sqrt{\alpha\eta_h}$",
        transform=ax_a.get_xaxis_transform(),
        ha="left", va="bottom", fontsize=6, color=BASELINE, clip_on=False,
    )
    ax_a.set_ylabel(r"relative change of credit, $\delta_t$ (dimensionless)")
    ax_a.legend(loc="upper right", frameon=False)

    # Panel (b): per-layer credit norm vs t, light (input-side) to dark
    # (output-side).
    shades = family("green", len(CREDIT_LAYERS))
    header_layers = {
        int(k.split("_l")[1]) for k in rows[0].keys() if k.startswith("credit_l")
    }
    for layer, color in zip(CREDIT_LAYERS, shades):
        key = f"credit_l{layer}"
        if layer not in header_layers:
            continue
        y = [row[key] for row in rows]
        ax_b.plot(t, y, color=color, label=f"layer {layer}")
    ax_b.set_ylabel(r"$\Vert\lambda_i + \rho r_i\Vert$ (Frobenius, batch of 64)")
    ax_b.legend(loc="upper right", frameon=False, fontsize=6)

    panel_labels(axes)

    save(
        fig,
        "/Users/sidvivek/.local/share/openresearch/files/augmented-lagrangian-predictive-coding/adaptive-budget-phase1/figures/trigger-trace",
    )


if __name__ == "__main__":
    main()
