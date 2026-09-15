"""Test accuracy vs. mean inference steps per batch, one panel per (N, L) cell.

Regenerate: uv run --no-project --with matplotlib --with numpy python acc-vs-steps.py
"""

import json
import os

import numpy as np
from orx_figstyle import (
    BASELINE,
    MUTED,
    PALETTE,
    WIDE,
    family,
    figure_grid,
    panel_labels,
    save,
    use_style,
)

RUNS = "/Users/sidvivek/.local/share/openresearch/files/augmented-lagrangian-predictive-coding/runs"
CELLS = [(32, 16), (32, 32), (32, 64), (8, 64)]

ADAPTIVE_SUM_TAUS = [0.01, 0.03, 0.1]
ADAPTIVE_MAX_TAU = 0.03


def tag(n, l, variant):
    return f"fashion_relu_n{n}_l{l}_{variant}"


def load_aggregate(n, l, variant):
    path = os.path.join(RUNS, tag(n, l, variant), "aggregate.json")
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        return json.load(handle)


def main():
    use_style()
    fig, axes = figure_grid(1, 4, width=WIDE, ratio=0.34, sharey=True)

    sum_colors = family("blue", len(ADAPTIVE_SUM_TAUS))
    max_color = PALETTE["orange"]

    handles, labels = [], []

    for i, (n, l) in enumerate(CELLS):
        ax = axes[i]
        ax.axvline(1, color=MUTED, linewidth=0.6, zorder=0)
        ax.axvline(2, color=MUTED, linewidth=0.6, zorder=0)
        ax.axvline(3, color=MUTED, linewidth=0.6, zorder=0)

        # BP: horizontal dashed line with a shaded +/- std band.
        bp = load_aggregate(n, l, "bp")
        if bp is not None:
            mean = bp["final_test_acc_mean"]
            std = bp["final_test_acc_std"]
            (line,) = ax.plot(
                [0, 3.2], [mean, mean],
                linestyle="--", color=BASELINE, linewidth=1.2, zorder=2,
            )
            ax.fill_between([0, 3.2], mean - std, mean + std, color=BASELINE, alpha=0.18, linewidth=0)
            if "BP" not in labels:
                handles.append(line)
                labels.append("BP")

        # Fixed-budget PC-ALM T=L, T=2L: filled squares.
        for variant, mk_label in (("pcalm_L", "PC-ALM T=L"), ("pcalm_2L", "PC-ALM T=2L")):
            agg = load_aggregate(n, l, variant)
            if agg is None:
                continue
            x = agg["mean_inf_steps_mean"] / l
            xerr = agg["mean_inf_steps_std"] / l
            y = agg["final_test_acc_mean"]
            yerr = agg["final_test_acc_std"]
            h = ax.errorbar(
                x, y, xerr=xerr, yerr=yerr,
                marker="s", color=PALETTE["green"] if variant == "pcalm_L" else PALETTE["red"],
                linestyle="none", markersize=4, capsize=2, elinewidth=0.8, zorder=3,
            )
            if mk_label not in labels:
                handles.append(h)
                labels.append(mk_label)

        # PC T=2L: open triangle.
        agg = load_aggregate(n, l, "pc_2L")
        if agg is not None:
            x = agg["mean_inf_steps_mean"] / l
            xerr = agg["mean_inf_steps_std"] / l
            y = agg["final_test_acc_mean"]
            yerr = agg["final_test_acc_std"]
            h = ax.errorbar(
                x, y, xerr=xerr, yerr=yerr,
                marker="^", markerfacecolor="none", markeredgecolor=PALETTE["purple"],
                color=PALETTE["purple"], linestyle="none", markersize=4.5, capsize=2,
                elinewidth=0.8, zorder=3,
            )
            if "PC T=2L" not in labels:
                handles.append(h)
                labels.append("PC T=2L")

        # Adaptive sum, 3 taus: filled circles, 3 shades of blue.
        for tau, color in zip(ADAPTIVE_SUM_TAUS, sum_colors):
            variant = f"adaptive_sum_tau{tau}_tmax3L"
            agg = load_aggregate(n, l, variant)
            if agg is None:
                continue
            x = agg["mean_inf_steps_mean"] / l
            xerr = agg["mean_inf_steps_std"] / l
            y = agg["final_test_acc_mean"]
            yerr = agg["final_test_acc_std"]
            h = ax.errorbar(
                x, y, xerr=xerr, yerr=yerr,
                marker="o", color=color, linestyle="none", markersize=4, capsize=2,
                elinewidth=0.8, zorder=4,
            )
            lbl = f"adaptive sum τ={tau}"
            if lbl not in labels:
                handles.append(h)
                labels.append(lbl)

        # Adaptive max, tau=0.03: filled diamond, different hue.
        variant = f"adaptive_max_tau{ADAPTIVE_MAX_TAU}_tmax3L"
        agg = load_aggregate(n, l, variant)
        if agg is not None:
            x = agg["mean_inf_steps_mean"] / l
            xerr = agg["mean_inf_steps_std"] / l
            y = agg["final_test_acc_mean"]
            yerr = agg["final_test_acc_std"]
            h = ax.errorbar(
                x, y, xerr=xerr, yerr=yerr,
                marker="D", color=max_color, linestyle="none", markersize=4, capsize=2,
                elinewidth=0.8, zorder=4,
            )
            lbl = f"adaptive max τ={ADAPTIVE_MAX_TAU}"
            if lbl not in labels:
                handles.append(h)
                labels.append(lbl)

        ax.set_xlim(0, 3.2)
        ax.set_xlabel("steps / L")
        if i == 0:
            ax.set_ylabel("test accuracy after 1 epoch")
        ax.text(
            0.97, 0.05, f"N={n}, L={l}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5,
        )

    panel_labels(axes)
    fig.legend(
        handles, labels, loc="outside lower center", ncol=4, frameon=False,
    )

    save(fig, "/Users/sidvivek/.local/share/openresearch/files/augmented-lagrangian-predictive-coding/adaptive-budget-phase1/figures/acc-vs-steps")


if __name__ == "__main__":
    main()
