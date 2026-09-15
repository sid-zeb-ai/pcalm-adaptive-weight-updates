"""Test accuracy vs. active layer-cycles (fraction of the fixed T=2L schedule).

Regenerate: uv run --no-project --with matplotlib --with numpy python acc-vs-active.py
"""

import json
import os

from orx_figstyle import (
    BASELINE,
    MUTED,
    PALETTE,
    TEXT,
    family,
    figure_grid,
    panel_labels,
    save,
    use_style,
)

RUNS = "/Users/sidvivek/.local/share/openresearch/files/augmented-lagrangian-predictive-coding/runs"
CELLS = [(32, 32), (32, 64)]


def tag(n, l, variant):
    return f"fashion_relu_n{n}_l{l}_{variant}"


def load_aggregate(n, l, variant):
    path = os.path.join(RUNS, tag(n, l, variant), "aggregate.json")
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        return json.load(handle)


def active_frac(agg, l):
    """Active layer-cycles as a fraction of the fixed T=2L schedule.

    Layerwise runs report the exact quantity directly. Every other method
    activates all layers every cycle, so mean_inf_steps/(2L) is exact for
    them too.
    """
    if "active_layer_cycles_frac_mean" in agg:
        return agg["active_layer_cycles_frac_mean"], agg["active_layer_cycles_frac_std"]
    x = agg["mean_inf_steps_mean"] / (2 * l)
    xerr = agg["mean_inf_steps_std"] / (2 * l)
    return x, xerr


def main():
    use_style()
    fig, axes = figure_grid(1, 2, width=TEXT, ratio=0.62, sharey=True)

    freeze_colors = family("purple", 2)

    handles, labels = [], []

    for i, (n, l) in enumerate(CELLS):
        ax = axes[i]
        ax.axvline(0.5, color=MUTED, linewidth=0.6, zorder=0)
        ax.axvline(1.0, color=MUTED, linewidth=0.6, zorder=0)

        # BP: horizontal dashed line with a shaded +/- std band.
        bp = load_aggregate(n, l, "bp")
        if bp is not None:
            mean = bp["final_test_acc_mean"]
            std = bp["final_test_acc_std"]
            (line,) = ax.plot(
                [0, 1.1], [mean, mean],
                linestyle="--", color=BASELINE, linewidth=1.2, zorder=2,
            )
            ax.fill_between([0, 1.1], mean - std, mean + std, color=BASELINE, alpha=0.18, linewidth=0)
            if "BP" not in labels:
                handles.append(line)
                labels.append("BP")

        # Fixed-budget PC-ALM T=L, T=2L: filled squares.
        for variant, mk_label, color in (
            ("pcalm_L", "PC-ALM T=L", PALETTE["green"]),
            ("pcalm_2L", "PC-ALM T=2L", PALETTE["red"]),
        ):
            agg = load_aggregate(n, l, variant)
            if agg is None:
                continue
            x, xerr = active_frac(agg, l)
            y = agg["final_test_acc_mean"]
            yerr = agg["final_test_acc_std"]
            h = ax.errorbar(
                x, y, xerr=xerr, yerr=yerr,
                marker="s", color=color, linestyle="none", markersize=4, capsize=2,
                elinewidth=0.8, zorder=3,
            )
            if mk_label not in labels:
                handles.append(h)
                labels.append(mk_label)

        # PC T=2L: open triangle.
        agg = load_aggregate(n, l, "pc_2L")
        if agg is not None:
            x, xerr = active_frac(agg, l)
            y = agg["final_test_acc_mean"]
            yerr = agg["final_test_acc_std"]
            h = ax.errorbar(
                x, y, xerr=xerr, yerr=yerr,
                marker="^", markerfacecolor="none", markeredgecolor=PALETTE["cyan"],
                color=PALETTE["cyan"], linestyle="none", markersize=4.5, capsize=2,
                elinewidth=0.8, zorder=3,
            )
            if "PC T=2L" not in labels:
                handles.append(h)
                labels.append("PC T=2L")

        # Global adaptive sum, tau=0.1: filled circle.
        agg = load_aggregate(n, l, "adaptive_sum_tau0.1_tmax3L")
        if agg is not None:
            x, xerr = active_frac(agg, l)
            y = agg["final_test_acc_mean"]
            yerr = agg["final_test_acc_std"]
            h = ax.errorbar(
                x, y, xerr=xerr, yerr=yerr,
                marker="o", color=PALETTE["blue"], linestyle="none", markersize=4, capsize=2,
                elinewidth=0.8, zorder=4,
            )
            lbl = "global adaptive sum τ=0.1"
            if lbl not in labels:
                handles.append(h)
                labels.append(lbl)

        # Layerwise freeze, tau in {0.1, 0.03}: filled stars, 2-shade family.
        for tau, color in zip((0.1, 0.03), freeze_colors):
            variant = f"layerwise_freeze_tau{tau}_tmax3L"
            agg = load_aggregate(n, l, variant)
            if agg is None:
                continue
            x, xerr = active_frac(agg, l)
            y = agg["final_test_acc_mean"]
            yerr = agg["final_test_acc_std"]
            h = ax.errorbar(
                x, y, xerr=xerr, yerr=yerr,
                marker="*", color=color, linestyle="none", markersize=7, capsize=2,
                elinewidth=0.8, zorder=5,
            )
            lbl = f"layerwise freeze τ={tau}"
            if lbl not in labels:
                handles.append(h)
                labels.append(lbl)

        # Layerwise fire_only, tau=0.1: open star.
        agg = load_aggregate(n, l, "layerwise_fire_only_tau0.1_tmax3L")
        if agg is not None:
            x, xerr = active_frac(agg, l)
            y = agg["final_test_acc_mean"]
            yerr = agg["final_test_acc_std"]
            h = ax.errorbar(
                x, y, xerr=xerr, yerr=yerr,
                marker="*", markerfacecolor="none", markeredgecolor=PALETTE["orange"],
                color=PALETTE["orange"], linestyle="none", markersize=7, capsize=2,
                elinewidth=0.8, zorder=5,
            )
            lbl = "layerwise fire-only τ=0.1"
            if lbl not in labels:
                handles.append(h)
                labels.append(lbl)

        ax.set_xlim(0, 1.1)
        ax.set_xlabel("active layer-cycles / (2L)")
        if i == 0:
            ax.set_ylabel("test accuracy after 1 epoch")
        ax.text(
            0.97, 0.05, f"N={n}, L={l}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5,
        )

    panel_labels(axes)
    fig.legend(
        handles, labels, loc="outside lower center", ncol=3, frameon=False,
    )

    save(
        fig,
        "/Users/sidvivek/.local/share/openresearch/files/augmented-lagrangian-predictive-coding/adaptive-budget-phase2/figures/acc-vs-active",
    )


if __name__ == "__main__":
    main()
