"""Wall-clock per batch relative to dense fixed T=2L, across width and impl.

Reads timing.csv fresh on every run (it will be regenerated with cleaner
numbers later).

Regenerate: uv run --no-project --with matplotlib --with numpy python wallclock-vs-width.py
"""

import csv
from collections import defaultdict

import numpy as np
from orx_figstyle import BASELINE, PALETTE, TEXT, figure_grid, panel_labels, save, use_style

DATA = "/Users/sidvivek/.local/share/openresearch/files/augmented-lagrangian-predictive-coding/runs/timing_sparse_l64_widths_and_l128/timing.csv"

L64_CELLS = ["N32_L64", "N128_L64", "N512_L64"]
L64_WIDTHS = {"N32_L64": 32, "N128_L64": 128, "N512_L64": 512}
BAR_CELL = "N32_L128"

IMPLS = ["dense_layerwise_freeze", "sparse_freeze", "sparse_wavefront"]
IMPL_LABELS = {
    "dense_fixed_2L": "dense fixed T=2L",
    "dense_layerwise_freeze": "dense layerwise-freeze",
    "sparse_freeze": "sparse freeze",
    "sparse_wavefront": "sparse wavefront",
}
IMPL_COLORS = {
    "dense_layerwise_freeze": PALETTE["green"],
    "sparse_freeze": PALETTE["red"],
    "sparse_wavefront": PALETTE["blue"],
}


def load(path):
    rows = defaultdict(dict)
    with open(path) as handle:
        for row in csv.DictReader(handle):
            rows[row["cell"]][row["impl"]] = {
                "ms_median": float(row["ms_median"]),
                "ms_iqr": float(row["ms_iqr"]),
                "active_frac": float(row["active_frac"]),
                "executed_frac": float(row["executed_frac"]),
                "ratio_to_dense_2L": float(row["ratio_to_dense_2L"]),
            }
    return rows


def main():
    data = load(DATA)

    use_style()
    fig, (ax_a, ax_b) = figure_grid(1, 2, width=TEXT, ratio=0.5)

    handles, labels = [], []

    # Panel (a): ratio to dense-2L vs width, one line per impl, log2 x-axis.
    widths = [L64_WIDTHS[c] for c in L64_CELLS if c in data]
    (ref_line,) = ax_a.plot(
        [min(widths), max(widths)], [1.0, 1.0],
        linestyle="--", color=BASELINE, linewidth=1.1, zorder=2,
    )
    handles.append(ref_line)
    labels.append(IMPL_LABELS["dense_fixed_2L"])

    for impl in IMPLS:
        xs, ys, yerrs = [], [], []
        for cell in L64_CELLS:
            if cell not in data or impl not in data[cell] or "dense_fixed_2L" not in data[cell]:
                continue
            row = data[cell][impl]
            dense_ms = data[cell]["dense_fixed_2L"]["ms_median"]
            xs.append(L64_WIDTHS[cell])
            ys.append(row["ratio_to_dense_2L"])
            yerrs.append(row["ms_iqr"] / dense_ms)
        if not xs:
            continue
        color = IMPL_COLORS[impl]
        h = ax_a.errorbar(
            xs, ys, yerr=yerrs, marker="o", markersize=4, color=color,
            linewidth=1.1, capsize=2, elinewidth=0.8, zorder=3,
        )
        handles.append(h)
        labels.append(IMPL_LABELS[impl])

    # Dotted horizontal executed-fraction guides for the two sparse impls,
    # at their largest-width value (the regime the guide is meant to read at).
    for impl in ("sparse_freeze", "sparse_wavefront"):
        cell = L64_CELLS[-1] if L64_CELLS[-1] in data else None
        if cell is None or impl not in data.get(cell, {}):
            continue
        frac = data[cell][impl]["executed_frac"]
        color = IMPL_COLORS[impl]
        ax_a.axhline(frac, linestyle=":", linewidth=0.8, color=color, alpha=0.7, zorder=1)
        ax_a.annotate(
            "executed fraction",
            xy=(0.98, frac), xycoords=("axes fraction", "data"),
            xytext=(0, 3), textcoords="offset points",
            ha="right", va="bottom", fontsize=5.5, color=color, clip_on=True,
        )

    ax_a.set_xscale("log", base=2)
    ax_a.set_xticks([32, 128, 512])
    ax_a.set_xticklabels(["32", "128", "512"])
    ax_a.set_xlabel("width N (L=64)")
    ax_a.set_ylabel("wall-clock per batch / dense T=2L")

    # Panel (b): grouped bars for N32_L128 — ratio_to_dense_2L and
    # executed_frac per impl.
    all_impls = ["dense_fixed_2L"] + IMPLS
    bar_impls = [impl for impl in all_impls if impl in data.get(BAR_CELL, {})]
    x = np.arange(len(bar_impls))
    bar_width = 0.35

    ratios = [data[BAR_CELL][impl]["ratio_to_dense_2L"] for impl in bar_impls]
    executed = [data[BAR_CELL][impl]["executed_frac"] for impl in bar_impls]

    bars1 = ax_b.bar(
        x - bar_width / 2, ratios, bar_width,
        color=[IMPL_COLORS.get(impl, BASELINE) for impl in bar_impls],
        label="ratio to dense T=2L",
    )
    bars2 = ax_b.bar(
        x + bar_width / 2, executed, bar_width,
        color=[IMPL_COLORS.get(impl, BASELINE) for impl in bar_impls],
        alpha=0.45,
        label="executed fraction",
    )
    for bars in (bars1, bars2):
        for bar in bars:
            height = bar.get_height()
            ax_b.annotate(
                f"{height:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 2), textcoords="offset points",
                ha="center", va="bottom", fontsize=5.5,
            )

    ax_b.set_xticks(x)
    ax_b.set_xticklabels([IMPL_LABELS[impl].replace(" ", "\n") for impl in bar_impls], fontsize=6)
    ax_b.set_ylabel("fraction of dense T=2L")
    ax_b.legend(
        loc="lower left", bbox_to_anchor=(0.0, 1.0), frameon=False, fontsize=6,
        borderaxespad=0.0,
    )

    panel_labels((ax_a, ax_b))
    fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False)

    save(
        fig,
        "/Users/sidvivek/.local/share/openresearch/files/augmented-lagrangian-predictive-coding/adaptive-budget-phase2/figures/wallclock-vs-width",
    )


if __name__ == "__main__":
    main()
