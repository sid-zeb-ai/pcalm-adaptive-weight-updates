"""Per-layer fire cycle for layerwise-freeze, vs. the wave prediction.

Regenerate: uv run --no-project --with matplotlib --with numpy python fire-times.py
"""

import csv
import json
import math
import os

from orx_figstyle import BASELINE, PALETTE, COLUMN, figure, save, use_style

RUNS = "/Users/sidvivek/.local/share/openresearch/files/augmented-lagrangian-predictive-coding/runs"
CELLS = [(32, 32, "blue"), (32, 64, "red")]
ALPHA = 1.0
C = 1.0


def tag(n, l):
    return f"fashion_relu_n{n}_l{l}_layerwise_freeze_tau0.1_tmax3L"


def load_fire_times(path):
    layers, times = [], []
    with open(path) as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            layers.append(int(row["layer"]))
            times.append(float(row["mean_fire_time"]))
    return layers, times


def main():
    use_style()
    fig, ax = figure(width=COLUMN, ratio=0.75)

    any_data = False
    for n, l, color_name in CELLS:
        run_tag = tag(n, l)
        run_dir = os.path.join(RUNS, run_tag)
        fire_path = os.path.join(run_dir, "seed0", "fire_times_mean.csv")
        agg_path = os.path.join(run_dir, "aggregate.json")
        if not os.path.exists(fire_path) or not os.path.exists(agg_path):
            continue
        any_data = True

        with open(agg_path) as handle:
            agg = json.load(handle)
        eta_h = agg["state_lr"]

        layers, times = load_fire_times(fire_path)
        color = PALETTE[color_name]
        ax.plot(
            layers, times, marker="o", markersize=2.5, linewidth=0.9,
            color=color, label=f"L={l}",
        )

        t_pred = [C * (l - i) / math.sqrt(ALPHA * eta_h) for i in layers]
        ax.plot(
            layers, t_pred, linestyle="--", linewidth=0.9, color=color, alpha=0.5,
        )

        ax.axhline(2 * l, color=color, linestyle=":", linewidth=0.7, alpha=0.6, zorder=0)

    if not any_data:
        print("fire-times.py: skipping — no layerwise_freeze_tau0.1 data found for either cell")
        return

    ax.set_xlabel("layer index i (1 = input side)")
    ax.set_ylabel("fire cycle t")
    ax.legend(loc="upper right", frameon=False)

    save(
        fig,
        "/Users/sidvivek/.local/share/openresearch/files/augmented-lagrangian-predictive-coding/adaptive-budget-phase2/figures/fire-times",
    )


if __name__ == "__main__":
    main()
