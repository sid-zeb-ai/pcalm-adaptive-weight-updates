"""Fixed run entry point for one experiment node.

Reads `configs/run.yaml` (the only file an experiment branch is expected to change), runs
every listed seed sequentially, prints one RESULT line per seed and an AGGREGATE line at
the end, and writes per-seed outputs under `<output-root>/<tag>/seed<k>/`.

Node-level conveniences on top of `pcalm.config`:
  seeds: [0, 1, 2]              -> list of seeds to run (default [0])
  tag: fashion_relu_n32_l32_pcalm_2L
  method.budget / method.t_max  -> may be "L", "2L", "3L" strings (multiples of depth)
  method.state_lr: auto         -> looked up from configs/eta_best_by_cell.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pcalm.config import config_from_dict  # noqa: E402
from pcalm.training import train_one  # noqa: E402

ETA_TABLE = Path(__file__).resolve().parents[1] / "configs" / "eta_best_by_cell.csv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run one experiment node from configs/run.yaml.")
    p.add_argument("--config", type=Path, default=Path("configs/run.yaml"))
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--output-root", type=str, default="results")
    return p.parse_args()


def depth_multiple(value, depth: int) -> int:
    if isinstance(value, int):
        return value
    m = re.fullmatch(r"\s*(\d*(?:\.\d+)?)\s*L\s*", str(value))
    if not m:
        raise ValueError(f"budget/t_max must be an int or '<k>L', got {value!r}")
    k = float(m.group(1)) if m.group(1) else 1.0
    return int(round(k * depth))


def lookup_state_lr(dataset: str, activation: str, width: int, depth: int) -> float:
    with ETA_TABLE.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if (row["dataset"], row["activation"], int(row["N"]), int(row["L"])) == (dataset, activation, width, depth):
                return float(row["eta_best_1_over_lambda_median"])
    raise KeyError(f"no eta_h in {ETA_TABLE} for {(dataset, activation, width, depth)}")


def main() -> None:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if str(raw.pop("kind", "")) == "timing":
        import pcalm.timing as timing  # noqa: E402 (local import: only needed for kind=timing)

        timing.run_timing(raw, args.output_root, args.data_dir)
        return
    seeds = [int(s) for s in raw.pop("seeds", [0])]
    tag = str(raw.pop("tag", "run"))
    method_raw = dict(raw.get("method", {}) or {})
    model_raw = dict(raw.get("model", {}) or {})
    depth = int(model_raw.get("depth", 32))
    width = int(model_raw.get("width", 32))
    activation = str(model_raw.get("activation", "relu"))
    dataset = str(raw.get("dataset", "synthetic"))
    if "budget" in method_raw:
        method_raw["budget"] = depth_multiple(method_raw["budget"], depth)
    if "t_max" in method_raw:
        method_raw["t_max"] = depth_multiple(method_raw["t_max"], depth)
    if method_raw.get("state_lr", None) in ("auto", None):
        method_raw["state_lr"] = lookup_state_lr(dataset, activation, width, depth)
    raw["method"] = method_raw
    raw["output_dir"] = str(Path(args.output_root) / tag)
    base = config_from_dict(raw)
    print(f"NODE tag={tag} seeds={seeds} data_dir={args.data_dir} output_root={args.output_root}", flush=True)

    summaries = []
    for seed in seeds:
        cfg = replace(
            base,
            output_dir=str(Path(args.output_root) / tag / f"seed{seed}"),
            training=replace(base.training, seed=seed),
        )
        summaries.append(train_one(cfg, data_dir=args.data_dir))

    keys = [
        "final_test_acc", "final_train_acc", "grad_cos_to_bp", "mean_inf_steps", "median_inf_steps", "frac_at_cap",
        "active_layer_cycles_frac", "executed_layer_cycles_frac", "mean_fire_time_over_L",
    ]
    agg = {"tag": tag, "n_seeds": len(summaries), "method": base.method.name, "width": width, "depth": depth,
           "activation": activation, "dataset": dataset, "tau": base.method.tau, "t_max": base.method.t_max,
           "budget": base.method.budget, "state_lr": base.method.state_lr}
    for k in keys:
        vals = np.asarray([s[k] for s in summaries], dtype=np.float64)
        agg[f"{k}_mean"] = float(vals.mean())
        agg[f"{k}_std"] = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
    out_dir = Path(args.output_root) / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "seeds.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summaries[0]))
        w.writeheader()
        w.writerows(summaries)
    with (out_dir / "aggregate.json").open("w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2, sort_keys=True)
        f.write("\n")
    print("AGGREGATE " + " ".join(f"{k}={agg[k]}" for k in agg), flush=True)


if __name__ == "__main__":
    main()
