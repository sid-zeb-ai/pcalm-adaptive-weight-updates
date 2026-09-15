"""Write configs/run.yaml for one experiment node.

Example:
  python scripts/make_node_config.py --width 32 --depth 64 --method pcalm_adaptive \
      --tau 0.03 --t-max 3L --criterion sum
The tag is derived from the arguments unless --tag is given.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="fashion_mnist")
    p.add_argument("--activation", default="relu")
    p.add_argument("--width", type=int, required=True)
    p.add_argument("--depth", type=int, required=True)
    p.add_argument(
        "--method",
        choices=["bp", "pc", "pcalm", "pcalm_adaptive", "pcalm_layerwise", "pcalm_layerwise_sparse"],
        required=True,
    )
    p.add_argument("--budget", default="2L", help="int or '<k>L'; ignored by bp")
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--rho", type=float, default=1.0)
    p.add_argument("--tau", type=float, default=0.0)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--t-min", type=int, default=2)
    p.add_argument("--t-max", default="0", help="int or '<k>L'; 0 -> budget")
    p.add_argument("--arrival-frac", type=float, default=0.1)
    p.add_argument("--criterion", choices=["sum", "max"], default="sum")
    p.add_argument("--mode", choices=["freeze", "fire_only", "freeze_h"], default="freeze")
    p.add_argument("--gate", choices=["freeze", "wavefront"], default="freeze")
    p.add_argument("--eps-arrive", type=float, default=1e-3)
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--tag")
    p.add_argument("--out", type=Path, default=ROOT / "configs" / "run.yaml")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    ds = {"fashion_mnist": "fashion", "mnist": "mnist"}.get(a.dataset, a.dataset)
    ep = f"_ep{a.epochs}" if a.epochs != 1 else ""
    if a.tag:
        tag = a.tag
    elif a.method == "bp":
        tag = f"{ds}_{a.activation}_n{a.width}_l{a.depth}_bp{ep}"
    elif a.method == "pcalm_adaptive":
        tag = f"{ds}_{a.activation}_n{a.width}_l{a.depth}_adaptive_{a.criterion}_tau{a.tau:g}_tmax{a.t_max}{ep}"
    elif a.method == "pcalm_layerwise":
        tag = f"{ds}_{a.activation}_n{a.width}_l{a.depth}_layerwise_{a.mode}_tau{a.tau:g}_tmax{a.t_max}{ep}"
    elif a.method == "pcalm_layerwise_sparse":
        tag = f"{ds}_{a.activation}_n{a.width}_l{a.depth}_sparse_{a.gate}_tau{a.tau:g}_tmax{a.t_max}{ep}"
    else:
        tag = f"{ds}_{a.activation}_n{a.width}_l{a.depth}_{a.method}_{a.budget}{ep}"

    def int_or_mult(v: str):
        return int(v) if v.isdigit() else v

    cfg = {
        "tag": tag,
        "seeds": seeds,
        "dataset": a.dataset,
        "model": {"width": a.width, "depth": a.depth, "activation": a.activation},
        "method": {
            "name": a.method,
            "budget": int_or_mult(a.budget) if a.method != "bp" else 0,
            "alpha": a.alpha,
            "rho": a.rho,
            "state_lr": "auto",
            "inner_steps": 1,
            "weight_credit_timing": "pre_dual_energy",
            "tau": a.tau,
            "patience": a.patience,
            "t_min": a.t_min,
            "t_max": int_or_mult(a.t_max),
            "arrival_frac": a.arrival_frac,
            "criterion": a.criterion,
            "eps_arrive": a.eps_arrive,
            "mode": a.mode,
            "gate": a.gate,
        },
        "training": {
            "epochs": a.epochs,
            "batch_size": 64,
            "eta0": 0.001,
            "gamma0": 1.0,
            "train_subset": 60000,
            "test_subset": 10000,
        },
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", encoding="utf-8") as f:
        f.write("# Node configuration read by scripts/run_node.py. Experiment branches change ONLY this file.\n")
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"wrote {a.out} tag={tag}")


if __name__ == "__main__":
    main()
