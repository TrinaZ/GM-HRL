#!/usr/bin/env python3
"""Evolve the GA dispatching-rule baseline on generated training instances
(k = 100..) and save its weights.

  python scripts/train_ga.py --domains genomic --N 50 --out runs/ga_genomic_N50.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from gmhrl.baselines import train_ga  # noqa: E402
from gmhrl.generate import write_split  # noqa: E402
from gmhrl.instance import load_instance  # noqa: E402
from gmhrl.sim import MD2POSSim, SimConfig  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--domains", nargs="+", default=["genomic"])
    p.add_argument("--N", nargs="+", type=int, default=[50])
    p.add_argument("--n_train", type=int, default=10)
    p.add_argument("--pop", type=int, default=20)
    p.add_argument("--gens", type=int, default=15)
    p.add_argument("--sla_weight", type=float, default=1.0)
    p.add_argument("--rho", type=float, default=0.8)
    p.add_argument("--sla_factor", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data", default=str(ROOT / "data" / "generated"))
    p.add_argument("--out", required=True)
    a = p.parse_args()
    scfg = SimConfig(rho=a.rho, sla_factor=a.sla_factor)
    sims = []
    for dom in a.domains:
        for N in a.N:
            for i, path in enumerate(write_split(a.data, dom, N, range(100, 100 + a.n_train))):
                sims.append(MD2POSSim(load_instance(path), scfg, seed=5000 + i))
    res = train_ga(sims, pop=a.pop, gens=a.gens, sla_weight=a.sla_weight, seed=a.seed, verbose=True)
    res["args"] = vars(a)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2)
    print("weights:", np.round(res["weights"], 3).tolist(), "fitness:", round(res["fitness"], 4))


if __name__ == "__main__":
    main()
