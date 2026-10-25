"""Deterministic wrapper around the MD2POS benchmark v1.0 generator.

The original script seeded instance k with ``k * 1000 + hash(domain) % 1000``.
Python randomises string hashes per process, so the benchmark could not be
regenerated. The offsets below were recovered by search and reproduce the
published benchmark exactly (verified on 264 instances across all domains and
scales). Benchmark (test) instances use k = 0..19; training instances use
k >= 100, so they never coincide with test instances.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "scripts"))
import generate_dataset_v1 as _G  # noqa: E402

DOMAIN_OFFSET = {"genomic": 851, "video": 176, "ai_ft": 496, "financial": 719}
DOMAINS = tuple(DOMAIN_OFFSET)


def generate_json(domain: str, N: int, k: int) -> Dict:
    inst = _G.generate_instance(domain, N, seed=k * 1000 + DOMAIN_OFFSET[domain])
    return {
        "metadata": {"domain": domain, "N": N, "seed": k, "num_sw": inst["num_sw"],
                     "software_names": inst["software_names"], "dag_type": inst["dag_type"],
                     "p_rrd": inst["p_rrd"], "n_edges": len(inst["edges"])},
        "tasks": [{
            "id": i,
            "sw_type": int(inst["sw_types"][i]),
            "sw_name": inst["software_names"][int(inst["sw_types"][i])],
            "proc_time_s": round(float(inst["proc_times"][i]), 2),
            "cpu": round(float(inst["cpu_demand"][i]), 3),
            "ram": round(float(inst["ram_demand"][i]), 3),
            "gpu": round(float(inst["gpu_demand"][i]), 3),
            "disk": round(float(inst["disk_demand"][i]), 3),
            "priority": round(float(inst["priorities"][i]), 3),
            "arrival_time_s": round(float(inst["arrival_times"][i]), 2),
            "rrd_prob": round(float(inst["rrd_probs"][i]), 3),
        } for i in range(N)],
        "edges": [[int(u), int(v)] for u, v in inst["edges"]],
    }


def write_split(root: str | Path, domain: str, N: int, ks) -> list:
    d = Path(root) / domain / f"N{N}"
    d.mkdir(parents=True, exist_ok=True)
    out = []
    for k in ks:
        p = d / f"instance_seed{k:04d}.json"
        if not p.exists():
            with open(p, "w", encoding="utf-8") as f:
                json.dump(generate_json(domain, N, k), f)
        out.append(p)
    return out
