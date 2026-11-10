"""MD2POS-MW: batches of independent workflows of different urgency
(HIERARCHY_SPEC.md, section 1).

A batch holds K workflows. Workflow j of batch k is an instance of the domain
generator with N_w ~ U{10..25} tasks (generator seed 100000 + 100 k + j, disjoint
from the benchmark k = 0..19, validation k = 50..54 and training k >= 100).
All tasks of a workflow are released with the workflow; workflow releases
follow a Poisson process (mean gap 60 s; the simulator later stretches release
dates to the offered load rho). A workflow is STAT (weight 5, deadline factor 1.5)
with probability s and routine (weight 1, factor 2) otherwise; the STAT draw
uses its own random stream, so batches with different s have the same
workflows and releases. Deadlines D_w = r_w + f_w * CP_w are computed by the
simulator from the critical path CP_w under estimated durations.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .generate import generate_json, DOMAIN_OFFSET

STAT_WEIGHT, STAT_FACTOR = 5.0, 1.5
ROUTINE_WEIGHT, ROUTINE_FACTOR = 1.0, 2.0
NW_MIN, NW_MAX = 10, 25
MEAN_GAP_S = 60.0


def generate_mw_json(domain: str, K: int, stat_share: float, k: int) -> dict:
    rng = np.random.default_rng([k, K, DOMAIN_OFFSET[domain], 1])
    rng_stat = np.random.default_rng([k, K, DOMAIN_OFFSET[domain], 2])
    tasks, edges, wfs = [], [], []
    t_rel = 0.0
    sw_names = None
    for j in range(K):
        n_w = int(rng.integers(NW_MIN, NW_MAX + 1))
        sub = generate_json(domain, n_w, 100000 + 100 * k + j)
        sw_names = sub["metadata"]["software_names"]
        off = len(tasks)
        for t in sub["tasks"]:
            t = dict(t)
            t["id"] = off + t["id"]
            t["arrival_time_s"] = round(t_rel, 2)
            t["workflow"] = j
            tasks.append(t)
        for u, v in sub["edges"]:
            if int(u) < int(v):                      # dataset v1.1 rule (drop back edges)
                edges.append([off + int(u), off + int(v)])
        stat = bool(rng_stat.random() < stat_share)
        wfs.append({"id": j, "n_tasks": n_w, "stat": stat, "release_s": round(t_rel, 2),
                    "weight": STAT_WEIGHT if stat else ROUTINE_WEIGHT,
                    "deadline_factor": STAT_FACTOR if stat else ROUTINE_FACTOR})
        t_rel += float(rng.exponential(MEAN_GAP_S))
    return {"metadata": {"domain": domain, "N": len(tasks), "seed": k, "num_sw": len(sw_names),
                         "software_names": sw_names, "kind": "MD2POS-MW", "K": K,
                         "stat_share": stat_share, "workflows": wfs, "n_edges": len(edges)},
            "tasks": tasks, "edges": edges}


def mw_dir(root, domain: str, K: int, stat_share: float) -> Path:
    return Path(root) / domain / f"MW_K{K}_s{stat_share:g}"


def write_split_mw(root, domain: str, K: int, stat_share: float, ks) -> list:
    d = mw_dir(root, domain, K, stat_share)
    d.mkdir(parents=True, exist_ok=True)
    out = []
    for k in ks:
        p = d / f"instance_seed{k:04d}.json"
        if not p.exists():
            with open(p, "w", encoding="utf-8") as f:
                json.dump(generate_mw_json(domain, K, stat_share, k), f)
        out.append(p)
    return out
