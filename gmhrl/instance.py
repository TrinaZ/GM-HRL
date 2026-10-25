"""Benchmark instance loading (MD2POS benchmark v1.0, JSON format).

Each instance file holds N tasks with: sw_type (the single eligible processor
type), proc_time_s (submission-time estimate), cpu/ram/gpu/disk demand
(fractions of one capacity unit), arrival_time_s (release date) and rrd_prob
(probability that the task is a runtime-revealed-dependency branch point that
triggers). Edges are the static precedence pairs known at submission.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import numpy as np

RES = ("cpu", "ram", "gpu", "disk")


@dataclass
class Instance:
    domain: str
    N: int
    seed: int
    sw_names: List[str]
    sw_type: np.ndarray          # [N] int
    p_est: np.ndarray            # [N] float, seconds
    demand: np.ndarray           # [N, 4] float
    release: np.ndarray          # [N] float, seconds
    rrd_prob: np.ndarray         # [N] float
    edges: List[Tuple[int, int]] = field(default_factory=list)
    path: str = ""
    n_removed_back_edges: int = 0

    @property
    def num_sw(self) -> int:
        return len(self.sw_names)


def load_instance(path: str | Path, repair_back_edges: bool = True) -> Instance:
    """Load one benchmark instance.

    repair_back_edges: the v1.0 generator emitted a few edges (u, v) with u > v
    in the ai_ft domain, i.e. a task depending on a task released after it;
    16 of 100 ai_ft instances contain cycles because of them. With repair on
    (dataset v1.1), these edges are dropped and counted in
    ``n_removed_back_edges``.
    """
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    md, T = d["metadata"], d["tasks"]
    T = sorted(T, key=lambda t: t["id"])
    ids = [t["id"] for t in T]
    if ids != list(range(len(T))):
        raise ValueError(f"{path}: task ids must be 0..N-1")
    edges = [(int(u), int(v)) for u, v in d["edges"] if int(u) != int(v)]
    edges = list(dict.fromkeys(edges))
    n_back = 0
    if repair_back_edges:
        n_back = sum(1 for u, v in edges if u > v)
        edges = [(u, v) for u, v in edges if u < v]
    return Instance(
        domain=md["domain"], N=len(T), seed=int(md.get("seed", 0)),
        sw_names=list(md["software_names"]),
        sw_type=np.array([t["sw_type"] for t in T], dtype=np.int64),
        p_est=np.array([t["proc_time_s"] for t in T], dtype=np.float64),
        demand=np.array([[t[r] for r in RES] for t in T], dtype=np.float64),
        release=np.array([t["arrival_time_s"] for t in T], dtype=np.float64),
        rrd_prob=np.array([t.get("rrd_prob", 0.0) for t in T], dtype=np.float64),
        edges=edges, path=str(path), n_removed_back_edges=n_back,
    )


def instance_paths(root: str | Path, domain: str, N: int) -> List[Path]:
    d = Path(root) / domain / f"N{N}"
    return sorted(d.glob("instance_seed*.json"))
