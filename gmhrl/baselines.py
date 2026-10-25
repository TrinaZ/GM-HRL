"""Online baseline dispatchers for the MD2POS simulator.

All baselines are non-delay list schedulers: whenever at least one task is
dispatchable they start the highest-priority one. They only use information
available online (visible tasks, estimated durations, revealed edges).

* FIFO   - earliest release date first.
* SJF    - shortest estimated processing time first.
* HEFT   - online HEFT: highest upward rank on the currently revealed graph
           (rank_u(i) = p_est_i + max over revealed successors rank_u(s)).
           Every task has one eligible processor type with identical
           instances, so the earliest-finish-time instance choice is trivial.
* Random - uniform among dispatchable tasks.
* GA     - a dispatching rule priority(i) = w . f(i) whose weights w are
           evolved by a genetic algorithm on training instances, then applied
           online to unseen test instances.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .sim import MD2POSSim, run_policy, HIDDEN, DONE


# --------------------------------------------------------------------- ranks
def upward_rank(sim: MD2POSSim) -> np.ndarray:
    """Upward rank on the revealed graph, cached per graph version."""
    cache = getattr(sim, "_rank_cache", None)
    if cache is not None and cache[0] == sim.graph_version:
        return cache[1]
    vis = sim.status != HIDDEN
    rank = np.where(vis, sim.p_est, 0.0).astype(np.float64)
    # process in reverse topological order of the visible subgraph;
    # original ids are topologically ordered (u < v); dynamic tasks lie
    # between their parent and the parent's successors.
    order = _topo_visible(sim, vis)
    for u in reversed(order):
        best = 0.0
        for s in sim.succ[u]:
            if vis[s] and rank[s] > best:
                best = rank[s]
        rank[u] = sim.p_est[u] + best
    sim._rank_cache = (sim.graph_version, rank)
    return rank


def _topo_visible(sim: MD2POSSim, vis: np.ndarray) -> List[int]:
    idx = np.nonzero(vis)[0]
    indeg = {int(i): 0 for i in idx}
    for u in idx:
        for s in sim.succ[u]:
            if vis[s]:
                indeg[int(s)] += 1
    stack = [i for i, d in indeg.items() if d == 0]
    out = []
    while stack:
        u = stack.pop()
        out.append(u)
        for s in sim.succ[u]:
            if vis[s]:
                indeg[int(s)] -= 1
                if indeg[int(s)] == 0:
                    stack.append(int(s))
    return out


# ----------------------------------------------------------------- policies
def fifo(sim, D):
    return min(D, key=lambda i: (sim.rel[i], i))


def sjf(sim, D):
    return min(D, key=lambda i: (sim.p_est[i], i))


def heft(sim, D):
    r = upward_rank(sim)
    return max(D, key=lambda i: (r[i], -i))


def make_random(seed: int = 0):
    rng = np.random.default_rng(seed)
    return lambda sim, D: int(D[int(rng.integers(len(D)))])


# ----------------------------------------------------------------------- GA
GA_FEATURES = ("p_est", "rank", "wait", "demand", "n_succ", "rrd_prob", "slack")


def ga_features(sim: MD2POSSim, D: Sequence[int]) -> np.ndarray:
    r = upward_rank(sim)
    D = np.asarray(D)
    scale = max(float(sim.p_est.mean()), 1.0)
    f = np.stack([
        sim.p_est[D] / scale,
        r[D] / (scale * 10.0),
        (sim.t - sim.rel[D]) / sim.sla,
        sim.dem[D].sum(1) / 4.0,
        np.array([len(sim.succ[i]) for i in D]) / 5.0,
        sim.rrd_prob[D],
        (sim.rel[D] + sim.sla - sim.t) / sim.sla,
    ], axis=1)
    return f


def make_ga_policy(w: np.ndarray) -> Callable:
    w = np.asarray(w, dtype=np.float64)

    def pol(sim, D):
        s = ga_features(sim, D) @ w
        return int(D[int(np.argmax(s))])
    return pol


def train_ga(train_sims: List[MD2POSSim], pop: int = 20, gens: int = 15,
             sla_weight: float = 1.0, seed: int = 0, verbose: bool = False) -> Dict:
    """Evolve rule weights on training instances. Fitness = mean over training
    instances of makespan/LB + sla_weight * SLA violation rate (lower is better)."""
    rng = np.random.default_rng(seed)
    k = len(GA_FEATURES)

    def fitness(w):
        vals = []
        for sim in train_sims:
            sim.reset(sim.real_seed)
            m = run_policy(sim, make_ga_policy(w))
            vals.append(m["makespan_over_lb"] + sla_weight * m["sla_violation_rate"])
        return float(np.mean(vals))

    P = rng.normal(0, 1, size=(pop, k))
    P[0] = np.array([0, 1, 0, 0, 0, 0, 0])          # seed with HEFT-like rule
    P[1] = np.array([-1, 0, 0, 0, 0, 0, 0])         # SJF-like
    F = np.array([fitness(w) for w in P])
    hist = [float(F.min())]
    for g in range(gens):
        children = []
        while len(children) < pop:
            a, b = rng.integers(pop, size=2), rng.integers(pop, size=2)
            pa = P[a[np.argmin(F[a])]]
            pb = P[b[np.argmin(F[b])]]
            mask = rng.random(k) < 0.5
            c = np.where(mask, pa, pb) + rng.normal(0, 0.2, size=k) * (rng.random(k) < 0.3)
            children.append(c)
        C = np.array(children)
        FC = np.array([fitness(w) for w in C])
        allP, allF = np.vstack([P, C]), np.concatenate([F, FC])
        keep = np.argsort(allF)[:pop]
        P, F = allP[keep], allF[keep]
        hist.append(float(F.min()))
        if verbose:
            print(f"GA gen {g+1}/{gens}: best fitness {F.min():.4f}")
    best = P[int(np.argmin(F))]
    return {"weights": best.tolist(), "features": list(GA_FEATURES),
            "fitness": float(F.min()), "history": hist}


BASELINES = {"FIFO": fifo, "SJF": sjf, "HEFT": heft}
