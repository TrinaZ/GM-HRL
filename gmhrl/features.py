"""Observation builder: dynamic dependency graph G_t and global state s_t.

Node set: visible tasks that are not yet completed (waiting, ready, running).
Edges: revealed precedence edges between nodes in the node set, including
edges inserted by RRD events. Completed predecessors are summarised in the
node feature ``npred_left``.

Node features (NODE_DIM = 18 + num_sw_pad):
  0-3   status: waiting, ready, running, dispatchable now
  4     estimated duration p_est / mean p_est
  5-8   resource demand (cpu, ram, gpu, disk) in capacity units
  9     branch probability rrd_prob (0 if not a branch point)
  10    1 if the task was inserted by an RRD event
  11    elapsed fraction of p_est (running tasks), clipped to [0, 3]
  12    age (t - r_i) / Delta t_resp, clipped to [0, 3]
  13    SLA slack (r_i + Delta t_resp - t) / Delta t_resp, clipped to [-3, 1]
  14    upward rank on the revealed graph / (10 * mean p_est)
  15    number of revealed successors / 5
  16    number of unfinished predecessors / 5
  17    goal annotation written by the Meta-Controller (filled by the model)
  18..  processor-type one-hot (padded to NUM_SW_PAD)
Edge features (EDGE_DIM = 2): [1 if inserted by RRD, 1 if source is running]

Global state s_t (GLOBAL_DIM = 15 + NUM_SW_PAD):
  free capacity / capacity (4); free instances / M_s per type (NUM_SW_PAD);
  ready, running, waiting, completed counts / N (4); elapsed time / LB (1);
  share of original tasks not yet released (1); summed demand of ready tasks
  / capacity (4); mean age of ready tasks / Delta t_resp (1).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .sim import MD2POSSim, WAITING, READY, RUNNING, DONE, HIDDEN
from .baselines import upward_rank

NUM_SW_PAD = 8
GOAL_IDX = 17
NODE_DIM = 18 + NUM_SW_PAD
EDGE_DIM = 2
GLOBAL_DIM = 15 + NUM_SW_PAD


@dataclass
class Obs:
    x: torch.Tensor          # [n, NODE_DIM]
    ei: torch.Tensor         # [2, E] long (local indices)
    ea: torch.Tensor         # [E, EDGE_DIM]
    s: torch.Tensor          # [GLOBAL_DIM]
    node_ids: np.ndarray     # [n] task ids
    cand: torch.Tensor       # [k] local indices of dispatchable tasks
    can_wait: bool


def build_obs(sim: MD2POSSim, D, dynamic_graph: bool = True) -> Obs:
    st = sim.status
    ids = np.nonzero((st == WAITING) | (st == READY) | (st == RUNNING))[0]
    n = len(ids)
    loc = {int(i): k for k, i in enumerate(ids)}
    scale = max(float(sim.inst.p_est.mean()), 1.0)
    sla = sim.sla
    t = sim.t
    rank = upward_rank(sim)
    Dset = set(int(i) for i in D)

    x = np.zeros((n, NODE_DIM), dtype=np.float32)
    s_ = st[ids]
    x[:, 0] = s_ == WAITING
    x[:, 1] = s_ == READY
    x[:, 2] = s_ == RUNNING
    x[:, 3] = [i in Dset for i in ids]
    x[:, 4] = sim.p_est[ids] / scale
    x[:, 5:9] = sim.dem[ids]
    x[:, 9] = sim.rrd_prob[ids]
    x[:, 10] = sim.is_dyn[ids] if dynamic_graph else 0.0
    run = s_ == RUNNING
    el = np.zeros(n)
    el[run] = (t - sim.start[ids[run]]) / np.maximum(sim.p_est[ids[run]], 1e-6)
    x[:, 11] = np.clip(el, 0, 3)
    x[:, 12] = np.clip((t - sim.rel[ids]) / sla, 0, 3)
    x[:, 13] = np.clip((sim.rel[ids] + sla - t) / sla, -3, 1)
    x[:, 14] = rank[ids] / (10 * scale)
    x[:, 15] = [len(sim.succ[i]) / 5 for i in ids]
    x[:, 16] = sim.npred_left[ids] / 5
    sw = np.clip(sim.sw[ids], 0, NUM_SW_PAD - 1)
    x[np.arange(n), 18 + sw] = 1.0

    src, dst, ea = [], [], []
    dyn_set = set(sim.dyn_edges) if dynamic_graph else set()
    for i in ids:
        li = loc[int(i)]
        for sct in sim.succ[i]:
            lj = loc.get(int(sct))
            if lj is None:
                continue
            is_dyn_edge = (int(i), int(sct)) in dyn_set
            if not dynamic_graph and (sim.is_dyn[i] or sim.is_dyn[sct]):
                continue
            src.append(li); dst.append(lj)
            ea.append((float(is_dyn_edge), float(st[i] == RUNNING)))
    ei = torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros(2, 0, dtype=torch.long)
    eat = torch.tensor(ea, dtype=torch.float32) if ea else torch.zeros(0, EDGE_DIM)

    g = np.zeros(GLOBAL_DIM, dtype=np.float32)
    g[0:4] = sim.free / sim.cap
    nsw = min(sim.inst.num_sw, NUM_SW_PAD)
    g[4:4 + nsw] = sim.free_inst[:nsw] / sim.cfg.m_per_type
    o = 4 + NUM_SW_PAD
    N0 = sim.N0
    g[o + 0] = len(sim.ready) / N0
    g[o + 1] = len(sim.running) / N0
    g[o + 2] = float(np.sum(st == WAITING)) / N0
    g[o + 3] = sim.n_done / N0
    g[o + 4] = (t - sim.t0) / sim.lb_ref
    g[o + 5] = (N0 - sim._rel_ptr) / N0
    if sim.ready:
        r = np.fromiter(sim.ready, dtype=np.int64)
        g[o + 6:o + 10] = sim.dem[r].sum(0) / sim.cap
        g[o + 10] = float(np.clip(np.mean((t - sim.rel[r]) / sla), 0, 3))
    cand = torch.tensor([loc[int(i)] for i in D], dtype=torch.long)
    return Obs(torch.from_numpy(x), ei, eat, torch.from_numpy(g), ids, cand, sim.can_wait())
