"""ORHG: online rolling-horizon exact re-planning baseline.

At a decision point, if the revealed graph has changed since the last plan
(new arrivals or an RRD event) and at least ``replan_interval`` simulated
seconds have passed (RRD events always trigger a re-plan), the residual
problem is solved with CP-SAT under a wall-clock limit (default 1 s):

* tasks: visible tasks not yet started, with ESTIMATED durations;
* running tasks occupy their resources and instance until their estimated end
  (at least 1 s from now);
* constraints: release dates, revealed precedences, one cumulative constraint
  per resource dimension and per processor type (M_s instances);
* objective: minimise the estimated makespan (ties broken by summed start times).

The plan's start times are used as priorities: at every decision the
dispatchable task with the earliest planned start is dispatched (non-delay).
If no plan is available (no feasible solution within the limit) the online
HEFT rule is used.
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np

from .baselines import upward_rank
from .sim import MD2POSSim, WAITING, READY, RUNNING


def _solve(sim: MD2POSSim, time_limit: float, workers: int, scale: float = 100.0) -> Optional[Dict[int, int]]:
    from ortools.sat.python import cp_model
    t = sim.t
    P = [int(i) for i in np.nonzero((sim.status == WAITING) | (sim.status == READY))[0]]
    R = [int(i) for i in sim.running]
    if not P:
        return {}
    dur = {i: max(1, int(math.ceil(sim.p_est[i]))) for i in P}
    rem = {i: max(1, int(math.ceil(sim.start[i] + sim.p_est[i] - t))) for i in R}
    H = int(sum(dur.values()) + (max(rem.values()) if rem else 0)
            + max(0.0, float(max(sim.rel[i] for i in P)) - t) + 1)
    m = cp_model.CpModel()
    s, e, iv = {}, {}, {}
    for i in P:
        lo = int(max(0.0, math.ceil(sim.rel[i] - t)))
        s[i] = m.NewIntVar(lo, H, f"s{i}")
        e[i] = m.NewIntVar(lo + dur[i], H + dur[i], f"e{i}")
        iv[i] = m.NewIntervalVar(s[i], dur[i], e[i], f"iv{i}")
    fixed = {i: m.NewFixedSizeIntervalVar(0, rem[i], f"r{i}") for i in R}
    Pset = set(P)
    for u in P + R:
        for v in sim.succ[u]:
            if v in Pset:
                m.Add(s[v] >= (e[u] if u in Pset else rem[u]))
    for k in range(4):
        cap = int(math.floor(sim.cap[k] * scale + 1e-6))
        ivs = [iv[i] for i in P] + [fixed[i] for i in R]
        dem = [int(round(sim.dem[i][k] * scale)) for i in P] + [int(round(sim.dem[i][k] * scale)) for i in R]
        keep = [j for j, d in enumerate(dem) if d > 0]
        if keep:
            m.AddCumulative([ivs[j] for j in keep], [dem[j] for j in keep], cap)
    for ty in range(sim.inst.num_sw):
        ivs = [iv[i] for i in P if sim.sw[i] == ty] + [fixed[i] for i in R if sim.sw[i] == ty]
        if ivs:
            m.AddCumulative(ivs, [1] * len(ivs), sim.cfg.m_per_type)
    cmax = m.NewIntVar(0, 2 * H, "cmax")
    m.AddMaxEquality(cmax, [e[i] for i in P] + [m.NewConstant(rem[i]) for i in R])
    m.Minimize(cmax * (len(P) + 1) * H + sum(s.values()) if H * (len(P) + 1) < 2**40 else cmax)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit)
    solver.parameters.num_search_workers = int(workers)
    st = solver.Solve(m)
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None
    return {i: int(solver.Value(s[i])) + int(t) for i in P}


def make_orhg_policy(time_limit: float = 1.0, workers: int = 1, replan_interval_factor: float = 0.5):
    state = {"plan": None, "version": -1, "last_t": -math.inf, "rrd": -1, "sim": None,
             "n_solves": 0, "n_fallback": 0}

    def pol(sim: MD2POSSim, D):
        if state["sim"] is not sim or sim.t < state["last_t"]:      # new episode
            state.update(plan=None, version=-1, last_t=-math.inf, rrd=-1, sim=sim)
        interval = replan_interval_factor * float(sim.inst.p_est.mean())
        changed = sim.graph_version != state["version"]
        rrd_event = sim.n_rrd_events != state["rrd"]
        if state["plan"] is None or rrd_event or (changed and sim.t - state["last_t"] >= interval):
            plan = _solve(sim, time_limit, workers)
            state["n_solves"] += 1
            if plan is None:
                state["n_fallback"] += 1
            state.update(plan=plan, version=sim.graph_version, last_t=sim.t, rrd=sim.n_rrd_events)
        plan = state["plan"]
        r = upward_rank(sim)
        if plan:
            return min(D, key=lambda i: (plan.get(int(i), math.inf), -r[i], i))
        return max(D, key=lambda i: (r[i], -i))

    pol.state = state
    return pol
