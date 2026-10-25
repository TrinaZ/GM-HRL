"""Event-driven simulator for MD2POS with runtime-revealed dependencies (RRD).

Model (documented here so that it can be stated verbatim in the paper):

* Resources. Four dimensions (cpu, ram, gpu, disk). Task demands in the
  benchmark are fractions of one capacity unit; the system capacity is
  ``capacity * cap_scale`` units per dimension (default 4 units, i.e. 32 cores /
  128 GB / 4 GPUs when one unit = 8 cores / 32 GB / 1 GPU). A task may start
  only if its demand fits into the free capacity in every dimension.
* Processor instances. Each task has a single eligible processor type
  (``sw_type``); each type has ``m_per_type`` identical instances, and an
  instance runs one task at a time. Non-preemptive.
* Online arrivals. Original task i becomes visible at its release date r_i.
  The raw release dates of benchmark v1.0 span about 1,300 s for every N, so
  the offered load grows with N. With ``rho`` set, release dates are stretched
  linearly about the first release so that the submitted work on the
  bottleneck (max over resource dimensions of sum p_i d_ir / R_r, and over
  processor types of sum p_i / M_s) divided by the release window equals rho.
* SLA. Delta t_resp = sla_factor * mean_i p_est_i unless ``sla`` is given;
  task i violates the SLA if C_i - r_i > Delta t_resp.
* Content-dependent durations. The scheduler sees the estimate p_est; the
  actual duration is p_est * eps with eps ~ LogNormal(-sigma^2/2, sigma)
  (mean 1), revealed only at completion.
* RRD. Task b with rrd_prob > 0 is a branch point. When b completes, it
  triggers with probability min(1, rrd_mult * rrd_prob[b]). A trigger inserts
  1 or 2 new tasks d between b and every original successor s of b (edges
  b->d and d->s); each new task takes the type of a successor of b (or of b
  if b has none) and duration/demand drawn from tasks of that type in the
  same instance. The new tasks and edges are unknown before b completes.
* All random outcomes (eps, triggers, inserted tasks) are drawn at reset from
  a generator seeded by (instance seed, realisation seed), so a realisation is
  reproducible and can be exported for an offline clairvoyant bound.

Metrics: makespan = max_i C_i - min_i r_i; SLA violation rate = share of
executed tasks with C_i - r_i > sla; utilisation per dimension = time integral
of used/capacity over [min r_i, max C_i] divided by the makespan.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from .instance import Instance, RES

HIDDEN, WAITING, READY, RUNNING, DONE = 0, 1, 2, 3, 4
WAIT = -1  # action: wait until the next event


@dataclass
class SimConfig:
    capacity: Tuple[float, float, float, float] = (4.0, 4.0, 4.0, 4.0)
    cap_scale: float = 1.0        # resource-scarcity knob (capacity multiplier)
    m_per_type: int = 4
    rho: Optional[float] = 0.8    # offered load; None keeps the raw release dates
    sla_factor: float = 4.0       # Delta t_resp = sla_factor * mean estimated duration
    sla: Optional[float] = None   # explicit Delta t_resp in seconds (overrides sla_factor)
    noise_sigma: float = 0.3
    rrd_enabled: bool = True
    rrd_mult: float = 1.0         # scales branch trigger probabilities (p_rrd sensitivity)
    rrd_max_extra: int = 2
    allow_wait: bool = True

    def to_dict(self):
        return asdict(self)


class MD2POSSim:
    def __init__(self, inst: Instance, cfg: Optional[SimConfig] = None, seed: int = 0):
        self.cfg = cfg or SimConfig()
        self.cap = np.array(self.cfg.capacity, dtype=np.float64) * self.cfg.cap_scale
        if np.any(inst.demand.max(axis=0) > self.cap + 1e-9):
            raise ValueError("a task demand exceeds system capacity")
        if self.cfg.rho is not None:
            inst = _rescale_releases(inst, self.cap, self.cfg.m_per_type, self.cfg.rho)
        self.inst = inst
        self.sla = float(self.cfg.sla) if self.cfg.sla is not None else \
            float(self.cfg.sla_factor * inst.p_est.mean())
        self.reset(seed)

    # ------------------------------------------------------------------ setup
    def _realise(self, seed: int):
        inst, cfg = self.inst, self.cfg
        N = inst.N
        rng = np.random.default_rng([inst.seed, seed, 7919])
        sig = cfg.noise_sigma
        eps = rng.lognormal(-0.5 * sig * sig, sig, size=N) if sig > 0 else np.ones(N)

        succ0: List[List[int]] = [[] for _ in range(N)]
        for u, v in inst.edges:
            succ0[u].append(v)
        by_type: Dict[int, List[int]] = {}
        for i in range(N):
            by_type.setdefault(int(inst.sw_type[i]), []).append(i)

        dyn = []      # (parent b, sw_type, p_est, demand, eps)
        trig = np.zeros(N, dtype=bool)
        if cfg.rrd_enabled:
            for b in range(N):
                p = min(1.0, cfg.rrd_mult * float(inst.rrd_prob[b]))
                if p <= 0 or rng.random() >= p:
                    continue
                trig[b] = True
                k = 1 + int(rng.integers(0, cfg.rrd_max_extra))
                for _ in range(k):
                    if succ0[b]:
                        ty = int(inst.sw_type[succ0[b][int(rng.integers(len(succ0[b])))]])
                    else:
                        ty = int(inst.sw_type[b])
                    src = by_type[ty][int(rng.integers(len(by_type[ty])))]
                    e = rng.lognormal(-0.5 * sig * sig, sig) if sig > 0 else 1.0
                    dyn.append((b, ty, float(inst.p_est[src]), inst.demand[src].copy(), float(e)))
        return eps, succ0, trig, dyn

    def reset(self, seed: int = 0):
        inst, cfg = self.inst, self.cfg
        N = inst.N
        self.real_seed = seed
        eps, succ0, trig, dyn = self._realise(seed)
        M = N + len(dyn)
        self.N0, self.M = N, M
        self.sw = np.concatenate([inst.sw_type, np.array([d[1] for d in dyn], dtype=np.int64)])
        self.p_est = np.concatenate([inst.p_est, np.array([d[2] for d in dyn])])
        self.p_act = self.p_est * np.concatenate([eps, np.array([d[4] for d in dyn])])
        self.dem = np.vstack([inst.demand] + [d[3][None, :] for d in dyn]) if dyn else inst.demand.copy()
        self.rel = np.concatenate([inst.release, np.full(len(dyn), np.inf)])
        self.rrd_prob = np.concatenate([inst.rrd_prob, np.zeros(len(dyn))])
        self.is_dyn = np.concatenate([np.zeros(N, bool), np.ones(len(dyn), bool)])
        self.parent = np.full(M, -1, dtype=np.int64)
        self.trigger = trig
        self.dyn_children: Dict[int, List[int]] = {}
        for k, d in enumerate(dyn):
            self.parent[N + k] = d[0]
            self.dyn_children.setdefault(d[0], []).append(N + k)

        self.succ: List[List[int]] = [list(s) for s in succ0] + [[] for _ in dyn]
        self.pred: List[List[int]] = [[] for _ in range(M)]
        for u, v in inst.edges:
            self.pred[v].append(u)
        self.npred_left = np.array([len(p) for p in self.pred], dtype=np.int64)
        self.dyn_edges: List[Tuple[int, int]] = []

        self.status = np.full(M, HIDDEN, dtype=np.int8)
        self.start = np.full(M, np.nan)
        self.finish = np.full(M, np.inf)
        self.comp = np.full(M, np.nan)
        self.free = self.cap.copy()
        self.free_inst = np.full(inst.num_sw, cfg.m_per_type, dtype=np.int64)
        self.ready: set = set()
        self.running: set = set()
        self._heap: List[Tuple[float, int]] = []
        self._rel_order = np.argsort(inst.release, kind="stable")
        self._rel_ptr = 0
        self._dl_heap: List[Tuple[float, int]] = []

        self.t0 = float(inst.release.min())
        self.t = self.t0
        # integrals and counters (used for metrics and rewards)
        self.util_int = np.zeros(4)
        self.idle_int = 0.0
        self.sla_viol = 0
        self.work_done = 0.0
        self.n_done = 0
        self.n_rrd_events = 0
        self.n_decisions = 0
        self.graph_version = 0
        self.lb_ref = self.lower_bound()
        self.w_ref = float(inst.p_est.sum())
        self._release_until(self.t)

    # ----------------------------------------------------------------- bounds
    def lower_bound(self) -> float:
        """Lower bound on the makespan of the submission-time instance (estimates,
        no RRD): max of release-aware critical path, resource volume and
        per-type load bounds."""
        inst = self.inst
        N = inst.N
        order = _topo(N, inst.edges)
        est = inst.release.copy()
        succ = [[] for _ in range(N)]
        for u, v in inst.edges:
            succ[u].append(v)
        for u in order:
            for v in succ[u]:
                est[v] = max(est[v], est[u] + inst.p_est[u])
        t0 = inst.release.min()
        cp = float((est + inst.p_est).max() - t0)
        vol = float(((inst.p_est[:, None] * inst.demand).sum(0) / self.cap).max())
        load = np.bincount(inst.sw_type, weights=inst.p_est, minlength=inst.num_sw) / self.cfg.m_per_type
        return max(cp, vol, float(load.max()), 1.0)

    # ------------------------------------------------------------- queries
    def fits(self, i: int) -> bool:
        return self.free_inst[self.sw[i]] > 0 and bool(np.all(self.dem[i] <= self.free + 1e-9))

    def dispatchable(self) -> List[int]:
        return sorted(i for i in self.ready if self.fits(i))

    def has_future_event(self) -> bool:
        return bool(self._heap) or self._rel_ptr < self.N0

    def can_wait(self) -> bool:
        return self.cfg.allow_wait and self.has_future_event()

    def done(self) -> bool:
        return (not self.ready and not self.running and self._rel_ptr >= self.N0
                and not np.any(self.status == WAITING))

    def visible(self) -> np.ndarray:
        return np.nonzero(self.status != HIDDEN)[0]

    # ------------------------------------------------------------- actions
    def dispatch(self, i: int):
        if i not in self.ready or not self.fits(i):
            raise ValueError(f"task {i} is not dispatchable at t={self.t:.1f}")
        self.ready.discard(i)
        self.running.add(i)
        self.status[i] = RUNNING
        self.start[i] = self.t
        self.finish[i] = self.t + self.p_act[i]
        self.free -= self.dem[i]
        self.free_inst[self.sw[i]] -= 1
        heapq.heappush(self._heap, (self.finish[i], i))
        self.n_decisions += 1

    def advance(self) -> bool:
        """Move to the next event time and process it. Returns False if there
        is no future event."""
        cands = []
        if self._heap:
            cands.append(self._heap[0][0])
        if self._rel_ptr < self.N0:
            cands.append(self.inst.release[self._rel_order[self._rel_ptr]])
        if not cands:
            if self.ready:
                raise RuntimeError("deadlock: ready tasks never fit")
            return False
        t1 = max(min(cands), self.t)
        self._integrate(t1)
        self.t = t1
        while self._heap and self._heap[0][0] <= t1 + 1e-9:
            _, i = heapq.heappop(self._heap)
            self._complete(i)
        self._release_until(t1)
        return True

    # ------------------------------------------------------------ internals
    def _integrate(self, t1: float):
        dt = t1 - self.t
        if dt <= 0:
            return
        used = self.cap - self.free
        self.util_int += used * dt
        if self.ready:
            self.idle_int += float(np.mean(self.free / self.cap)) * dt
        while self._dl_heap and self._dl_heap[0][0] <= t1:
            dl, i = heapq.heappop(self._dl_heap)
            if not (self.finish[i] <= dl + 1e-9):
                self.sla_viol += 1

    def _make_visible(self, i: int, t: float):
        self.graph_version += 1
        self.rel[i] = min(self.rel[i], t) if np.isfinite(self.rel[i]) else t
        heapq.heappush(self._dl_heap, (self.rel[i] + self.sla, i))
        if self.npred_left[i] == 0:
            self.status[i] = READY
            self.ready.add(i)
        else:
            self.status[i] = WAITING

    def _release_until(self, t: float):
        while self._rel_ptr < self.N0:
            i = int(self._rel_order[self._rel_ptr])
            if self.inst.release[i] > t + 1e-9:
                break
            self._rel_ptr += 1
            self._make_visible(i, float(self.inst.release[i]))

    def _complete(self, i: int):
        self.running.discard(i)
        self.status[i] = DONE
        self.comp[i] = self.t
        self.free += self.dem[i]
        self.free_inst[self.sw[i]] += 1
        self.work_done += self.p_est[i]
        self.n_done += 1
        # outcome-driven topology update
        if i < self.N0 and self.trigger[i]:
            self.n_rrd_events += 1
            kids = self.dyn_children.get(i, [])
            for d in kids:
                for s in self.succ[i]:
                    if self.is_dyn[s]:
                        continue
                    self.succ[d].append(s)
                    self.pred[s].append(d)
                    self.npred_left[s] += 1
                    self.dyn_edges.append((d, s))
                self.pred[d].append(i)
                self.dyn_edges.append((i, d))
            self.succ[i].extend(kids)
            for d in kids:
                self._make_visible(d, self.t)
        for s in self.succ[i]:
            if self.is_dyn[s]:
                continue
            self.npred_left[s] -= 1
            if self.npred_left[s] == 0 and self.status[s] == WAITING:
                self.status[s] = READY
                self.ready.add(s)

    # ------------------------------------------------------------- results
    def metrics(self) -> Dict[str, float]:
        ex = self.status == DONE
        C = self.comp[ex]
        mk = float(np.nanmax(C) - self.t0) if ex.any() else float("nan")
        resp = C - self.rel[ex]
        util = self.util_int / (self.cap * max(mk, 1e-9))
        wait = self.start[ex] - self.rel[ex]
        return {
            "makespan": mk,
            "makespan_over_lb": mk / self.lb_ref,
            "lb_ref": self.lb_ref,
            "sla_violation_rate": float(np.mean(resp > self.sla + 1e-9)),
            "sla_threshold": self.sla,
            "n_executed": int(ex.sum()),
            "n_rrd_events": int(self.n_rrd_events),
            "n_dynamic_tasks": int(np.sum(ex & self.is_dyn)),
            "util_cpu": float(util[0]), "util_ram": float(util[1]),
            "util_gpu": float(util[2]), "util_disk": float(util[3]),
            "mean_wait": float(np.mean(wait)),
            "mean_response": float(np.mean(resp)),
        }

    def export_realised(self) -> Dict:
        """Realised instance (all tasks that will execute, durations as realised,
        edges including those inserted by RRD events) in the input format of the
        offline clairvoyant bound. Determined at reset; independent of the policy."""
        succ0: List[List[int]] = [[] for _ in range(self.N0)]
        for u, v in self.inst.edges:
            succ0[u].append(v)
        edges = list(self.inst.edges)
        for b, kids in self.dyn_children.items():
            for d in kids:
                edges.append((b, d))
                for s in succ0[b]:
                    edges.append((d, s))
        tasks = []
        for i in range(self.M):
            rel = self.inst.release[i] if i < self.N0 else self.inst.release[self.parent[i]]
            tasks.append({"id": f"t{i}", "release": float(rel),
                          "durations": {self.inst.sw_names[self.sw[i]]: float(math.ceil(self.p_act[i]))},
                          "demand": {r: float(self.dem[i][k]) for k, r in enumerate(RES)}})
        return {"time_unit": "s",
                "capacity": {r: float(self.cap[k]) for k, r in enumerate(RES)},
                "instances": {n: int(self.cfg.m_per_type) for n in self.inst.sw_names},
                "tasks": tasks, "edges": [[f"t{u}", f"t{v}"] for u, v in edges]}


def bottleneck_work(inst: Instance, cap: np.ndarray, m_per_type: int) -> float:
    vol = float(((inst.p_est[:, None] * inst.demand).sum(0) / cap).max())
    load = np.bincount(inst.sw_type, weights=inst.p_est, minlength=inst.num_sw) / m_per_type
    return max(vol, float(load.max()))


def _rescale_releases(inst: Instance, cap: np.ndarray, m_per_type: int, rho: float) -> Instance:
    import dataclasses
    r0 = float(inst.release.min())
    span = float(inst.release.max() - r0)
    if span <= 0:
        return inst
    target = bottleneck_work(inst, cap, m_per_type) / rho
    rel = r0 + (inst.release - r0) * (target / span)
    return dataclasses.replace(inst, release=rel)


def _topo(n: int, edges) -> List[int]:
    succ = [[] for _ in range(n)]
    indeg = np.zeros(n, dtype=np.int64)
    for u, v in edges:
        succ[u].append(v)
        indeg[v] += 1
    stack = [i for i in range(n) if indeg[i] == 0]
    out = []
    while stack:
        u = stack.pop()
        out.append(u)
        for v in succ[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                stack.append(v)
    if len(out) != n:
        raise ValueError("precedence graph has a cycle")
    return out


def run_policy(sim: MD2POSSim, choose, max_steps: int = 10**7) -> Dict[str, float]:
    """Generic rollout: ``choose(sim, D)`` returns a task id from D or WAIT."""
    steps = 0
    while not sim.done():
        D = sim.dispatchable()
        if not D:
            if not sim.advance():
                break
            continue
        a = choose(sim, D)
        if a == WAIT:
            if not sim.advance():
                raise RuntimeError("policy waited with no future event")
        else:
            sim.dispatch(a)
        steps += 1
        if steps > max_steps:
            raise RuntimeError("step limit exceeded")
    return sim.metrics()
