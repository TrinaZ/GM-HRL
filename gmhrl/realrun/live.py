"""Live system state with the same interface as MD2POSSim, so that FIFO, HEFT,
ORHG and the trained GM-HRL policy can be used unchanged on a real pipeline.

Time is measured in seconds since the start of the run. Static tasks of all
samples are created up front (hidden until the sample's arrival time);
dynamic tasks are created when a branch task reveals them.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..sim import HIDDEN, WAITING, READY, RUNNING, DONE
from .spec import PipelineSpec


class LiveState:
    def __init__(self, spec: PipelineSpec, samples: List[Tuple[str, float]]):
        self.spec = spec
        self.samples = samples
        self.type_idx = {n: i for i, n in enumerate(spec.model_types)}
        nst = len(spec.stages)
        self._cap_n = max(16, 2 * len(samples) * nst)
        self._alloc(self._cap_n)
        self.M = 0
        self.cap = np.array(spec.capacity_units(), dtype=np.float64)
        self.free = self.cap.copy()
        self.free_inst = np.full(len(spec.model_types), spec.instances_per_stage, dtype=np.int64)
        self.cfg = SimpleNamespace(m_per_type=spec.instances_per_stage, allow_wait=True)
        self.sla = spec.sla_seconds
        self.ready, self.running = set(), set()
        self.dyn_edges: List[Tuple[int, int]] = []
        self.graph_version = 0
        self.n_done = 0
        self.n_rrd_events = 0
        self.t0, self.t = 0.0, 0.0
        self.real_seed = 0
        self.task_info: List[Dict] = []
        self.sample_tasks: Dict[int, Dict[str, int]] = {}
        # static tasks for every sample
        static = spec.static_stages()
        for si, (sid, arr) in enumerate(samples):
            ids = {}
            for st in static:
                ids[st.name] = self._new_task(si, st, float(arr), dynamic=False)
            for st in static:
                for a in st.after:
                    self._edge(ids[a], ids[st.name])
            self.sample_tasks[si] = ids
        self.N0 = self.M
        self._rel_ptr = 0
        self._order = sorted(range(self.N0), key=lambda i: (self.rel[i], i))
        self.inst = SimpleNamespace(p_est=self.p_est[:self.N0].copy(), num_sw=len(spec.model_types),
                                    sw_names=list(spec.model_types))
        self.lb_ref = self._lower_bound()

    # ---------------------------------------------------------------- storage
    def _alloc(self, n):
        def grow(a, fill, dtype, shape=None):
            new = np.full((n,) + (shape or ()), fill, dtype=dtype)
            if a is not None:
                new[:len(a)] = a
            return new
        g = lambda name: getattr(self, name, None)
        self.status = grow(g("status"), HIDDEN, np.int8)
        self.sw = grow(g("sw"), 0, np.int64)
        self.p_est = grow(g("p_est"), 0.0, np.float64)
        self.dem = grow(g("dem"), 0.0, np.float64, (4,))
        self.rel = grow(g("rel"), np.inf, np.float64)
        self.rrd_prob = grow(g("rrd_prob"), 0.0, np.float64)
        self.is_dyn = grow(g("is_dyn"), False, bool)
        self.npred_left = grow(g("npred_left"), 0, np.int64)
        self.start = grow(g("start"), np.nan, np.float64)
        self.comp = grow(g("comp"), np.nan, np.float64)
        self.succ = (g("succ") or []) + [[] for _ in range(n - len(g("succ") or []))]
        self.pred = (g("pred") or []) + [[] for _ in range(n - len(g("pred") or []))]
        self._cap_n = n

    def _new_task(self, si: int, st, rel: float, dynamic: bool) -> int:
        if self.M >= self._cap_n:
            self._alloc(2 * self._cap_n)
        i = self.M
        self.M += 1
        self.sw[i] = self.type_idx[st.model_type]
        self.p_est[i] = st.est_seconds
        self.dem[i] = self.spec.demand_units(st)
        self.rel[i] = rel
        self.rrd_prob[i] = st.rrd_prob
        self.is_dyn[i] = dynamic
        self.task_info.append({"task": i, "sample": self.samples[si][0], "sample_idx": si,
                               "stage": st.name, "model_type": st.model_type, "dynamic": dynamic,
                               "request": dict(st.request), "est_seconds": st.est_seconds})
        return i

    def _edge(self, u: int, v: int):
        self.succ[u].append(v)
        self.pred[v].append(u)
        if self.status[u] != DONE:
            self.npred_left[v] += 1

    def _lower_bound(self) -> float:
        N = self.N0
        est = self.rel[:N].copy()
        order = _topo(N, self.succ)
        for u in order:
            for v in self.succ[u]:
                if v < N:
                    est[v] = max(est[v], est[u] + self.p_est[u])
        cp = float((est + self.p_est[:N]).max() - self.rel[:N].min())
        vol = float(((self.p_est[:N, None] * self.dem[:N]).sum(0) / self.cap).max())
        load = np.bincount(self.sw[:N], weights=self.p_est[:N], minlength=len(self.spec.model_types))
        return max(cp, vol, float(load.max() / self.spec.instances_per_stage), 1.0)

    # ------------------------------------------------------- MD2POSSim interface
    def fits(self, i: int) -> bool:
        return self.free_inst[self.sw[i]] > 0 and bool(np.all(self.dem[i] <= self.free + 1e-9))

    def dispatchable(self) -> List[int]:
        return sorted(i for i in self.ready if self.fits(i))

    def has_future_event(self) -> bool:
        return bool(self.running) or self._rel_ptr < self.N0

    def can_wait(self) -> bool:
        return self.has_future_event()

    def done(self) -> bool:
        return (not self.ready and not self.running and self._rel_ptr >= self.N0
                and not np.any(self.status[:self.M] == WAITING))

    # ---------------------------------------------------------------- updates
    def release_until(self, t: float) -> List[int]:
        out = []
        while self._rel_ptr < self.N0 and self.rel[self._order[self._rel_ptr]] <= t + 1e-9:
            i = self._order[self._rel_ptr]
            self._rel_ptr += 1
            if self.status[i] != HIDDEN:          # cancelled before release
                continue
            self._make_visible(i)
            out.append(i)
        return out

    def _make_visible(self, i: int):
        self.graph_version += 1
        if self.npred_left[i] == 0:
            self.status[i] = READY
            self.ready.add(i)
        else:
            self.status[i] = WAITING

    def start_task(self, i: int, t: float):
        assert i in self.ready and self.fits(i)
        self.ready.discard(i)
        self.running.add(i)
        self.status[i] = RUNNING
        self.start[i] = t
        self.free -= self.dem[i]
        self.free_inst[self.sw[i]] -= 1

    def complete_task(self, i: int, t: float, reveal: Optional[List[str]] = None) -> List[int]:
        """Mark task i finished; insert revealed dynamic stages between i and its
        static successors. Returns the ids of the new tasks."""
        self.running.discard(i)
        self.status[i] = DONE
        self.comp[i] = t
        self.free += self.dem[i]
        self.free_inst[self.sw[i]] += 1
        self.n_done += 1
        new = []
        if reveal:
            self.n_rrd_events += 1
            si = self.task_info[i]["sample_idx"]
            static_succ = [v for v in self.succ[i] if not self.is_dyn[v]]
            for name in reveal:
                d = self._new_task(si, self.spec.stages[name], t, dynamic=True)
                self._edge(i, d)
                self.dyn_edges.append((i, d))
                for s in static_succ:
                    self._edge(d, s)
                    self.dyn_edges.append((d, s))
                self.sample_tasks[si][name] = d
                new.append(d)
            for d in new:
                self._make_visible(d)
        for s in self.succ[i]:
            if s in new:
                continue
            self.npred_left[s] -= 1
            if self.npred_left[s] == 0 and self.status[s] == WAITING:
                self.status[s] = READY
                self.ready.add(s)
        return new

    def cancel_sample(self, si: int, t: float) -> List[int]:
        """After a failure: drop all not-yet-started tasks of the sample."""
        out = []
        for i in range(self.M):
            if self.task_info[i]["sample_idx"] == si and self.status[i] in (HIDDEN, WAITING, READY):
                self.ready.discard(i)
                self.status[i] = DONE
                self.comp[i] = np.nan
                self.task_info[i]["cancelled"] = True
                out.append(i)
        return out


def _topo(n: int, succ) -> List[int]:
    indeg = np.zeros(n, dtype=np.int64)
    for u in range(n):
        for v in succ[u]:
            if v < n:
                indeg[v] += 1
    stack = [i for i in range(n) if indeg[i] == 0]
    out = []
    while stack:
        u = stack.pop()
        out.append(u)
        for v in succ[u]:
            if v < n:
                indeg[v] -= 1
                if indeg[v] == 0:
                    stack.append(v)
    if len(out) != n:
        raise ValueError("pipeline stages contain a cycle")
    return out
