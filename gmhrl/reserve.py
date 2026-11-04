"""Capacity reservation for runtime-revealed work (GM-HRL+R).

Specification (fixed before validation; see RESERVATION_SPEC.md):

Branch points. An original task u with rrd_prob > 0 is a branch point. While
u is running, work that u may reveal on completion is anticipated.

Online statistics (computed from the revealed state only, never from hidden
realisations). For every processor type k of branch points:
    n_k  = number of completed branch points of type k
    t_k  = number of those that revealed at least one task
    p_k  = t_k / (n_k + N0)                       (N0 = 5 pseudo-observations
                                                    without a trigger: shrinks
                                                    towards 0 at cold start)
    d_k  = mean over the t_k triggering branch points of the summed demand
           (4 resources) of the tasks they revealed; d_k = 0 until the first
           trigger of type k (no reservation before any evidence).

Reservation. With g_res in [0, 1] set by the Meta-Controller (the level in
force is the one written at the latest Meta-Controller decision; 0 before the
first one):
    R_raw = g_res * sum_{u running branch point} p_type(u) * d_type(u)
    R     = min(R_raw, c_max * capacity, free capacity)        elementwise,
with c_max = 0.25. Hand-off: when a branch point that revealed tasks completes
at time t_c, its last share of R is kept for the decisions taken at the same
time t_c (the revealed tasks become ready in that decision epoch) and is
released as soon as time advances; a branch point that revealed nothing
releases its share at t_c. R is recomputed at every decision; the share of
each branch point is its fraction of R_raw, so concurrent branch points are
accounted for separately and jointly bounded by the caps above.

Dispatch rule. A revealed (dynamic) task may use the full free capacity. An
ordinary task i may start only if d_i <= free - R, unless (a) it has been
ready for longer than Delta t_resp (anti-starvation: aged tasks ignore the
reservation), or (b) no task could otherwise start and no future event exists
(liveness guard). Processor-instance slots are not reserved.

Logging. Every decision with R > 0, every hand-off/release and every dispatch
that consumes reserved capacity is recorded: time, g_res, R_raw, R, number of
running branch points, ordinary tasks blocked by the reservation, released
amounts and amounts used by revealed tasks.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

# status codes shared by MD2POSSim and the live state
HIDDEN, WAITING, READY, RUNNING, DONE = 0, 1, 2, 3, 4

C_MAX = 0.25
PRIOR_N0 = 5.0


class Reservation:
    def __init__(self, c_max: float = C_MAX, prior_n0: float = PRIOR_N0, keep_log: bool = True):
        self.c_max, self.prior_n0, self.keep_log = float(c_max), float(prior_n0), keep_log
        self.log: List[Dict] = []
        self._shares: Dict[int, np.ndarray] = {}     # running branch point -> last share of R
        self._handoff: Dict[int, tuple] = {}          # completed branch point -> (share, t_c)
        self.R = np.zeros(4)
        self.stats = {"decisions": 0, "decisions_with_res": 0, "res_frac_sum": 0.0,
                      "blocked_pairs": 0, "used_frac_sum": 0.0, "released_frac_sum": 0.0,
                      "aged_bypass": 0, "liveness_bypass": 0}
        self._blocked_ids = set()

    # ------------------------------------------------------------- statistics
    @staticmethod
    def _children(state, u) -> List[int]:
        return [int(v) for v in state.succ[u] if state.is_dyn[v] and state.status[v] != HIDDEN]

    def _type_stats(self, state, M):
        st = state.status[:M]
        bp = np.nonzero((state.rrd_prob[:M] > 0) & (~state.is_dyn[:M]))[0]
        done = bp[st[bp] == DONE]
        n, t, dsum = {}, {}, {}
        for u in done:
            k = int(state.sw[u])
            n[k] = n.get(k, 0) + 1
            ch = self._children(state, u)
            if ch:
                t[k] = t.get(k, 0) + 1
                dsum[k] = dsum.get(k, np.zeros(4)) + state.dem[ch].sum(0)
        p = {k: t.get(k, 0) / (n[k] + self.prior_n0) for k in n}
        d = {k: dsum[k] / t[k] for k in t}
        running = bp[st[bp] == RUNNING]
        return p, d, running

    # ---------------------------------------------------------------- filter
    def filter(self, state, D: List[int], g_res: float) -> List[int]:
        """Return the subset of the dispatchable set D allowed under the
        reservation, and update R, shares, hand-offs and the log."""
        M = getattr(state, "M", len(state.status))
        t = float(state.t)
        cap = np.asarray(state.cap, dtype=np.float64)
        free = np.asarray(state.free, dtype=np.float64)
        p, d, running = self._type_stats(state, M)
        ev: Dict = {}

        # branch points that completed since the last decision
        released = []
        for u in list(self._shares):
            if state.status[u] == DONE:
                sh = self._shares.pop(u)
                if self._children(state, u) and abs(float(state.comp[u]) - t) <= 1e-9:
                    self._handoff[u] = (sh, t)
                else:
                    released.append((u, sh))
        for u in list(self._handoff):
            sh, tc = self._handoff[u]
            if t > tc + 1e-9:
                released.append((u, self._handoff.pop(u)[0]))
        if released:
            ev["released"] = [{"bp": int(u), "amount": sh.round(4).tolist()} for u, sh in released]
            self.stats["released_frac_sum"] += float(sum(np.mean(sh / cap) for _, sh in released))

        contrib = {}
        for u in running:
            k = int(state.sw[u])
            if k in d:
                contrib[int(u)] = g_res * p.get(k, 0.0) * d[k]
        R_raw = sum(contrib.values()) if contrib else np.zeros(4)
        hand = sum((sh for sh, _ in self._handoff.values()), np.zeros(4))
        R = np.minimum(np.minimum(R_raw + hand, self.c_max * cap), np.maximum(free, 0.0))
        tot = R_raw + hand
        scale = np.where(tot > 1e-12, R / np.maximum(tot, 1e-12), 0.0)
        self._shares = {u: c * scale for u, c in contrib.items()}
        self._handoff = {u: (sh * scale, tc) for u, (sh, tc) in self._handoff.items()}
        self.R = R
        self.stats["decisions"] += 1
        if not np.any(R > 1e-12):
            if ev and self.keep_log:
                self.log.append({"t": round(t, 3), **ev})
            return list(D)

        avail = free - R
        sla = float(state.sla)
        allowed, blocked = [], []
        for i in D:
            if state.is_dyn[i] or np.all(state.dem[i] <= avail + 1e-9):
                allowed.append(i)
            elif t - float(state.rel[i]) > sla:
                allowed.append(i)
                self.stats["aged_bypass"] += 1
            else:
                blocked.append(int(i))
        if not allowed and not state.has_future_event():
            allowed = list(D)
            blocked = []
            self.stats["liveness_bypass"] += 1
        self.stats["decisions_with_res"] += 1
        self.stats["res_frac_sum"] += float(np.mean(R / cap))
        self.stats["blocked_pairs"] += len(blocked)
        self._blocked_ids.update(blocked)
        if self.keep_log:
            self.log.append({"t": round(t, 3), "g_res": round(float(g_res), 4),
                             "R_raw": np.asarray(R_raw).round(4).tolist(), "R": R.round(4).tolist(),
                             "n_bp_running": len(contrib), "n_handoff": len(self._handoff),
                             "blocked": blocked, **ev})
        return allowed

    def on_dispatch(self, state, i: int):
        """Record reserved capacity consumed by a revealed task."""
        if state.is_dyn[i] and np.any(self.R > 1e-12):
            used = np.minimum(state.dem[i], self.R)
            self.stats["used_frac_sum"] += float(np.mean(used / np.asarray(state.cap)))
            if self.keep_log:
                self.log.append({"t": round(float(state.t), 3), "used_by": int(i), "amount": used.round(4).tolist()})

    def summary(self) -> Dict[str, float]:
        s = self.stats
        n = max(s["decisions"], 1)
        return {"res_share_decisions": s["decisions_with_res"] / n,
                "res_mean_frac": s["res_frac_sum"] / n,
                "res_blocked_pairs": s["blocked_pairs"],
                "res_blocked_tasks": len(self._blocked_ids),
                "res_used_frac_sum": s["used_frac_sum"],
                "res_released_frac_sum": s["released_frac_sum"],
                "res_aged_bypass": s["aged_bypass"],
                "res_liveness_bypass": s["liveness_bypass"]}
