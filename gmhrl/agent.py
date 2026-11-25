"""GM-HRL rollouts, rewards and hierarchical PPO (H-PPO).

Rewards (all terms normalised per instance, so the same weights apply at
every scale). Between two consecutive Sub-Controller decisions at times
t_k < t_{k+1}:

  progress  dC_k = (W(t_{k+1}) - W(t_k)) / W_ref - (t_{k+1} - t_k) / LB
  util      U_k  = time integral over [t_k, t_{k+1}] of mean_r used_r/R_r, / LB
  SLA       P_k  = number of tasks whose response time exceeded Delta t_resp
                   during [t_k, t_{k+1}], / N
  idleness  I_k  = time integral of mean_r free_r/R_r while tasks were ready, / LB
  deadline  M_k  = (MD2POS-MW only) summed weight of workflows whose deadline
                   passes in (t_k, t_{k+1}] before they complete, / total weight

  b_k   = w1*dC_k + w2*U_k - kappa*w3*P_k - w4*I_k - w6*M_k
  r^M   = sum of b_k over the Meta-Controller interval
  r^S_k = eta1*b_k + (eta2*Sim_k - eta3*Frag_k) / N

v3 (HIERARCHY_SPEC_v3.md): goal_mode="none", res_goal=True.
  Meta-Controller outputs g; r_t = net.resource_goal(g) in [0,1]^4 is the
  target utilisation per resource over [t, t+Delta t_H).
  Sub-Controller logits of task i are penalised by
    lambda_res * sum_k |util_after(i)_k - r_t_k|,
  where util_after(i) = (cap - free + dem[i]) / cap.
  Sub reward: the Sim term is replaced by
    -eta_cap * mean_k (util_mean_k - r_t_k)^2, where util_mean is the average
    utilisation of the interval since the last Sub decision.
  Meta reward: b_k gains -w_r * mean_k (util_mean_k - r_t_k)^2, with util_mean
    averaged over the current Meta interval.
  res_head is in the Meta optimiser; it is trained only by the w_r loss in
  update(phase="meta"), on the detached g of each Meta transition.

v2 (HIERARCHY_SPEC.md): goal_mode="workflow", eta2 = 1.0, meta_bc=True.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional

import numpy as np
import torch

from .features import build_obs, NODE_DIM, NODE_DIM_MW, GOAL_IDX
from .model import GMHRLNet
from .reserve import Reservation, C_MAX, PRIOR_N0
from .sim import MD2POSSim, WAIT


def _res_penalty(cfg: "AgentConfig", util_after_t: torch.Tensor, r_t: torch.Tensor) -> Optional[torch.Tensor]:
    """v3: lambda_res * sum_k |util_after_k - r_t_k|, shape [k]."""
    if not getattr(cfg, "res_goal", False) or r_t is None:
        return None
    return cfg.lambda_res * torch.abs(util_after_t - r_t).sum(-1)


@dataclass
class RewardConfig:
    w1: float = 1.0
    w2: float = 0.1
    w3: float = 1.0
    w4: float = 0.1
    kappa: float = 1.0
    eta1: float = 1.0
    eta2: float = 0.1
    eta3: float = 0.1
    w5: float = 0.0
    w6: float = 0.0
    eta_cap: float = 0.0   # v3: cap-compliance weight (replaces eta2*Sim when > 0)
    w_r: float = 0.0       # v3: Meta reward weight for mean_k (util_mean_k - r_t_k)^2


@dataclass
class AgentConfig:
    d: int = 64
    layers: int = 2
    d_g: int = 8
    hidden: int = 128
    use_gnn: bool = True
    hierarchical: bool = True
    dynamic_graph: bool = True
    meta_interval_factor: float = 2.0
    lr: float = 3e-4
    gamma_s: float = 0.995
    gamma_m: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    epochs: int = 4
    minibatch: int = 64
    max_grad_norm: float = 0.5
    alt_period: int = 5
    reserve: bool = False
    res_c_max: float = C_MAX
    res_prior_n0: float = PRIOR_N0
    mw: bool = False
    goal_mode: str = "task"
    meta_bc: bool = False
    device: str = "cpu"
    res_goal: bool = False
    n_resource: int = 4
    lambda_res: float = 1.0
    sub_obs: str = "full"


def _snap(sim: MD2POSSim):
    return (sim.t, sim.work_done, sim.util_int.copy(), sim.sla_viol, sim.idle_int, sim.flow_int,
            sim.wf_miss_w if sim.mw else 0.0)


def _base_reward(sim: MD2POSSim, s0, rc: RewardConfig, r_t: Optional[np.ndarray] = None) -> float:
    t0, w0, u0, v0, i0, f0, m0 = s0
    prog = (sim.work_done - w0) / sim.w_ref - (sim.t - t0) / sim.lb_ref
    util = float(np.mean((sim.util_int - u0) / sim.cap)) / sim.lb_ref
    sla = (sim.sla_viol - v0) / sim.N0
    idle = (sim.idle_int - i0) / sim.lb_ref
    resp = (sim.flow_int - f0) / (sim.N0 * sim.p_mean_ref)
    miss = (sim.wf_miss_w - m0) / sim.wf_wsum if sim.mw else 0.0
    base = (rc.w1 * prog + rc.w2 * util - rc.kappa * rc.w3 * sla
            - rc.w4 * idle - rc.w5 * resp - rc.w6 * miss)
    if rc.w_r > 0 and r_t is not None:
        dt = sim.t - t0
        if dt > 0:
            util_mean = (sim.util_int - u0) / (sim.cap * dt)
            base += -rc.w_r * float(np.mean((util_mean - r_t) ** 2))
    return base


def _frag(sim: MD2POSSim) -> float:
    if not sim.ready:
        return 0.0
    if any(sim.fits(i) for i in sim.ready):
        return 0.0
    return float(np.mean(sim.free / sim.cap))


def _wf_rank_target(sim: MD2POSSim, wf: np.ndarray):
    W, first = np.unique(wf, return_index=True)
    order = sorted(range(len(W)), key=lambda j: (-sim.wf_weight[W[j]], sim.wf_deadline[W[j]], W[j]))
    tgt = np.zeros(len(W), dtype=np.float32)
    n = len(W)
    for r, j in enumerate(order):
        tgt[j] = 1.0 - r / (n - 1) if n > 1 else 1.0
    return torch.from_numpy(first.astype(np.int64)), torch.from_numpy(tgt)


def _shuffle_goals(p: torch.Tensor, wf: np.ndarray, key: np.ndarray) -> torch.Tensor:
    W, first, inv = np.unique(wf, return_index=True, return_inverse=True)
    vals = p[torch.from_numpy(first.astype(np.int64))]
    dest = np.argsort(key[W], kind="stable")
    new = torch.empty_like(vals)
    new[torch.from_numpy(dest.astype(np.int64))] = vals
    return new[torch.from_numpy(inv.astype(np.int64))]


class GMHRLAgent:
    def __init__(self, cfg: AgentConfig, rcfg: Optional[RewardConfig] = None):
        self.cfg, self.rcfg = cfg, rcfg or RewardConfig()
        self.use_res = bool(cfg.reserve and cfg.hierarchical)
        self.net = GMHRLNet(cfg.d, cfg.layers, cfg.d_g, cfg.hidden,
                            cfg.use_gnn, cfg.hierarchical, n_res=1 if self.use_res else 0,
                            node_dim=NODE_DIM_MW if cfg.mw else NODE_DIM,
                            goal_mode=cfg.goal_mode,
                            res_goal=cfg.res_goal, n_resource=cfg.n_resource, sub_obs=getattr(cfg, "sub_obs", "full")).to(cfg.device)
        self.goal_test = "own"
        self.measure_goal = False
        self.measure_res_neutral = False
        self.last_reservation: Optional[Reservation] = None
        n = self.net
        sub_params = (list(n.enc.parameters()) + list(n.Wg.parameters()) + list(n.Wx.parameters())
                      + list(n.ctx.parameters()) + list(n.score.parameters())
                      + list(n.wait.parameters()) + list(n.sub_v.parameters()))
        meta_params = list(n.meta_mu.parameters()) + [n.meta_logstd] + list(n.meta_v.parameters())
        if cfg.res_goal:
            # v3: res_head is a head of the Meta-Controller, trained only by the w_r loss
            meta_params += list(n.res_head.parameters())
        self.opt_sub = torch.optim.Adam(sub_params, lr=cfg.lr, eps=1e-5)
        self.opt_meta = torch.optim.Adam(meta_params, lr=cfg.lr, eps=1e-5)
        self.iteration = 0

    # ---------------------------------------------------------------- rollout
    def rollout(self, sim: MD2POSSim, seed: int, train: bool = True, expert=None) -> Dict:
        cfg, rc, net = self.cfg, self.rcfg, self.net
        dev = cfg.device
        sim.reset(seed)
        meta_dt = cfg.meta_interval_factor * float(sim.inst.p_est.mean())
        g = None
        r_t = None
        last_meta_t = -math.inf
        sub_tr: List[Dict] = []
        meta_tr: List[Dict] = []
        prev = None
        latencies = []
        res = Reservation(cfg.res_c_max, cfg.res_prior_n0, keep_log=not train) if self.use_res else None
        self.last_reservation = res
        gt = "own" if train else self.goal_test
        g_rng = np.random.default_rng(seed)
        t_rng = torch.Generator().manual_seed(seed)
        shuf_key = None
        n_goal, n_changed = 0, 0
        meta_bc = bool(cfg.meta_bc and expert is not None and cfg.hierarchical)

        def get_D():
            D0 = sim.dispatchable()
            return res.filter(sim, D0, net.g_res(g)) if (res is not None and D0) else D0

        while True:
            D = get_D()
            while not D and not sim.done():
                if not sim.advance():
                    break
                D = get_D()
            if prev is not None:
                b = _base_reward(sim, prev[1], rc, prev[2])
                tr = sub_tr[prev[0]]
                if rc.eta_cap > 0 and prev[2] is not None:
                    dt = sim.t - prev[1][0]
                    if dt > 0:
                        util_mean = (sim.util_int - prev[1][2]) / (sim.cap * dt)
                        cap_term = -rc.eta_cap * float(np.mean((util_mean - prev[2]) ** 2))
                    else:
                        cap_term = 0.0
                    tr["r"] = rc.eta1 * b + (cap_term - rc.eta3 * tr["frag"]) / sim.N0
                else:
                    tr["r"] = rc.eta1 * b + (rc.eta2 * tr["sim"] - rc.eta3 * tr["frag"]) / sim.N0
                if meta_tr:
                    meta_tr[-1]["r"] += b
                    if rc.w_r > 0 and meta_tr[-1]["util_int_accum"] is not None:
                        dt = sim.t - prev[1][0]
                        if dt > 0:
                            meta_tr[-1]["util_int_accum"] += sim.util_int - prev[1][2]
                            meta_tr[-1]["dt_accum"] += dt
            if sim.done() or not D:
                break
            t_start = time.perf_counter()
            obs = build_obs(sim, D, cfg.dynamic_graph, mw=cfg.mw)
            x, ei, ea, s = obs.x.to(dev), obs.ei.to(dev), obs.ea.to(dev), obs.s.to(dev)
            with torch.no_grad():
                if cfg.hierarchical and (sim.t - last_meta_t >= meta_dt or g is None):
                    # close the previous Meta interval (finalise its util_mean)
                    if meta_tr and rc.w_r > 0 and meta_tr[-1]["util_mean"] is None \
                            and meta_tr[-1]["dt_accum"] > 0:
                        meta_tr[-1]["util_mean"] = (
                            meta_tr[-1]["util_int_accum"]
                            / (sim.cap * meta_tr[-1]["dt_accum"])).astype(np.float32)
                    g_in = g
                    _, z = net.encode(x, ei, ea, g_in)
                    dist, v = net.meta_dist(s, z)
                    g = dist.sample() if train else dist.mean
                    if gt == "random":
                        g = torch.randn(g.shape, generator=t_rng).to(dev)
                    if gt == "shuffle":
                        shuf_key = g_rng.random(sim.wf_K)
                    r_t = net.resource_goal(g) if cfg.res_goal else None
                    tr_m = {"obs": obs, "g_in": g_in, "g": g,
                            "g_det": g.detach(),
                            "logp": float(dist.log_prob(g).sum()),
                            "v": float(v), "r": 0.0,
                            "r_t_np": None if r_t is None else r_t.detach().cpu().numpy().copy(),
                            "util_int_accum": np.zeros_like(sim.cap) if rc.w_r > 0 else None,
                            "dt_accum": 0.0,
                            "util_mean": None}
                    if meta_bc:
                        tr_m["target"] = _wf_rank_target(sim, obs.wf)
                    meta_tr.append(tr_m)
                    last_meta_t = sim.t
                p_ov = None
                if gt == "neutral":
                    p_ov = torch.full((x.size(0),), 0.5, device=dev)
                elif gt == "shuffle" and cfg.hierarchical and g is not None:
                    p_ov = _shuffle_goals(net.goal_annotation(x, g), obs.wf, shuf_key)
                h, z = net.encode(x, ei, ea, g, p_ov)
                logits, v = net.sub_logits(h, z, s, obs.cand.to(dev), obs.can_wait)
                util_after = None
                if cfg.res_goal and r_t is not None:
                    ids = obs.node_ids[obs.cand.numpy()]
                    util_after = (sim.cap - sim.free + sim.dem[ids]) / sim.cap
                    ua_t = torch.tensor(util_after, dtype=torch.float32, device=dev)
                    pen = _res_penalty(cfg, ua_t, r_t)
                    if pen is not None:
                        logits = logits.clone()
                        logits[:len(obs.cand)] -= pen
                dist = torch.distributions.Categorical(logits=logits)
                if expert is not None:
                    tid = expert(sim, D)
                    a = int(np.nonzero(obs.node_ids[obs.cand.numpy()] == tid)[0][0])
                else:
                    a = int(dist.sample()) if train else int(torch.argmax(logits))
                logp = float(dist.log_prob(torch.tensor(a)))
                if cfg.hierarchical and a < len(obs.cand):
                    p = net.goal_annotation(x, g)[obs.cand.to(dev)]
                    simv = float(p[a] / p.max().clamp(min=1e-6))
                else:
                    simv = 0.0
            latencies.append(time.perf_counter() - t_start)
            if self.measure_goal and not train and cfg.hierarchical and expert is None:
                with torch.no_grad():
                    h0, z0 = net.encode(x, ei, ea, g, torch.full((x.size(0),), 0.5, device=dev))
                    l0, _ = net.sub_logits(h0, z0, s, obs.cand.to(dev), obs.can_wait)
                n_goal += 1
                n_changed += int(int(torch.argmax(l0)) != a)
            if self.measure_res_neutral and not train and cfg.res_goal and r_t is not None and expert is None:
                with torch.no_grad():
                    r_t_neutral = torch.ones_like(r_t)
                    l0, _ = net.sub_logits(h, z, s, obs.cand.to(dev), obs.can_wait)
                    if util_after is not None:
                        ua_t0 = torch.tensor(util_after, dtype=torch.float32, device=dev)
                        pen0 = _res_penalty(cfg, ua_t0, r_t_neutral)
                        if pen0 is not None:
                            l0 = l0.clone()
                            l0[:len(obs.cand)] -= pen0
                n_goal += 1
                n_changed += int(int(torch.argmax(l0)) != a)
            snap = _snap(sim)
            if a == len(obs.cand):
                frag = _frag(sim)
                sim.advance()
            else:
                tid = int(obs.node_ids[int(obs.cand[a])])
                if res is not None:
                    res.on_dispatch(sim, tid)
                sim.dispatch(tid)
                frag = _frag(sim)
            sub_tr.append({"obs": obs, "g": g, "a": a, "logp": logp, "v": float(v),
                           "sim": simv, "frag": frag, "r": 0.0,
                           "r_t": None if r_t is None else r_t.detach(),
                           "util_after": util_after})
            prev = (len(sub_tr) - 1, snap,
                    None if r_t is None else r_t.detach().cpu().numpy().copy())
        # close the last Meta interval
        if meta_tr and rc.w_r > 0 and meta_tr[-1]["util_mean"] is None and meta_tr[-1]["dt_accum"] > 0:
            meta_tr[-1]["util_mean"] = (
                meta_tr[-1]["util_int_accum"] / (sim.cap * meta_tr[-1]["dt_accum"])).astype(np.float32)
        m = sim.metrics()
        m["latency_ms_mean"] = 1000 * float(np.mean(latencies)) if latencies else 0.0
        m["latency_ms_p95"] = 1000 * float(np.percentile(latencies, 95)) if latencies else 0.0
        m["n_sub_decisions"] = len(sub_tr)
        m["n_meta_decisions"] = len(meta_tr)
        m["return_sub"] = float(sum(t["r"] for t in sub_tr))
        if n_goal:
            m["goal_changed"] = n_changed / n_goal
        if cfg.res_goal and meta_tr:
            rts = [t["r_t_np"] for t in meta_tr if t.get("r_t_np") is not None]
            if rts:
                arr = np.stack(rts)
                m["r_t_range"] = float((arr.max(0) - arr.min(0)).max())
                m["r_t_mean"] = arr.mean(0).tolist()
        if res is not None:
            m.update(res.summary())
        return {"metrics": m, "sub": sub_tr, "meta": meta_tr}

    # -------------------------------------------------------------------- PPO
    @staticmethod
    def _gae(trs: List[Dict], gamma: float, lam: float):
        adv = np.zeros(len(trs))
        last = 0.0
        for k in reversed(range(len(trs))):
            nv = trs[k + 1]["v"] if k + 1 < len(trs) else 0.0
            delta = trs[k]["r"] + gamma * nv - trs[k]["v"]
            last = delta + gamma * lam * last
            adv[k] = last
        ret = adv + np.array([t["v"] for t in trs])
        return adv, ret

    def update(self, episodes: List[Dict], policy_coef: float = 1.0) -> Dict:
        cfg = self.cfg
        phase = "sub"
        if cfg.hierarchical and (self.iteration // cfg.alt_period) % 2 == 1:
            phase = "meta"
        self.iteration += 1
        key, gamma = ("sub", cfg.gamma_s) if phase == "sub" else ("meta", cfg.gamma_m)
        data = []
        for ep in episodes:
            trs = ep[key]
            if not trs:
                continue
            adv, ret = self._gae(trs, gamma, cfg.lam)
            for k, t in enumerate(trs):
                data.append((t, adv[k], ret[k]))
        if not data:
            return {"phase": phase}
        advs = np.array([d[1] for d in data])
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        opt = self.opt_sub if phase == "sub" else self.opt_meta
        params = [p for grp in opt.param_groups for p in grp["params"]]
        stats = {"pl": 0.0, "vl": 0.0, "ent": 0.0, "n": 0}
        idx = np.arange(len(data))
        dev = cfg.device
        for _ in range(cfg.epochs):
            np.random.shuffle(idx)
            for st in range(0, len(idx), cfg.minibatch):
                mb = idx[st:st + cfg.minibatch]
                lp_new, ents, vals, lp_old, A, R = [], [], [], [], [], []
                for j in mb:
                    t, _, ret = data[j]
                    o = t["obs"]
                    x, ei, ea, s = o.x.to(dev), o.ei.to(dev), o.ea.to(dev), o.s.to(dev)
                    if phase == "sub":
                        h, z = self.net.encode(x, ei, ea, t["g"])
                        logits, v = self.net.sub_logits(h, z, s, o.cand.to(dev), o.can_wait)
                        if cfg.res_goal and t.get("r_t") is not None and t.get("util_after") is not None:
                            ua = torch.as_tensor(t["util_after"], dtype=torch.float32, device=dev)
                            rt = t["r_t"].to(dev) if t["r_t"].device != torch.device(dev) else t["r_t"]
                            pen = _res_penalty(cfg, ua, rt)
                            if pen is not None:
                                logits = logits.clone()
                                logits[:len(o.cand)] -= pen
                        dist = torch.distributions.Categorical(logits=logits)
                        lp_new.append(dist.log_prob(torch.tensor(t["a"], device=dev)))
                        ents.append(dist.entropy())
                    else:
                        with torch.no_grad():
                            _, z = self.net.encode(x, ei, ea, t["g_in"])
                        dist, v = self.net.meta_dist(s, z)
                        lp_new.append(dist.log_prob(t["g"]).sum())
                        ents.append(dist.entropy().sum())
                    vals.append(v)
                    lp_old.append(t["logp"]); A.append(advs[j]); R.append(ret)
                lp_new = torch.stack(lp_new); ents = torch.stack(ents); vals = torch.stack(vals)
                lp_old = torch.tensor(lp_old, device=dev); A = torch.tensor(A, dtype=torch.float32, device=dev)
                R = torch.tensor(R, dtype=torch.float32, device=dev)
                ratio = torch.exp(lp_new - lp_old)
                pl = -torch.min(ratio * A, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * A).mean()
                vl = torch.nn.functional.mse_loss(vals, R)
                ent = ents.mean()
                loss = policy_coef * (pl - cfg.ent_coef * ent) + cfg.vf_coef * vl
                # v3: res_head loss (Meta phase only, on the detached g)
                if cfg.res_goal and self.rcfg.w_r > 0 and phase == "meta":
                    res_losses = []
                    for j in mb:
                        t = data[j][0]
                        if t.get("util_mean") is not None:
                            g_det = t["g_det"]
                            if g_det.device != torch.device(dev):
                                g_det = g_det.to(dev)
                            r_t_pred = self.net.resource_goal(g_det)
                            util_mean_t = torch.as_tensor(t["util_mean"], dtype=torch.float32, device=dev)
                            res_losses.append(self.rcfg.w_r * torch.mean((util_mean_t - r_t_pred) ** 2))
                    if res_losses:
                        loss = loss + torch.stack(res_losses).mean()
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
                opt.step()
                stats["pl"] += pl.item(); stats["vl"] += vl.item(); stats["ent"] += ent.item(); stats["n"] += 1
        n = max(stats.pop("n"), 1)
        return {"phase": phase, **{k: v / n for k, v in stats.items()}, "n_transitions": len(data)}

    def bc_update(self, episodes: List[Dict], epochs: int = 4) -> Dict:
        cfg, dev = self.cfg, self.cfg.device
        data = []
        for ep in episodes:
            trs = ep["sub"]
            G, ret = 0.0, []
            for t in reversed(trs):
                G = t["r"] + cfg.gamma_s * G
                ret.append(G)
            ret.reverse()
            data += list(zip(trs, ret))
        params = [p for grp in self.opt_sub.param_groups for p in grp["params"]]
        idx = np.arange(len(data))
        tot, acc, n = 0.0, 0.0, 0
        for _ in range(epochs):
            np.random.shuffle(idx)
            for st in range(0, len(idx), cfg.minibatch):
                losses, hits = [], []
                for j in idx[st:st + cfg.minibatch]:
                    t, R = data[j]
                    o = t["obs"]
                    h, z = self.net.encode(o.x.to(dev), o.ei.to(dev), o.ea.to(dev), t["g"])
                    logits, v = self.net.sub_logits(h, z, o.s.to(dev), o.cand.to(dev), o.can_wait)
                    if cfg.res_goal and t.get("r_t") is not None and t.get("util_after") is not None:
                        ua = torch.as_tensor(t["util_after"], dtype=torch.float32, device=dev)
                        rt = t["r_t"].to(dev) if t["r_t"].device != torch.device(dev) else t["r_t"]
                        pen = _res_penalty(cfg, ua, rt)
                        if pen is not None:
                            logits = logits.clone()
                            logits[:len(o.cand)] -= pen
                    ce = torch.nn.functional.cross_entropy(logits[None], torch.tensor([t["a"]], device=dev))
                    losses.append(ce + cfg.vf_coef * (v - R) ** 2)
                    hits.append(float(int(torch.argmax(logits)) == t["a"]))
                loss = torch.stack(losses).mean()
                self.opt_sub.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
                self.opt_sub.step()
                tot += loss.item(); acc += float(np.mean(hits)); n += 1
        out = {"bc_loss": tot / max(n, 1), "bc_acc": acc / max(n, 1), "n_transitions": len(data)}
        if cfg.meta_bc and cfg.hierarchical:
            out.update(self._meta_bc_update(episodes, epochs))
        return out

    def _meta_bc_update(self, episodes: List[Dict], epochs: int) -> Dict:
        cfg, dev, net = self.cfg, self.cfg.device, self.net
        data = []
        for ep in episodes:
            trs = [t for t in ep["meta"] if "target" in t]
            G, ret = 0.0, []
            for t in reversed(trs):
                G = t["r"] + cfg.gamma_m * G
                ret.append(G)
            ret.reverse()
            data += list(zip(trs, ret))
        if not data:
            return {}
        params = [p for o in (self.opt_sub, self.opt_meta) for grp in o.param_groups for p in grp["params"]]
        idx = np.arange(len(data))
        tot, err, n = 0.0, 0.0, 0
        for _ in range(epochs):
            np.random.shuffle(idx)
            for st in range(0, len(idx), cfg.minibatch):
                losses, errs = [], []
                for j in idx[st:st + cfg.minibatch]:
                    t, R = data[j]
                    o = t["obs"]
                    x = o.x.to(dev)
                    with torch.no_grad():
                        _, z = net.encode(x, o.ei.to(dev), o.ea.to(dev), t["g_in"])
                    dist, v = net.meta_dist(o.s.to(dev), z)
                    first, target = t["target"]
                    p = net.goal_annotation(x, dist.mean)[first]
                    mse = torch.mean((p - target.to(dev)) ** 2)
                    losses.append(mse + cfg.vf_coef * (v - R) ** 2)
                    errs.append(float(mse.detach()))
                loss = torch.stack(losses).mean()
                self.opt_sub.zero_grad(); self.opt_meta.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
                self.opt_sub.step(); self.opt_meta.step()
                tot += loss.item(); err += float(np.mean(errs)); n += 1
        return {"meta_bc_loss": tot / max(n, 1), "meta_bc_mse": err / max(n, 1), "n_meta_transitions": len(data)}

    # ------------------------------------------------------------------- misc
    def online_policy(self):
        cfg, net = self.cfg, self.net
        mem = {"g": None, "last_meta_t": -math.inf, "state": None, "res": None, "r_t": None}
        use_res = self.use_res

        def pol(state, D):
            if mem["state"] is not state:
                mem.update(g=None, last_meta_t=-math.inf, state=state, r_t=None,
                           res=Reservation(cfg.res_c_max, cfg.res_prior_n0) if use_res else None)
            if mem["res"] is not None:
                D = mem["res"].filter(state, list(D), net.g_res(mem["g"]))
                if not D:
                    return WAIT
            dev = cfg.device
            meta_dt = cfg.meta_interval_factor * float(state.inst.p_est.mean())
            obs = build_obs(state, D, cfg.dynamic_graph, mw=cfg.mw)
            x, ei, ea, s = obs.x.to(dev), obs.ei.to(dev), obs.ea.to(dev), obs.s.to(dev)
            with torch.no_grad():
                if cfg.hierarchical and (state.t - mem["last_meta_t"] >= meta_dt or mem["g"] is None):
                    _, z = net.encode(x, ei, ea, mem["g"])
                    dist, _ = net.meta_dist(s, z)
                    mem["g"] = dist.mean
                    mem["last_meta_t"] = state.t
                    mem["r_t"] = net.resource_goal(mem["g"]) if cfg.res_goal else None
                h, z = net.encode(x, ei, ea, mem["g"])
                logits, _ = net.sub_logits(h, z, s, obs.cand.to(dev), obs.can_wait)
                if cfg.res_goal and mem["r_t"] is not None:
                    ids = obs.node_ids[obs.cand.numpy()]
                    ua = (state.cap - state.free + state.dem[ids]) / state.cap
                    ua_t = torch.tensor(ua, dtype=torch.float32, device=dev)
                    pen = _res_penalty(cfg, ua_t, mem["r_t"])
                    if pen is not None:
                        logits = logits.clone()
                        logits[:len(obs.cand)] -= pen
                a = int(torch.argmax(logits))
            if a == len(obs.cand):
                return WAIT
            tid = int(obs.node_ids[int(obs.cand[a])])
            if mem["res"] is not None:
                mem["res"].on_dispatch(state, tid)
            return tid

        pol.mem = mem
        return pol

    def policy_fn(self):
        return lambda sim, seed: self.rollout(sim, seed, train=False)["metrics"]

    def save(self, path: str, extra: Optional[Dict] = None):
        torch.save({"net": self.net.state_dict(), "cfg": asdict(self.cfg),
                    "rcfg": asdict(self.rcfg), "extra": extra or {}}, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "GMHRLAgent":
        ck = torch.load(path, map_location=device, weights_only=False)
        cfg = AgentConfig(**{**ck["cfg"], "device": device})
        ag = cls(cfg, RewardConfig(**ck["rcfg"]))
        ag.net.load_state_dict(ck["net"])
        return ag