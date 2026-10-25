"""GM-HRL rollouts, rewards and hierarchical PPO (H-PPO).

Rewards (all terms normalised per instance, so the same weights apply at
every scale). Between two consecutive Sub-Controller decisions at times
t_k < t_{k+1}:

  progress  dC_k = (W(t_{k+1}) - W(t_k)) / W_ref - (t_{k+1} - t_k) / LB
                   W = summed estimated duration of completed tasks,
                   W_ref = summed estimated duration of the N submitted tasks,
                   LB = submission-time makespan lower bound of the instance.
                   Summed over an episode: W_total/W_ref - makespan/LB.
  util      U_k  = time integral over [t_k, t_{k+1}] of mean_r used_r/R_r, / LB
  SLA       P_k  = number of tasks whose response time exceeded Delta t_resp
                   during [t_k, t_{k+1}], / N
  idleness  I_k  = time integral of mean_r free_r/R_r while tasks were ready, / LB

  b_k   = w1*dC_k + w2*U_k - kappa*w3*P_k - w4*I_k
  r^M   = sum of b_k over the Meta-Controller interval
  r^S_k = eta1*b_k + (eta2*Sim_k - eta3*Frag_k) / N

  Sim_k  = p_a / max_{v in D_k} p_v for a dispatched task a (0 for WAIT),
           where p_v is the goal attribute written by the Meta-Controller.
  Frag_k = mean_r free_r/R_r right after the decision if tasks are ready but
           none fits into the free capacity, else 0.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional

import numpy as np
import torch

from .features import build_obs
from .model import GMHRLNet
from .sim import MD2POSSim, WAIT


@dataclass
class RewardConfig:
    w1: float = 1.0     # makespan progress
    w2: float = 0.1     # utilisation
    w3: float = 1.0     # SLA violations
    w4: float = 0.1     # idleness while work is waiting
    kappa: float = 1.0  # SLA/idleness trade-off multiplier (Pareto experiments)
    eta1: float = 1.0
    eta2: float = 0.1   # goal consistency (Sim)
    eta3: float = 0.1   # fragmentation


@dataclass
class AgentConfig:
    d: int = 64
    layers: int = 2
    d_g: int = 8
    hidden: int = 128
    use_gnn: bool = True
    hierarchical: bool = True
    dynamic_graph: bool = True
    meta_interval_factor: float = 2.0   # Delta t_H = factor * mean p_est
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
    alt_period: int = 5                 # iterations per H-PPO phase
    device: str = "cpu"


def _snap(sim: MD2POSSim):
    return (sim.t, sim.work_done, sim.util_int.copy(), sim.sla_viol, sim.idle_int)


def _base_reward(sim: MD2POSSim, s0, rc: RewardConfig) -> float:
    t0, w0, u0, v0, i0 = s0
    prog = (sim.work_done - w0) / sim.w_ref - (sim.t - t0) / sim.lb_ref
    util = float(np.mean((sim.util_int - u0) / sim.cap)) / sim.lb_ref
    sla = (sim.sla_viol - v0) / sim.N0
    idle = (sim.idle_int - i0) / sim.lb_ref
    return rc.w1 * prog + rc.w2 * util - rc.kappa * rc.w3 * sla - rc.w4 * idle


def _frag(sim: MD2POSSim) -> float:
    if not sim.ready:
        return 0.0
    if any(sim.fits(i) for i in sim.ready):
        return 0.0
    return float(np.mean(sim.free / sim.cap))


class GMHRLAgent:
    def __init__(self, cfg: AgentConfig, rcfg: Optional[RewardConfig] = None):
        self.cfg, self.rcfg = cfg, rcfg or RewardConfig()
        self.net = GMHRLNet(cfg.d, cfg.layers, cfg.d_g, cfg.hidden,
                            cfg.use_gnn, cfg.hierarchical).to(cfg.device)
        n = self.net
        sub_params = (list(n.enc.parameters()) + list(n.Wg.parameters()) + list(n.Wx.parameters())
                      + list(n.ctx.parameters()) + list(n.score.parameters())
                      + list(n.wait.parameters()) + list(n.sub_v.parameters()))
        meta_params = list(n.meta_mu.parameters()) + [n.meta_logstd] + list(n.meta_v.parameters())
        self.opt_sub = torch.optim.Adam(sub_params, lr=cfg.lr, eps=1e-5)
        self.opt_meta = torch.optim.Adam(meta_params, lr=cfg.lr, eps=1e-5)
        self.iteration = 0

    # ---------------------------------------------------------------- rollout
    def rollout(self, sim: MD2POSSim, seed: int, train: bool = True, expert=None) -> Dict:
        """One episode. With ``expert`` (a baseline rule), the expert's choice is
        executed and recorded as the action (used for behaviour cloning)."""
        cfg, rc, net = self.cfg, self.rcfg, self.net
        dev = cfg.device
        sim.reset(seed)
        meta_dt = cfg.meta_interval_factor * float(sim.inst.p_est.mean())
        g = None
        last_meta_t = -math.inf
        sub_tr: List[Dict] = []
        meta_tr: List[Dict] = []
        prev = None          # (index in sub_tr, snapshot)
        latencies = []
        while True:
            D = sim.dispatchable()
            while not D and not sim.done():
                if not sim.advance():
                    break
                D = sim.dispatchable()
            if prev is not None:
                b = _base_reward(sim, prev[1], rc)
                tr = sub_tr[prev[0]]
                tr["r"] = rc.eta1 * b + (rc.eta2 * tr["sim"] - rc.eta3 * tr["frag"]) / sim.N0
                if meta_tr:
                    meta_tr[-1]["r"] += b
            if sim.done() or not D:
                break
            t_start = time.perf_counter()
            obs = build_obs(sim, D, cfg.dynamic_graph)
            x, ei, ea, s = obs.x.to(dev), obs.ei.to(dev), obs.ea.to(dev), obs.s.to(dev)
            with torch.no_grad():
                if cfg.hierarchical and (sim.t - last_meta_t >= meta_dt or g is None):
                    g_in = g
                    _, z = net.encode(x, ei, ea, g_in)
                    dist, v = net.meta_dist(s, z)
                    g = dist.sample() if train else dist.mean
                    meta_tr.append({"obs": obs, "g_in": g_in, "g": g, "logp": float(dist.log_prob(g).sum()),
                                    "v": float(v), "r": 0.0})
                    last_meta_t = sim.t
                h, z = net.encode(x, ei, ea, g)
                logits, v = net.sub_logits(h, z, s, obs.cand.to(dev), obs.can_wait)
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
            snap = _snap(sim)
            if a == len(obs.cand):
                frag = _frag(sim)
                sim.advance()
            else:
                sim.dispatch(int(obs.node_ids[int(obs.cand[a])]))
                frag = _frag(sim)
            sub_tr.append({"obs": obs, "g": g, "a": a, "logp": logp, "v": float(v),
                           "sim": simv, "frag": frag, "r": 0.0})
            prev = (len(sub_tr) - 1, snap)
        m = sim.metrics()
        m["latency_ms_mean"] = 1000 * float(np.mean(latencies)) if latencies else 0.0
        m["latency_ms_p95"] = 1000 * float(np.percentile(latencies, 95)) if latencies else 0.0
        m["n_sub_decisions"] = len(sub_tr)
        m["n_meta_decisions"] = len(meta_tr)
        m["return_sub"] = float(sum(t["r"] for t in sub_tr))
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
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
                opt.step()
                stats["pl"] += pl.item(); stats["vl"] += vl.item(); stats["ent"] += ent.item(); stats["n"] += 1
        n = max(stats.pop("n"), 1)
        return {"phase": phase, **{k: v / n for k, v in stats.items()}, "n_transitions": len(data)}

    def bc_update(self, episodes: List[Dict], epochs: int = 4) -> Dict:
        """Behaviour cloning of the Sub-Controller (cross-entropy to the expert's
        choices) plus value regression to discounted returns."""
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
                    ce = torch.nn.functional.cross_entropy(logits[None], torch.tensor([t["a"]], device=dev))
                    losses.append(ce + cfg.vf_coef * (v - R) ** 2)
                    hits.append(float(int(torch.argmax(logits)) == t["a"]))
                loss = torch.stack(losses).mean()
                self.opt_sub.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
                self.opt_sub.step()
                tot += loss.item(); acc += float(np.mean(hits)); n += 1
        return {"bc_loss": tot / max(n, 1), "bc_acc": acc / max(n, 1), "n_transitions": len(data)}

    # ------------------------------------------------------------------- misc
    def online_policy(self):
        """Step-wise deterministic policy ``pol(state, D) -> task id or WAIT``
        for use outside the simulator (e.g. the real-pipeline dispatcher).
        ``state`` must expose the same attributes as MD2POSSim. Decision
        logic is identical to ``rollout(..., train=False)``."""
        cfg, net = self.cfg, self.net
        mem = {"g": None, "last_meta_t": -math.inf, "state": None}

        def pol(state, D):
            if mem["state"] is not state:
                mem.update(g=None, last_meta_t=-math.inf, state=state)
            dev = cfg.device
            meta_dt = cfg.meta_interval_factor * float(state.inst.p_est.mean())
            obs = build_obs(state, D, cfg.dynamic_graph)
            x, ei, ea, s = obs.x.to(dev), obs.ei.to(dev), obs.ea.to(dev), obs.s.to(dev)
            with torch.no_grad():
                if cfg.hierarchical and (state.t - mem["last_meta_t"] >= meta_dt or mem["g"] is None):
                    _, z = net.encode(x, ei, ea, mem["g"])
                    dist, _ = net.meta_dist(s, z)
                    mem["g"] = dist.mean
                    mem["last_meta_t"] = state.t
                h, z = net.encode(x, ei, ea, mem["g"])
                logits, _ = net.sub_logits(h, z, s, obs.cand.to(dev), obs.can_wait)
                a = int(torch.argmax(logits))
            if a == len(obs.cand):
                return WAIT
            return int(obs.node_ids[int(obs.cand[a])])

        return pol

    def policy_fn(self):
        """Deterministic policy usable with evaluation helpers."""
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
