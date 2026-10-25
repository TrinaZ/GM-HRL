#!/usr/bin/env python3
"""Train GM-HRL (or an ablation / the monolithic PPO baseline).

Training instances are freshly generated with the benchmark generator
(instance indices k = 100 .. 100+n_train-1); validation uses k = 50..54;
the 20 benchmark instances (k = 0..19) are reserved for testing.

Example:
  python scripts/train.py --variant gmhrl --domains genomic --N 50 --iters 300 --out runs/gmhrl_genomic_N50_s0 --seed 0
"""
import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from gmhrl.agent import GMHRLAgent, AgentConfig, RewardConfig  # noqa: E402
from gmhrl.generate import write_split  # noqa: E402
from gmhrl.instance import load_instance  # noqa: E402
from gmhrl.sim import MD2POSSim, SimConfig  # noqa: E402

VARIANTS = {
    "gmhrl": {},
    "no_hierarchy": {"hierarchical": False},
    "no_gnn": {"use_gnn": False},
    "no_dynamic_graph": {"dynamic_graph": False},
    "ppo": {"hierarchical": False, "use_gnn": False},
    # single-level GNN + PPO dispatcher in the style of Zhang et al. (2020);
    # same network as the no_hierarchy ablation, trained on every (domain, N)
    "gnn_ppo": {"hierarchical": False},
}


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="gmhrl", choices=list(VARIANTS))
    p.add_argument("--domains", nargs="+", default=["genomic"])
    p.add_argument("--N", nargs="+", type=int, default=[50])
    p.add_argument("--n_train", type=int, default=100, help="training instances per (domain, N)")
    p.add_argument("--n_val", type=int, default=5)
    p.add_argument("--iters", type=int, default=300)
    p.add_argument("--episodes_per_iter", type=int, default=4)
    p.add_argument("--eval_every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--data", default=str(ROOT / "data" / "generated"))
    p.add_argument("--out", required=True)
    # environment
    p.add_argument("--rho", type=float, default=0.8)
    p.add_argument("--sla_factor", type=float, default=4.0)
    p.add_argument("--rrd_mult", type=float, default=1.0)
    p.add_argument("--cap_scale", type=float, default=1.0)
    # reward
    p.add_argument("--kappa", type=float, default=1.0)
    p.add_argument("--w", nargs=4, type=float, default=[1.0, 0.1, 1.0, 0.1])
    p.add_argument("--eta", nargs=3, type=float, default=[1.0, 0.1, 0.1])
    # agent
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--d", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--d_g", type=int, default=8)
    p.add_argument("--alt_period", type=int, default=5)
    p.add_argument("--meta_interval_factor", type=float, default=2.0)
    p.add_argument("--init", default=None, help="checkpoint to start from (e.g. curriculum)")
    p.add_argument("--bc_iters", type=int, default=0, help="behaviour-cloning iterations (expert: HEFT)")
    p.add_argument("--critic_warmup", type=int, default=0, help="PPO iterations with value loss only")
    p.add_argument("--ent_coef", type=float, default=0.01)
    return p.parse_args()


def make_sims(paths, scfg):
    return [MD2POSSim(load_instance(p), scfg, seed=0) for p in paths]


def evaluate(agent, sims, seeds):
    rows = []
    for sim in sims:
        for s in seeds:
            rows.append(agent.rollout(sim, s, train=False)["metrics"])
    return {k: float(np.mean([r[k] for r in rows])) for k in
            ("makespan_over_lb", "sla_violation_rate", "util_cpu", "latency_ms_mean")}


def main():
    a = parse()
    if a.threads:
        torch.set_num_threads(a.threads)
    rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    scfg = SimConfig(rho=a.rho, sla_factor=a.sla_factor, rrd_mult=a.rrd_mult, cap_scale=a.cap_scale)
    acfg = AgentConfig(d=a.d, layers=a.layers, d_g=a.d_g, lr=a.lr, alt_period=a.alt_period, ent_coef=a.ent_coef,
                       meta_interval_factor=a.meta_interval_factor, device=a.device, **VARIANTS[a.variant])
    rcfg = RewardConfig(w1=a.w[0], w2=a.w[1], w3=a.w[2], w4=a.w[3], kappa=a.kappa,
                        eta1=a.eta[0], eta2=a.eta[1], eta3=a.eta[2])
    agent = GMHRLAgent(acfg, rcfg)
    if a.init:
        agent.net.load_state_dict(GMHRLAgent.load(a.init, a.device).net.state_dict())

    train_paths, val_paths = [], []
    for dom in a.domains:
        for N in a.N:
            train_paths += write_split(a.data, dom, N, range(100, 100 + a.n_train))
            val_paths += write_split(a.data, dom, N, range(50, 50 + a.n_val))
    train_sims = make_sims(train_paths, scfg)
    val_sims = make_sims(val_paths, scfg)
    val_seeds = [9001, 9002]

    with open(out / "config.json", "w") as f:
        json.dump({"args": vars(a), "sim": scfg.to_dict(), "agent": asdict(acfg), "reward": asdict(rcfg),
                   "n_train_instances": len(train_paths), "n_val_instances": len(val_paths)}, f, indent=2)

    log = open(out / "train_log.jsonl", "w")
    best = float("inf")
    t_start = time.time()
    if a.bc_iters:
        from gmhrl.baselines import heft
        for it in range(a.bc_iters):
            eps = []
            for _ in range(a.episodes_per_iter):
                sim = train_sims[int(rng.integers(len(train_sims)))]
                eps.append(agent.rollout(sim, int(rng.integers(1 << 30)), train=True, expert=heft))
            st = agent.bc_update(eps)
            log.write(json.dumps({"bc_iter": it, **st}) + "\n")
            if (it + 1) % a.eval_every == 0 or it == a.bc_iters - 1:
                ev = evaluate(agent, val_sims, val_seeds)
                print(f"[{a.variant}] BC it {it+1}/{a.bc_iters} acc={st['bc_acc']:.3f} "
                      f"val mk/LB={ev['makespan_over_lb']:.3f} val SLA={ev['sla_violation_rate']:.3f}", flush=True)
                if ev["makespan_over_lb"] < best:
                    best = ev["makespan_over_lb"]
                    agent.save(str(out / "best.pt"), {"bc_iter": it, "val": ev})
        agent.save(str(out / "bc.pt"), {"bc_iters": a.bc_iters})
    for it in range(a.iters):
        eps = []
        for _ in range(a.episodes_per_iter):
            sim = train_sims[int(rng.integers(len(train_sims)))]
            eps.append(agent.rollout(sim, int(rng.integers(1 << 30)), train=True))
        st = agent.update(eps, policy_coef=0.0 if it < a.critic_warmup else 1.0)
        rec = {"iter": it, "time_s": round(time.time() - t_start, 1), **st,
               "train_makespan_over_lb": float(np.mean([e["metrics"]["makespan_over_lb"] for e in eps])),
               "train_sla": float(np.mean([e["metrics"]["sla_violation_rate"] for e in eps])),
               "train_return": float(np.mean([e["metrics"]["return_sub"] for e in eps]))}
        if (it + 1) % a.eval_every == 0 or it == a.iters - 1:
            ev = evaluate(agent, val_sims, val_seeds)
            rec.update({f"val_{k}": v for k, v in ev.items()})
            score = ev["makespan_over_lb"]   # model selection on the primary objective
            if score < best:
                best = score
                agent.save(str(out / "best.pt"), {"iter": it, "val": ev})
            print(f"[{a.variant}] it {it+1}/{a.iters} {rec['time_s']}s phase={st.get('phase')} "
                  f"train mk/LB={rec['train_makespan_over_lb']:.3f} val mk/LB={ev['makespan_over_lb']:.3f} "
                  f"val SLA={ev['sla_violation_rate']:.3f}", flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()
    agent.save(str(out / "last.pt"), {"iter": a.iters - 1})
    log.close()


if __name__ == "__main__":
    main()
