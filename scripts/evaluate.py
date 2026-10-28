#!/usr/bin/env python3
"""Evaluate methods on the benchmark test instances (k = 0..19).

Every method is run on the same (instance, realisation seed) pairs, so results
are paired. Output: runs.csv (one row per method x instance x seed),
summary.json (mean, std, n per method and paired two-sided Wilcoxon
signed-rank tests against --ref), and optionally offline clairvoyant bounds.

Methods:
  FIFO SJF HEFT Random            built-in baselines
  ORHG                            rolling-horizon CP-SAT re-planning (1 s per solve)
  GA=path/to/ga.json              GA-evolved dispatching rule
  NAME=path/to/checkpoint.pt      trained GM-HRL / ablation / PPO checkpoint

Example:
  python scripts/evaluate.py --domains genomic --N 50 \
      --methods FIFO SJF HEFT Random GM-HRL=runs/gmhrl_genomic_N50_s0/best.pt \
      --seeds 1000 1001 1002 --bound --bound_time 300 --out results/genomic_N50
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from gmhrl.agent import GMHRLAgent  # noqa: E402
from gmhrl.baselines import fifo, sjf, heft, make_random, make_ga_policy  # noqa: E402
from gmhrl.orhg import make_orhg_policy  # noqa: E402
from gmhrl.instance import load_instance, instance_paths  # noqa: E402
from gmhrl.sim import MD2POSSim, SimConfig, run_policy  # noqa: E402
from gmhrl import bounds as B  # noqa: E402

METRICS = ("makespan", "makespan_over_lb", "sla_violation_rate", "util_cpu", "util_ram", "util_gpu",
           "util_disk", "mean_wait", "mean_response", "n_rrd_events", "n_dynamic_tasks",
           "latency_ms_mean", "latency_ms_p95")


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--methods", nargs="+", required=True)
    p.add_argument("--domains", nargs="+", default=["genomic"])
    p.add_argument("--N", nargs="+", type=int, default=[50])
    p.add_argument("--instances", nargs="+", type=int, default=list(range(20)))
    p.add_argument("--seeds", nargs="+", type=int, default=[1000, 1001, 1002])
    p.add_argument("--bench", default=str(ROOT / "data" / "md2pos_benchmark"))
    p.add_argument("--ref", default=None, help="method for paired tests (default: first model)")
    p.add_argument("--rho", type=float, default=0.8)
    p.add_argument("--sla_factor", type=float, default=4.0)
    p.add_argument("--rrd_mult", type=float, default=1.0)
    p.add_argument("--cap_scale", type=float, default=1.0)
    p.add_argument("--bound", action="store_true", help="compute offline clairvoyant bounds (CP-SAT)")
    p.add_argument("--bound_time", type=float, default=300)
    p.add_argument("--bound_workers", type=int, default=8)
    p.add_argument("--bounds_csv", default=None, help="reuse bounds.csv from an earlier evaluation")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", required=True)
    return p.parse_args()


def build_methods(specs, device):
    out = {}
    for s in specs:
        if s == "FIFO":
            out[s] = ("rule", fifo)
        elif s == "SJF":
            out[s] = ("rule", sjf)
        elif s == "HEFT":
            out[s] = ("rule", heft)
        elif s == "Random":
            out[s] = ("random", None)
        elif s == "ORHG":
            out[s] = ("orhg", None)
        elif s.startswith("GA="):
            w = json.load(open(s[3:]))["weights"]
            out["GA"] = ("rule", make_ga_policy(np.array(w)))
        elif "=" in s:
            name, path = s.split("=", 1)
            out[name] = ("model", GMHRLAgent.load(path, device))
        else:
            raise ValueError(f"unknown method {s}")
    return out


def run_one(kind, obj, sim, seed):
    sim.reset(seed)
    if kind == "model":
        return obj.rollout(sim, seed, train=False)["metrics"]
    pol = obj if kind == "rule" else (make_random(seed) if kind == "random" else make_orhg_policy(1.0, 1))
    lat = []

    def timed(s, D):
        t0 = time.perf_counter()
        a = pol(s, D)
        lat.append(time.perf_counter() - t0)
        return a
    m = run_policy(sim, timed)
    m["latency_ms_mean"] = 1000 * float(np.mean(lat)) if lat else 0.0
    m["latency_ms_p95"] = 1000 * float(np.percentile(lat, 95)) if lat else 0.0
    return m


def main():
    a = parse()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    scfg = SimConfig(rho=a.rho, sla_factor=a.sla_factor, rrd_mult=a.rrd_mult, cap_scale=a.cap_scale)
    methods = build_methods(a.methods, a.device)
    rows = []
    bound_rows = []
    reuse = {}
    if a.bounds_csv:
        for r in csv.DictReader(open(a.bounds_csv)):
            reuse[(r["domain"], int(r["N"]), int(r["k"]), int(r["seed"]))] = r
    for dom in a.domains:
        for N in a.N:
            paths = {int(p.stem[-4:]): p for p in instance_paths(a.bench, dom, N)}  # k -> file (instance_seedKKKK)
            for k in a.instances:
                inst = load_instance(paths[k])
                sim = MD2POSSim(inst, scfg, seed=0)
                for seed in a.seeds:
                    if (dom, N, k, seed) in reuse:
                        r0 = reuse[(dom, N, k, seed)]
                        bound_rows.append({**r0, "N": N, "k": k, "seed": seed, "best_LB": float(r0["best_LB"])})
                    elif a.bound:
                        sim.reset(seed)
                        real = sim.export_realised()
                        tmp = out / "realised" / f"{dom}_N{N}_k{k}_s{seed}.json"
                        tmp.parent.mkdir(exist_ok=True)
                        json.dump(real, open(tmp, "w"))
                        ana = B.analytical_bounds(real)
                        cp = B.solve_cpsat(real, a.bound_time, a.bound_workers, 100.0, hint_lb=ana["analytical_LB"])
                        lb = max(ana["analytical_LB"], cp.get("best_bound") or 0)
                        bound_rows.append({"domain": dom, "N": N, "k": k, "seed": seed,
                                           "analytical_LB": ana["analytical_LB"], "cpsat_status": cp["status"],
                                           "cpsat_incumbent": cp.get("incumbent"), "cpsat_bound": cp.get("best_bound"),
                                           "best_LB": lb})
                    for name, (kind, obj) in methods.items():
                        m = run_one(kind, obj, sim, seed)
                        rows.append({"method": name, "domain": dom, "N": N, "k": k, "seed": seed,
                                     **{key: m.get(key) for key in METRICS}})
                print(f"{dom} N={N} instance {k} done", flush=True)

    if bound_rows:
        with open(out / "bounds.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(bound_rows[0].keys()))
            w.writeheader()
            w.writerows(bound_rows)
        lbmap = {(r["domain"], r["N"], r["k"], r["seed"]): r["best_LB"] for r in bound_rows}
        for r in rows:
            lb = lbmap[(r["domain"], r["N"], r["k"], r["seed"])]
            r["gap_to_clairvoyant_LB"] = (r["makespan"] - lb) / lb
    with open(out / "runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    if not bound_rows:
        with open(out / "runs.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    names = list(methods)
    ref = a.ref or next((n for n in names if methods[n][0] == "model"), names[0])
    summ = {"config": {**vars(a), "sim": scfg.to_dict()}, "methods": {}}
    keys = list(METRICS) + (["gap_to_clairvoyant_LB"] if bound_rows else [])
    for n in names:
        R = [r for r in rows if r["method"] == n]
        summ["methods"][n] = {k: {"mean": float(np.mean([r[k] for r in R])),
                                  "std": float(np.std([r[k] for r in R], ddof=1)) if len(R) > 1 else 0.0}
                              for k in keys}
        summ["methods"][n]["n_runs"] = len(R)
    tests = {}
    refR = [r for r in rows if r["method"] == ref]
    for n in names:
        if n == ref:
            continue
        R = [r for r in rows if r["method"] == n]
        tests[n] = {}
        for k in ("makespan", "sla_violation_rate"):
            x = np.array([r[k] for r in refR]); y = np.array([r[k] for r in R])
            if np.allclose(x, y):
                tests[n][k] = {"p_value": 1.0, "ref_better_share": 0.0}
                continue
            stat = wilcoxon(x, y, alternative="two-sided", zero_method="wilcox")
            tests[n][k] = {"p_value": float(stat.pvalue),
                           "ref_better_share": float(np.mean(x < y)),
                           "mean_diff_ref_minus_other": float(np.mean(x - y))}
    summ["paired_tests_vs"] = ref
    summ["paired_tests"] = tests
    json.dump(summ, open(out / "summary.json", "w"), indent=2)

    print(f"\n{'method':16s} {'makespan':>10s} {'mk/LB':>7s} {'SLA':>7s} {'CPU':>6s}" +
          (f" {'gap':>7s}" if bound_rows else "") + f" {'lat ms':>7s}")
    for n in names:
        s = summ["methods"][n]
        line = (f"{n:16s} {s['makespan']['mean']:10.1f} {s['makespan_over_lb']['mean']:7.3f} "
                f"{s['sla_violation_rate']['mean']:7.3f} {s['util_cpu']['mean']:6.3f}")
        if bound_rows:
            line += f" {100*s['gap_to_clairvoyant_LB']['mean']:6.1f}%"
        line += f" {s['latency_ms_mean']['mean']:7.2f}"
        print(line)
    print(f"\npaired Wilcoxon tests vs {ref}:")
    for n, t in tests.items():
        print(f"  {n:14s} makespan p={t['makespan']['p_value']:.2g} (ref better in "
              f"{100*t['makespan'].get('ref_better_share', 0):.0f}% of runs); "
              f"SLA p={t['sla_violation_rate']['p_value']:.2g}")


if __name__ == "__main__":
    main()
