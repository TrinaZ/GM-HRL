#!/usr/bin/env python3
"""Pre-registered replication criteria (REPLICATION_SPEC.md, section 4) and the
descriptive statistics of Part B (section 5).

Part A reads <res>/rep_<domain>_N<N>/runs.csv. For every (instance, realisation)
the learned methods are averaged over their three training seeds; relative
differences are means over the 60 paired runs of (method / HEFT - 1); p-values
are two-sided Wilcoxon signed-rank tests, Holm-corrected over the six
(domain, N) comparisons of each criterion.
  R1  GM-HRL makespan < HEFT, Holm p < 0.05     genomic, video x N = 200, 500, 1000
  R2  GM-HRL mean response < HEFT, Holm p < 0.05 same six cells
Everything else is reported, not decided.
Part B (--dyngraph) reads <res>/dyn_*/runs.csv: GM-HRL vs w/o dynamic graph.
"""
import argparse, csv, json, re
from collections import defaultdict
import numpy as np
from scipy.stats import wilcoxon

PAPER = {("genomic", 200): -3.17, ("genomic", 500): -2.52, ("genomic", 1000): -2.18,
         ("video", 200): -2.74, ("video", 500): -2.27, ("video", 1000): -1.82}   # Table 9, % vs HEFT


def fam(n): return re.sub(r"_s\d+$", "", n)


def load(path):
    d = defaultdict(lambda: defaultdict(list))
    for r in csv.DictReader(open(path)):
        d[fam(r["method"])][(r["k"], r["seed"])].append(r)
    return d


def series(d, f, key, metric):
    return np.array([np.mean([float(r[metric]) for r in d[f][k]]) for k in key])


def p_w(x, y):
    return 1.0 if np.allclose(x, y) else float(wilcoxon(x, y, alternative="two-sided").pvalue)


def holm(ps):
    o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for i, j in enumerate(o):
        run = max(run, min(1.0, (m - i) * ps[j])); adj[j] = run
    return adj.tolist()


def part_a(res):
    cells, rep = [], {"criteria": {"R1": "GM-HRL makespan < HEFT, Holm-adjusted p < 0.05",
                                   "R2": "GM-HRL mean response time < HEFT, Holm-adjusted p < 0.05"}, "cells": {}}
    for dom in ("genomic", "video", "ai_ft", "financial"):
        for n in (200, 500, 1000):
            d = load(f"{res}/rep_{dom}_N{n}/runs.csv")
            key = sorted(d["HEFT"]); assert len(key) == 60, (dom, n, len(key))
            ref = {m: series(d, "HEFT", key, m) for m in ("makespan", "mean_response", "sla_violation_rate")}
            c = {}
            for f in sorted(d):
                x = {m: series(d, f, key, m) for m in ref}
                c[f] = {"mk_rel_pct": 100 * float(np.mean(x["makespan"] / ref["makespan"] - 1)),
                        "resp_rel_pct": 100 * float(np.mean(x["mean_response"] / ref["mean_response"] - 1)),
                        "sla_pct": 100 * float(x["sla_violation_rate"].mean()),
                        "p_mk": p_w(x["makespan"], ref["makespan"]), "p_resp": p_w(x["mean_response"], ref["mean_response"])}
            g, s = series(d, "GM-HRL", key, "makespan"), series(d, "GNN-PPO", key, "makespan")
            c["GM-HRL_vs_GNN-PPO"] = {"mk_rel_pct": 100 * float(np.mean(g / s - 1)), "p": p_w(g, s)}
            rep["cells"][f"{dom}_N{n}"] = c
            if dom in ("genomic", "video"):
                cells.append((dom, n))
    for crit, mk, pk in (("R1", "mk_rel_pct", "p_mk"), ("R2", "resp_rel_pct", "p_resp")):
        ps = [rep["cells"][f"{d}_N{n}"]["GM-HRL"][pk] for d, n in cells]
        adj = holm(ps)
        ok = []
        for (d, n), pa in zip(cells, adj):
            e = rep["cells"][f"{d}_N{n}"]["GM-HRL"]; e[f"{pk}_holm"] = pa
            ok.append(e[mk] < 0 and pa < 0.05)
        rep[crit] = {"cells_passed": int(sum(ok)), "of": len(ok), "replicated": bool(all(ok)),
                     "per_cell": {f"{d}_N{n}": bool(o) for (d, n), o in zip(cells, ok)}}
    print(f"{'cell':16s} {'GM-HRL mk%':>10s} {'paper':>6s} {'p_holm':>8s} {'resp%':>7s} {'p_holm':>8s} {'vs GNN-PPO':>10s}")
    for d, n in cells:
        e = rep["cells"][f"{d}_N{n}"]
        g = e["GM-HRL"]
        print(f"{d+'_N'+str(n):16s} {g['mk_rel_pct']:+10.2f} {PAPER[(d, n)]:+6.2f} {g['p_mk_holm']:8.2g} "
              f"{g['resp_rel_pct']:+7.1f} {g['p_resp_holm']:8.2g} {e['GM-HRL_vs_GNN-PPO']['mk_rel_pct']:+10.2f}")
    for dom in ("ai_ft", "financial"):
        for n in (200, 500, 1000):
            g = rep["cells"][f"{dom}_N{n}"]["GM-HRL"]
            print(f"{dom+'_N'+str(n):16s} {g['mk_rel_pct']:+10.2f} {'':6s} {g['p_mk']:8.2g} (descriptive)")
    for crit in ("R1", "R2"):
        r = rep[crit]
        print(f"{crit}: {r['cells_passed']}/{r['of']} cells -> {'REPLICATED' if r['replicated'] else 'not fully replicated'}")
    return rep


def part_b(res):
    conds = ["default", "rrd0", "rrd2.6667", "rho0.6", "rho0.95", "cap0.7"]
    rows, ps = [], []
    for c in conds:
        d = load(f"{res}/dyn_{c}/runs.csv"); key = sorted(d["HEFT"])
        g, nd = series(d, "GM-HRL", key, "makespan"), series(d, "no_dynamic_graph", key, "makespan")
        h = series(d, "HEFT", key, "makespan")
        gs, ns = series(d, "GM-HRL", key, "sla_violation_rate"), series(d, "no_dynamic_graph", key, "sla_violation_rate")
        p = p_w(g, nd); ps.append(p)
        rows.append({"condition": c, "GM-HRL_vs_HEFT_pct": 100 * float(np.mean(g / h - 1)),
                     "noDG_vs_HEFT_pct": 100 * float(np.mean(nd / h - 1)),
                     "noDG_vs_GM-HRL_pct": 100 * float(np.mean(nd / g - 1)), "p": p,
                     "sla_GM-HRL_pct": 100 * float(gs.mean()), "sla_noDG_pct": 100 * float(ns.mean()),
                     "p_sla": p_w(gs, ns)})
    for r, pa in zip(rows, holm(ps)):
        r["p_holm"] = pa
        print(f"{r['condition']:10s} GM-HRL {r['GM-HRL_vs_HEFT_pct']:+6.2f}%  w/o dyn. graph {r['noDG_vs_HEFT_pct']:+6.2f}%  "
              f"(diff {r['noDG_vs_GM-HRL_pct']:+5.2f}%, p={r['p']:.2g}, Holm {pa:.2g})  "
              f"SLA {r['sla_GM-HRL_pct']:.1f} vs {r['sla_noDG_pct']:.1f} (p={r['p_sla']:.2g})")
    return {"note": "descriptive (REPLICATION_SPEC.md, section 5)", "conditions": rows}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="results_rep")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dyngraph", action="store_true")
    a = ap.parse_args()
    json.dump(part_b(a.res) if a.dyngraph else part_a(a.res), open(a.out, "w"), indent=2)
