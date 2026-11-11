#!/usr/bin/env python3
"""Descriptive statistics for the MD2POS-MW test and secondary evaluations
(HIERARCHY_SPEC.md section 4 and amendment v1.1): no decision is made here.
For each <dir>/runs.csv: learned methods averaged over training seeds per
(batch, realisation); paired two-sided Wilcoxon of every method's WV against
GM-HRL v2.  Usage: python3 scripts/stats_hier.py results_hier/test_* results_hier/test_rho*
"""
import csv, re, sys
from collections import defaultdict
import numpy as np
from scipy.stats import wilcoxon

def fam(n): return re.sub(r"_s\d+$", "", n)

def val(rows, m):
    v = [float(r[m]) for r in rows if r.get(m) not in ("", None)]
    v = [x for x in v if not np.isnan(x)]
    return np.mean(v) if v else np.nan

for d in sys.argv[1:]:
    data = defaultdict(lambda: defaultdict(list))
    for r in csv.DictReader(open(f"{d}/runs.csv")):
        data[fam(r["method"])][(r["k"], r["seed"])].append(r)
    key = sorted(data["GM-HRL-v2"])
    ref = np.array([val(data["GM-HRL-v2"][k], "wv") for k in key])
    print(f"== {d}  ({len(key)} paired runs)")
    print(f"  {'method':20s} {'WV':>7s} {'STAT':>7s} {'rout':>7s} {'mk/LB':>6s} {'chg':>6s}  p(WV vs v2)")
    for f in data:
        assert sorted(data[f]) == key, f
        g = {m: np.array([val(data[f][k], m) for k in key]) for m in
             ("wv", "stat_viol", "routine_viol", "makespan_over_lb", "goal_changed")}
        x = g["wv"]
        p = "" if f == "GM-HRL-v2" else (1.0 if np.allclose(x, ref) else
             wilcoxon(x, ref, alternative="two-sided", zero_method="wilcox").pvalue)
        p = p if p == "" else f"{p:.3g}"
        print(f"  {f:20s} {100*np.nanmean(x):7.2f} {100*np.nanmean(g['stat_viol']):7.2f} "
              f"{100*np.nanmean(g['routine_viol']):7.2f} {np.nanmean(g['makespan_over_lb']):6.3f} "
              f"{100*np.nanmean(g['goal_changed']) if not np.all(np.isnan(g['goal_changed'])) else float('nan'):6.2f}  {p}")
