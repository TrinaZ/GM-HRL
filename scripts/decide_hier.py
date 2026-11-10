#!/usr/bin/env python3
"""Pre-registered checks and hypotheses of HIERARCHY_SPEC.md (section 4).

Reads the validation evaluations <res>/val_<domain>/runs.csv (MD2POS-MW, K = 12,
s = 0.2, batches k = 50..54 x realisations 2000..2003) with the method names
written by run_hier.sh:
  GM-HRL-v2_s<seed>          own goals (goal_changed column = M1 measurement)
  GM-HRL-v2-shuffle_s<seed>  goals shuffled across workflows (M2, primary)
  GM-HRL-v2-random_s<seed>   random goals (M2, secondary, reported only)
  GNN-PPO_s<seed>            single-level counterpart (H1)
  WEDF                       weighted EDF rule (H2)
Every learned method's value per (k, realisation) is the mean over its three
training seeds; tests are two-sided Wilcoxon signed-rank tests on the 20 pairs.

  M1  pass if mean goal_changed >= 0.05, in both domains
  M2  pass if mean WV(own) < mean WV(shuffle) and p < 0.05, in both domains
  H1  pass if 1 - mean WV(v2) / mean WV(GNN-PPO) >= 0.05 and p < 0.05, in both domains
  H2  as H1 against WEDF
All results are reported whatever their outcome; the STAT and routine
violation rates of every method are printed alongside (descriptive).

  python scripts/decide_hier.py --res results_hier --out results_hier/hier_decision.json
"""
import argparse
import csv
import json
import re
from collections import defaultdict

import numpy as np
from scipy.stats import wilcoxon

DOMAINS = ("genomic", "video")
M1_MIN, REL_MIN, ALPHA = 0.05, 0.05, 0.05
N_PAIRS, N_SEEDS = 20, 3


def pvalue(x, y):
    if np.allclose(x, y):
        return 1.0
    return float(wilcoxon(x, y, alternative="two-sided", zero_method="wilcox").pvalue)


def family(name):
    """Method family: strip the training-seed suffix _s<seed>."""
    return re.sub(r"_s\d+$", "", name)


def load(path):
    """family -> (k, realisation) -> list over training seeds of row dicts."""
    out = defaultdict(lambda: defaultdict(list))
    seeds = defaultdict(set)
    for r in csv.DictReader(open(path)):
        f = family(r["method"])
        out[f][(int(r["k"]), int(r["seed"]))].append(r)
        seeds[f].add(r["method"])
    return out, seeds


def per_pair(fam, key, metric):
    """[N_PAIRS] values (mean over training seeds, NaN-aware) in a fixed pair order."""
    vals = []
    for kk in key:
        v = [float(r[metric]) for r in fam[kk] if r.get(metric) not in ("", None)]
        v = [x for x in v if not np.isnan(x)]
        vals.append(np.mean(v) if v else np.nan)
    return np.array(vals)


def compare(a, b):
    """a should be lower than b: relative reduction, p, pass (H-type criterion)."""
    rel = float(1 - a.mean() / b.mean()) if b.mean() > 0 else 0.0
    p = pvalue(a, b)
    return {"mean": float(a.mean()), "mean_ref": float(b.mean()), "rel_reduction": rel, "p": p,
            "pass": bool(rel >= REL_MIN and p < ALPHA)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="results_hier")
    ap.add_argument("--out", default="results_hier/hier_decision.json")
    a = ap.parse_args()
    report = {"criteria": {"M1": f"mean goal_changed >= {M1_MIN} in both domains",
                           "M2": f"WV(own) < WV(shuffled across workflows), p < {ALPHA}, both domains; "
                                 "random goals reported only",
                           "H1": f"GM-HRL v2 vs GNN-PPO: relative WV reduction >= {REL_MIN:.0%}, p < {ALPHA}, both domains",
                           "H2": f"GM-HRL v2 vs WEDF: same criterion"},
              "domains": {}}
    passes = defaultdict(list)
    for d in DOMAINS:
        data, seeds = load(f"{a.res}/val_{d}/runs.csv")
        for f in ("GM-HRL-v2", "GM-HRL-v2-shuffle", "GM-HRL-v2-random", "GNN-PPO"):
            assert len(seeds[f]) == N_SEEDS, (d, f, seeds[f])
        key = sorted(data["GM-HRL-v2"])
        assert len(key) == N_PAIRS, len(key)
        for f in data:
            assert sorted(data[f]) == key, (d, f)
        wv = {f: per_pair(data[f], key, "wv") for f in data}
        rep = {"n_pairs": len(key)}
        # M1
        chg = per_pair(data["GM-HRL-v2"], key, "goal_changed")
        rep["M1"] = {"goal_changed": float(np.nanmean(chg)), "pass": bool(np.nanmean(chg) >= M1_MIN)}
        # M2 (primary: shuffled; secondary: random)
        own, sh, rnd = wv["GM-HRL-v2"], wv["GM-HRL-v2-shuffle"], wv["GM-HRL-v2-random"]
        p_sh = pvalue(own, sh)
        rep["M2"] = {"wv_own": float(own.mean()), "wv_shuffle": float(sh.mean()), "p": p_sh,
                     "pass": bool(own.mean() < sh.mean() and p_sh < ALPHA),
                     "secondary_random": {"wv_random": float(rnd.mean()), "p": pvalue(own, rnd)}}
        rep["H1"] = compare(own, wv["GNN-PPO"])
        rep["H2"] = compare(own, wv["WEDF"])
        for h in ("M1", "M2", "H1", "H2"):
            passes[h].append(rep[h]["pass"])
        # descriptive: every method
        rep["methods"] = {}
        for f in sorted(data):
            row = {m: float(np.nanmean(per_pair(data[f], key, m)))
                   for m in ("wv", "stat_viol", "routine_viol", "w_tardiness", "makespan_over_lb",
                             "sla_violation_rate", "latency_ms_mean")}
            if f != "GM-HRL-v2":
                row["p_wv_vs_v2"] = pvalue(own, wv[f])
            rep["methods"][f] = row
        report["domains"][d] = rep

        print(f"== {d} (validation, {len(key)} pairs, learned methods averaged over {N_SEEDS} seeds)")
        print(f"  M1 goal_changed = {rep['M1']['goal_changed']:.3f}  {'PASS' if rep['M1']['pass'] else 'fail'}")
        print(f"  M2 WV own {own.mean():.4f} vs shuffled {sh.mean():.4f} (p={p_sh:.3g})  "
              f"{'PASS' if rep['M2']['pass'] else 'fail'};  random {rnd.mean():.4f} "
              f"(p={rep['M2']['secondary_random']['p']:.3g}, secondary)")
        for h, ref in (("H1", "GNN-PPO"), ("H2", "WEDF")):
            c = rep[h]
            print(f"  {h} WV v2 {c['mean']:.4f} vs {ref} {c['mean_ref']:.4f}: reduction "
                  f"{100 * c['rel_reduction']:+.1f}% (p={c['p']:.3g})  {'PASS' if c['pass'] else 'fail'}")
        print(f"  {'method':20s} {'WV':>7s} {'STAT':>7s} {'routine':>7s} {'wTard':>8s} {'mk/LB':>6s}")
        for f, row in rep["methods"].items():
            print(f"  {f:20s} {row['wv']:7.4f} {row['stat_viol']:7.4f} {row['routine_viol']:7.4f} "
                  f"{row['w_tardiness']:8.1f} {row['makespan_over_lb']:6.3f}")
    report["result"] = {h: bool(all(v)) for h, v in passes.items()}
    m1, m2 = report["result"]["M1"], report["result"]["M2"]
    report["interpretation"] = (
        "the upper level is used and carries information" if m1 and m2 else
        "the lower level uses the goal, but the learned goals carry no information" if m1 else
        "the lower level ignores the goal; the interface does not work")
    print("RESULT:", {h: ("PASS" if v else "fail") for h, v in report["result"].items()})
    print("M1/M2:", report["interpretation"])
    print("H1:", "the hierarchy helps in this setting" if report["result"]["H1"]
          else "the hierarchy did not help in this setting either")
    json.dump(report, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
