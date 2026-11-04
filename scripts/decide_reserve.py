#!/usr/bin/env python3
"""Pre-registered adoption decision for GM-HRL+R (RESERVATION_SPEC.md, section 4).

Reads the validation evaluations <dir>/val_res_<domain>_N200/runs.csv, in which
the control models are named GM-HRL_s<seed> and the new models GM-HRL+R_s<seed>,
and prints/saves the decision. Usage:
  python scripts/decide_reserve.py --res results_v2 --out results_v2/reserve_decision.json
"""
import argparse
import csv
import json
from collections import defaultdict

import numpy as np
from scipy.stats import wilcoxon

DOMAINS = ("genomic", "video")


def family(name):
    if name.startswith("GM-HRL+R_s"):
        return "R"
    if name.startswith("GM-HRL_s"):
        return "0"
    return None


def pvalue(x, y):
    if np.allclose(x, y):
        return 1.0
    return float(wilcoxon(x, y, alternative="two-sided", zero_method="wilcox").pvalue)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="results_v2")
    ap.add_argument("--out", default="results_v2/reserve_decision.json")
    a = ap.parse_args()
    report = {"criterion": "pass if (dmk <= -1% and p_mk < 0.05) or (dmk <= +0.5% and dsla <= -5% and "
                           "p_sla < 0.05); adopt iff both genomic and video pass", "domains": {}}
    passes = []
    for d in DOMAINS:
        vals = defaultdict(lambda: defaultdict(list))      # (k, seed) -> family -> [(mk, sla)]
        seeds_seen = defaultdict(set)
        for r in csv.DictReader(open(f"{a.res}/val_res_{d}_N200/runs.csv")):
            f = family(r["method"])
            if f is None:
                continue
            vals[(r["k"], r["seed"])][f].append((float(r["makespan"]), float(r["sla_violation_rate"])))
            seeds_seen[f].add(r["method"])
        assert len(seeds_seen["R"]) == 3 and len(seeds_seen["0"]) == 3, seeds_seen
        keys = sorted(vals)
        assert len(keys) == 20, len(keys)
        mkR = np.array([np.mean([v[0] for v in vals[k]["R"]]) for k in keys])
        mk0 = np.array([np.mean([v[0] for v in vals[k]["0"]]) for k in keys])
        slR = np.array([np.mean([v[1] for v in vals[k]["R"]]) for k in keys])
        sl0 = np.array([np.mean([v[1] for v in vals[k]["0"]]) for k in keys])
        dmk = float(np.mean(mkR / mk0 - 1))
        dsla = float(slR.mean() / sl0.mean() - 1) if sl0.mean() > 0 else 0.0
        p_mk, p_sla = pvalue(mkR, mk0), pvalue(slR, sl0)
        ok = (dmk <= -0.01 and p_mk < 0.05) or (dmk <= 0.005 and dsla <= -0.05 and p_sla < 0.05)
        passes.append(ok)
        report["domains"][d] = {"n_pairs": len(keys), "dmk": dmk, "p_mk": p_mk, "sla_R": float(slR.mean()),
                                "sla_0": float(sl0.mean()), "dsla_rel": dsla, "p_sla": p_sla, "pass": ok}
        print(f"{d:8s} dmk={100*dmk:+.2f}% (p={p_mk:.3g})  SLA {sl0.mean():.3f} -> {slR.mean():.3f} "
              f"({100*dsla:+.1f}%, p={p_sla:.3g})  {'PASS' if ok else 'fail'}")
    report["adopt"] = bool(all(passes))
    print("DECISION:", "ADOPT GM-HRL+R" if report["adopt"] else "DO NOT ADOPT (report: no improvement)")
    json.dump(report, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
