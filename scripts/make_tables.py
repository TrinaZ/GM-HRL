#!/usr/bin/env python3
"""Aggregate evaluation outputs into paper tables.

Reads results/<name>/runs.csv; methods named NAME_s<k> (one per training seed)
are pooled into NAME. Writes results/tables.md and results/tables.tex with
mean ± std over all (instance, realisation, training seed) runs.

  python scripts/make_tables.py --glob 'results/main_*'
"""
import argparse, csv, glob, json, re
from collections import defaultdict
from pathlib import Path
import numpy as np

COLS = [("makespan", "Makespan (s)", 1, 0), ("gap_to_clairvoyant_LB", "Gap to OPT (%)", 100, 1),
        ("sla_violation_rate", "SLA viol. (%)", 100, 1), ("util_cpu", "CPU util. (%)", 100, 1),
        ("latency_ms_mean", "Latency (ms)", 1, 2)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--glob", default="results/main_*")
    p.add_argument("--out", default="results/tables")
    a = p.parse_args()
    md, tex = [], []
    for d in sorted(glob.glob(a.glob)):
        f = Path(d) / "runs.csv"
        if not f.exists():
            continue
        rows = list(csv.DictReader(open(f)))
        by = defaultdict(list)
        for r in rows:
            by[re.sub(r"_s\d+$", "", r["method"])].append(r)
        cols = [c for c in COLS if c[0] in rows[0] and rows[0][c[0]] not in ("", None)]
        md.append(f"\n### {Path(d).name}\n\n| Method | n | " + " | ".join(c[1] for c in cols) + " |")
        md.append("|---|---|" + "---|" * len(cols))
        tex.append(f"% {Path(d).name}\n\\begin{{tabular}}{{l" + "c" * len(cols) + "}\n\\toprule\nMethod & "
                   + " & ".join(c[1] for c in cols) + " \\\\\n\\midrule")
        for m, R in by.items():
            cells = []
            for key, _, sc, dec in cols:
                v = np.array([float(r[key]) for r in R]) * sc
                cells.append(f"{v.mean():.{dec}f} ± {v.std(ddof=1) if len(v) > 1 else 0:.{dec}f}")
            md.append(f"| {m} | {len(R)} | " + " | ".join(cells) + " |")
            tex.append(m.replace("_", "\\_") + " & " + " & ".join(c.replace("±", "$\\pm$") for c in cells) + " \\\\")
        tex.append("\\bottomrule\n\\end{tabular}\n")
    Path(a.out + ".md").write_text("\n".join(md))
    Path(a.out + ".tex").write_text("\n".join(tex))
    print("\n".join(md))


if __name__ == "__main__":
    main()
