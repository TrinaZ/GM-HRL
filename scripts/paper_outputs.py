#!/usr/bin/env python3
"""Generate every table and figure of Section 6 from the evaluation outputs.

    python scripts/paper_outputs.py --res results_v2 --runs runs_v2 --out paper_out

Inputs  : <res>/main_<domain>_N<N>/runs.csv (and bounds.csv), <res>/ablation_genomic_N200,
          <res>/robust_*, <res>/xdomain_*, <res>/size_N*, <runs>/*/train_log.jsonl
Outputs : <out>/tables/*.tex and *.csv, <out>/figs/*.pdf, <out>/numbers.md (all numbers quoted in the text)
Learning-based methods trained with several seeds are averaged over seeds for every
(instance, realisation) pair before paired tests; Wilcoxon signed-rank p-values are
Holm-corrected within each table row.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

DOMAINS = ["genomic", "video", "ai_ft", "financial"]
DOM_LABEL = {"genomic": "Genomics", "video": "Video", "ai_ft": "AI fine-tuning", "financial": "Financial"}
SCALES = [50, 100, 200, 500, 1000]
METHODS = ["FIFO", "SJF", "Random", "GA", "ORH", "PPO", "GNN-PPO", "GM-HRL"]
LEARNED = ["PPO", "GNN-PPO", "GM-HRL"]
# fixed colour per method (validated categorical palette, light mode); rules in greys
COLOR = {"GM-HRL": "#2a78d6", "PPO": "#eb6834", "GNN-PPO": "#1baf7a", "GA": "#eda100", "ORH": "#e87ba4",
         "FIFO": "#8a8984", "SJF": "#b3b2ab", "Random": "#cfcec7", "HEFT": "#0b0b0b"}
MARKER = {"GM-HRL": "o", "PPO": "s", "GNN-PPO": "^", "GA": "D", "ORH": "v", "FIFO": "x", "SJF": "+", "Random": "."}


def fam(name: str) -> str:
    n = re.sub(r"_s\d+$", "", name)
    return "ORH" if n == "ORHG" else n


def load(path: str) -> pd.DataFrame | None:
    f = Path(path) / "runs.csv"
    if not f.exists():
        return None
    r = pd.read_csv(f)
    r["fam"] = r.method.map(fam)
    return r


def per_pair(r: pd.DataFrame, col: str) -> pd.DataFrame:
    """(k, seed) x method table, learned methods averaged over training seeds."""
    return r.groupby(["fam", "k", "seed"])[col].mean().unstack(0)


def holm(ps: dict) -> dict:
    items = sorted(ps.items(), key=lambda kv: kv[1])
    m, out, run = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = run
    return out


def wil(a, b) -> float:
    d = np.asarray(a) - np.asarray(b)
    if np.allclose(d, 0):
        return 1.0
    try:
        return float(wilcoxon(a, b).pvalue)
    except ValueError:
        return 1.0


def star(p: float) -> str:
    return "^{**}" if p < 0.01 else ("^{*}" if p < 0.05 else "")


def rel(mk: pd.DataFrame, m: str, ref: str = "HEFT") -> float:
    return float((mk[m] / mk[ref] - 1).mean() * 100)


class Out:
    def __init__(self, root: str):
        self.root = Path(root)
        (self.root / "tables").mkdir(parents=True, exist_ok=True)
        (self.root / "figs").mkdir(parents=True, exist_ok=True)
        self.notes: list[str] = []

    def table(self, name: str, df: pd.DataFrame, tex: str):
        df.to_csv(self.root / "tables" / f"{name}.csv", index=False)
        (self.root / "tables" / f"{name}.tex").write_text(tex, encoding="utf-8")

    def note(self, s: str):
        self.notes.append(s)


# ----------------------------------------------------------------------------- tables
def tab_main(res: str, out: Out):
    """Makespan relative to HEFT (%) for all domains and scales (RQ1)."""
    rows, tex_rows = [], []
    present = [m for m in METHODS if any(
        (lambda r: r is not None and m in set(r.fam))(load(f"{res}/main_{d}_N{n}")) for d in DOMAINS for n in SCALES)]
    for d in DOMAINS:
        for n in SCALES:
            r = load(f"{res}/main_{d}_N{n}")
            if r is None:
                continue
            mk = per_pair(r, "makespan")
            ms = [m for m in present if m in mk]
            ps = holm({m: wil(mk[m], mk["HEFT"]) for m in ms})
            vals = {m: rel(mk, m) for m in ms}
            best = min(vals, key=vals.get)
            row = {"domain": d, "N": n, "HEFT_makespan": float(mk["HEFT"].mean())}
            cells = []
            for m in present:
                if m not in vals:
                    cells.append("--"); continue
                row[m] = vals[m]; row[m + "_p"] = ps[m]
                c = f"{vals[m]:+.1f}{star(ps[m])}"
                cells.append(f"$\\mathbf{{{c}}}$" if m == best and vals[m] < 0 else f"${c}$")
            rows.append(row)
            first = not any(x["domain"] == d for x in rows[:-1])
            tex_rows.append(f"{DOM_LABEL[d] if first else ''} & {n} & " + " & ".join(cells) + r" \\")
        tex_rows.append(r"\midrule")
    df = pd.DataFrame(rows)
    head = " & ".join(present)
    tex = ("\\begin{tabular}{ll" + "r" * len(present) + "}\n\\toprule\nDomain & $N$ & " + head + " \\\\\n\\midrule\n"
           + "\n".join(tex_rows[:-1]) + "\n\\bottomrule\n\\end{tabular}\n")
    out.table("tab_makespan_vs_heft", df, tex)
    for d in DOMAINS:
        sub = df[df.domain == d]
        if len(sub) and "GM-HRL" in sub:
            out.note(f"main/{d}: GM-HRL vs HEFT by N: " + ", ".join(
                f"N={int(a.N)} {a['GM-HRL']:+.2f}% (p={a['GM-HRL_p']:.2g})" for _, a in sub.iterrows()))
    return df


def tab_gap(res: str, out: Out, bounds_dir: str | None):
    """Gap to the offline clairvoyant bound (%), N <= 200 (RQ1, R3-7)."""
    rows, tex_rows = [], []
    for d in DOMAINS:
        for n in [50, 100, 200]:
            r = load(f"{res}/main_{d}_N{n}")
            if r is None or "gap_to_clairvoyant_LB" not in r:
                continue
            g = per_pair(r, "gap_to_clairvoyant_LB") * 100
            bf = Path(res) / f"main_{d}_N{n}" / "bounds.csv"
            if not bf.exists() and bounds_dir:
                bf = Path(bounds_dir) / f"main_{d}_N{n}" / "bounds.csv"
            opt, inc = float("nan"), float("nan")
            if bf.exists():
                bd = pd.read_csv(bf)
                opt = float((bd.cpsat_status == "OPTIMAL").mean() * 100)
                if "cpsat_incumbent" in bd:   # offline clairvoyant schedule found by CP-SAT, relative to the same bound
                    inc = float(((bd.cpsat_incumbent - bd.best_LB) / bd.best_LB).clip(lower=0).mean() * 100)
            ms = [m for m in ["HEFT"] + METHODS if m in g]
            row = {"domain": d, "N": n, "proven_optimal_pct": opt, "offline_incumbent": inc, **{m: float(g[m].mean()) for m in ms}}
            rows.append(row)
            best = min(ms, key=lambda m: row[m])
            cells = [f"\\textbf{{{row[m]:.1f}}}" if m == best else f"{row[m]:.1f}" for m in ms]
            tex_rows.append(f"{DOM_LABEL[d] if n == 50 else ''} & {n} & {opt:.0f} & {inc:.1f} & " + " & ".join(cells) + r" \\")
    if not rows:
        return None
    df = pd.DataFrame(rows)
    ms = [m for m in ["HEFT"] + METHODS if m in df]
    tex = ("\\begin{tabular}{llrr" + "r" * len(ms) + "}\n\\toprule\nDomain & $N$ & opt.\\,\\% & offline & " + " & ".join(ms)
           + " \\\\\n\\midrule\n" + "\n".join(tex_rows) + "\n\\bottomrule\n\\end{tabular}\n")
    out.table("tab_gap_to_bound", df, tex)
    out.note("gap to bound (GM-HRL / HEFT / best): " + "; ".join(
        f"{a.domain} N={a.N}: {a.get('GM-HRL', float('nan')):.1f}/{a.HEFT:.1f}/{min(a[m] for m in ms):.1f} (opt {a.proven_optimal_pct:.0f}%)"
        for _, a in df.iterrows()))
    return df


def tab_service(res: str, out: Out):
    """SLA violation rate (%) and mean response time relative to HEFT (%)."""
    rows = []
    ms = ["HEFT", "GA", "ORH", "PPO", "GNN-PPO", "GM-HRL"]
    for d in DOMAINS:
        for n in SCALES:
            r = load(f"{res}/main_{d}_N{n}")
            if r is None:
                continue
            sl = per_pair(r, "sla_violation_rate") * 100
            rs = per_pair(r, "mean_response")
            row = {"domain": d, "N": n}
            for m in ms:
                if m in sl:
                    row[f"{m}_sla"] = float(sl[m].mean())
                    row[f"{m}_resp"] = rel(rs, m)
                    row[f"{m}_resp_p"] = wil(rs[m], rs["HEFT"]) if m != "HEFT" else 1.0
            rows.append(row)
    df = pd.DataFrame(rows)
    present = [m for m in ms if f"{m}_sla" in df]
    tex = ["\\begin{tabular}{ll" + "r" * len(present) + "|" + "r" * (len(present) - 1) + "}", "\\toprule",
           "& & \\multicolumn{%d}{c|}{SLA violation rate (\\%%)} & \\multicolumn{%d}{c}{Mean response vs.\\ HEFT (\\%%)} \\\\"
           % (len(present), len(present) - 1),
           "Domain & $N$ & " + " & ".join(present) + " & " + " & ".join(present[1:]) + " \\\\", "\\midrule"]
    for i, a in df.iterrows():
        cells = [f"{a[m + '_sla']:.1f}" for m in present] + [f"${a[m + '_resp']:+.1f}{star(a[m + '_resp_p'])}$" for m in present[1:]]
        tex.append(f"{DOM_LABEL[a.domain] if a.N == df[df.domain == a.domain].N.min() else ''} & {a.N} & " + " & ".join(cells) + r" \\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    out.table("tab_service", df, "\n".join(tex) + "\n")
    return df


def tab_ablation(res: str, out: Out):
    r = load(f"{res}/ablation_genomic_N200")
    if r is None:
        return None
    mk, sl, rs = per_pair(r, "makespan"), per_pair(r, "sla_violation_rate") * 100, per_pair(r, "mean_response")
    variants = [v for v in ["GM-HRL", "no_hierarchy", "no_gnn", "no_dynamic_graph", "kappa0.3", "kappa3", "kappa10", "GA"] if v in mk]
    label = {"GM-HRL": "GM-HRL (full)", "no_hierarchy": "w/o hierarchy", "no_gnn": "w/o GNN encoder",
             "no_dynamic_graph": "w/o dynamic graph", "kappa0.3": "$\\kappa=0.3$", "kappa3": "$\\kappa=3$",
             "kappa10": "$\\kappa=10$", "GA": "GA (reference)"}
    rows, tex = [], ["\\begin{tabular}{lrrrr}", "\\toprule",
                     "Variant & Makespan vs.\\ HEFT (\\%) & vs.\\ full (\\%) & SLA viol.\\ (\\%) & Response vs.\\ HEFT (\\%) \\\\", "\\midrule"]
    for v in variants:
        p_full = wil(mk[v], mk["GM-HRL"]) if v != "GM-HRL" else 1.0
        row = dict(variant=v, mk_vs_heft=rel(mk, v), mk_vs_full=rel(mk, v, "GM-HRL"), p_vs_full=p_full,
                   sla=float(sl[v].mean()), resp_vs_heft=rel(rs, v))
        rows.append(row)
        tex.append(f"{label[v]} & ${row['mk_vs_heft']:+.2f}$ & " + ("--" if v == "GM-HRL" else f"${row['mk_vs_full']:+.2f}{star(p_full)}$")
                   + f" & {row['sla']:.1f} & ${row['resp_vs_heft']:+.1f}$ \\\\")
        if v == "no_dynamic_graph":
            tex.append("\\midrule")
    tex += ["\\bottomrule", "\\end{tabular}"]
    df = pd.DataFrame(rows)
    out.table("tab_ablation", df, "\n".join(tex) + "\n")
    out.note("ablation (genomic N=200): " + "; ".join(f"{a.variant} {a.mk_vs_heft:+.2f}% vs HEFT, {a.mk_vs_full:+.2f}% vs full (p={a.p_vs_full:.2g}), SLA {a.sla:.1f}%" for _, a in df.iterrows()))
    return df


def _cond_table(res: str, names: dict, out: Out, fname: str, title: str):
    rows = []
    for key, lab in names.items():
        r = load(f"{res}/{key}")
        if r is None:
            continue
        mk, sl = per_pair(r, "makespan"), per_pair(r, "sla_violation_rate") * 100
        row = {"condition": lab, "HEFT_sla": float(sl["HEFT"].mean())}
        for m in mk.columns:
            if m == "HEFT":
                continue
            row[m] = rel(mk, m); row[m + "_p"] = wil(mk[m], mk["HEFT"]); row[m + "_sla"] = float(sl[m].mean())
        rows.append(row)
    if not rows:
        return None
    df = pd.DataFrame(rows)
    cols = [c for c in df.columns if c not in ("condition", "HEFT_sla") and not c.endswith(("_p", "_sla"))]
    order = METHODS + sorted(c for c in cols if c not in METHODS)
    ms = [c for c in order if c in cols and c != "Random" and df[c].notna().sum() > 1 or (c in cols and c not in METHODS)]
    tex = ["\\begin{tabular}{l" + "r" * len(ms) + "}", "\\toprule", f"{title} & " + " & ".join(ms) + " \\\\", "\\midrule"]
    for _, a in df.iterrows():
        tex.append(a.condition + " & " + " & ".join("--" if pd.isna(a.get(m)) else f"${a[m]:+.2f}{star(a[m + '_p'])}$" for m in ms) + " \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    out.table(fname, df, "\n".join(tex) + "\n")
    return df


def tab_robust(res: str, out: Out):
    names = {"robust_rrd0": "no RRD ($p_{rrd}\\times 0$)", "main_genomic_N200": "default", "robust_rrd2.6667": "RRD $\\times 2.67$",
             "robust_rho0.6": "load $\\rho=0.6$", "robust_rho0.95": "load $\\rho=0.95$", "robust_cap0.7": "capacity $\\times 0.7$"}
    df = _cond_table(res, names, out, "tab_robustness", "Condition (genomic, $N=200$)")
    if df is not None and "GM-HRL" in df:
        out.note("robustness GM-HRL vs HEFT: " + "; ".join(f"{a.condition}: {a['GM-HRL']:+.2f}%" for _, a in df.iterrows()))
    return df


def tab_transfer(res: str, out: Out):
    names = {f"xdomain_{d}": f"genomic $\\to$ {DOM_LABEL[d]}" for d in ["video", "ai_ft", "financial"]}
    names.update({f"size_N{n}": f"$N{{=}}100 \\to {n}$" for n in [200, 500, 1000]})
    df = _cond_table(res, names, out, "tab_transfer", "Zero-shot transfer")
    if df is not None:
        out.note("transfer: " + "; ".join(a.condition + ": " + ", ".join(f"{m} {a[m]:+.2f}%" for m in df.columns
                 if m not in ('condition', 'HEFT_sla') and not m.endswith(('_p', '_sla')) and not pd.isna(a[m])) for _, a in df.iterrows()))
    return df


def tab_training(runs: str, out: Out):
    """Training cost and episodes until the validation makespan stays within 5% of its final value (R3-2)."""
    rows = []
    for d in sorted(glob.glob(f"{runs}/*/train_log.jsonl")):
        run = Path(d).parent.name
        m = re.match(r"(gmhrl|ppo|gnn_ppo)_(\w+?)_N(\d+)_s(\d+)$", run)
        if not m:
            continue
        cfg = json.load(open(Path(d).parent / "config.json"))["args"]
        L = [json.loads(x) for x in open(d)]
        rl = [x for x in L if "iter" in x and "val_makespan_over_lb" in x]
        if not rl:
            continue
        v = np.array([x["val_makespan_over_lb"] for x in rl]); it = np.array([x["iter"] for x in rl])
        final = v[-1]; ok = np.abs(v - final) / final <= 0.05
        first = next((i for i in range(len(v)) if ok[i:].all()), len(v) - 1)
        eps = cfg.get("bc_iters", 0) * cfg.get("episodes_per_iter", 4) + (it[first] + 1) * cfg.get("episodes_per_iter", 4)
        rows.append(dict(method={"gmhrl": "GM-HRL", "ppo": "PPO", "gnn_ppo": "GNN-PPO"}[m.group(1)], domain=m.group(2), N=int(m.group(3)),
                         seed=int(m.group(4)), hours=rl[-1]["time_s"] / 3600, episodes_to_5pct=eps,
                         val_first=float(v[0]), val_best=float(v.min()), val_final=float(final)))
    if not rows:
        return None
    df = pd.DataFrame(rows)
    agg = df.groupby(["method", "N"]).agg(hours=("hours", "mean"), episodes=("episodes_to_5pct", "mean"),
                                          improve_pct=("val_first", "mean")).reset_index()
    agg["improve_pct"] = (df.groupby(["method", "N"]).apply(lambda g: ((g.val_first - g.val_best) / g.val_first).mean() * 100)).values
    tex = ["\\begin{tabular}{llrrr}", "\\toprule", "Method & $N$ & Training time (h) & Episodes to 5\\,\\% & Val.\\ improvement (\\%) \\\\", "\\midrule"]
    for _, a in agg.iterrows():
        tex.append(f"{a.method} & {a.N} & {a.hours:.1f} & {a.episodes:,.0f} & {a.improve_pct:.1f} \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    out.table("tab_training", agg, "\n".join(tex) + "\n")
    df.to_csv(out.root / "tables" / "training_runs.csv", index=False)
    return df


def tab_latency(res: str, out: Out):
    rows = []
    for n in SCALES:
        r = load(f"{res}/main_genomic_N{n}")
        if r is None or "latency_ms_mean" not in r:
            continue
        lat = r.groupby("fam")["latency_ms_mean"].mean()
        rows.append({"N": n, **{m: float(lat[m]) for m in lat.index}})
    if not rows:
        return None
    df = pd.DataFrame(rows)
    ms = [m for m in ["HEFT", "GA", "ORH", "PPO", "GNN-PPO", "GM-HRL"] if m in df]
    tex = ["\\begin{tabular}{l" + "r" * len(ms) + "}", "\\toprule", "$N$ & " + " & ".join(ms) + " \\\\", "\\midrule"]
    tex += [f"{int(a.N)} & " + " & ".join(f"{a[m]:.2f}" for m in ms) + " \\\\" for _, a in df.iterrows()]
    tex += ["\\bottomrule", "\\end{tabular}"]
    out.table("tab_latency_from_eval", df, "\n".join(tex) + "\n")
    out.note("NOTE: latency from evaluation runs was measured on a loaded machine; use the dedicated measurement for the paper.")
    return df


# ----------------------------------------------------------------------------- figures
def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e",
                         "ytick.color": "#52514e", "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
                         "legend.frameon": False, "lines.linewidth": 1.6, "lines.markersize": 4.5, "pdf.fonttype": 42})
    return plt


def fig_scale(main: pd.DataFrame, out: Out):
    plt = _style()
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.2), sharey=False)
    ms = [m for m in ["FIFO", "GA", "ORH", "PPO", "GNN-PPO", "GM-HRL"] if m in main]
    for ax, d in zip(axes, DOMAINS):
        sub = main[main.domain == d].sort_values("N")
        ax.axhline(0, color="#0b0b0b", lw=0.9)
        for m in ms:
            ax.plot(range(len(sub)), sub[m], color=COLOR[m], marker=MARKER[m], label=m, zorder=3 if m == "GM-HRL" else 2)
        ax.set_xticks(range(len(sub))); ax.set_xticklabels(sub.N.astype(int))
        ax.set_title(DOM_LABEL[d], fontsize=8.5); ax.set_xlabel("$N$")
    axes[0].set_ylabel("Makespan vs. HEFT (%)")
    axes[0].text(0.02, 0.02, "below 0 = better than HEFT", transform=axes[0].transAxes, fontsize=6.5, color="#52514e")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(ms), bbox_to_anchor=(0.5, 1.07), fontsize=7.5)
    fig.tight_layout()
    fig.savefig(out.root / "figs" / "fig_scale_vs_heft.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_robust(rob: pd.DataFrame, out: Out):
    if rob is None:
        return
    plt = _style()
    ms = [m for m in ["GA", "PPO", "GNN-PPO", "GM-HRL"] if m in rob]
    lab = dict(zip(rob.condition, range(len(rob))))
    panels = [("RRD intensity", ["no RRD ($p_{rrd}\\times 0$)", "default", "RRD $\\times 2.67$"], ["0", "1", "2.67"], "RRD multiplier"),
              ("Offered load", ["load $\\rho=0.6$", "default", "load $\\rho=0.95$"], ["0.6", "0.8", "0.95"], "$\\rho$")]
    fig, axes = plt.subplots(1, 2, figsize=(5.0, 2.2), sharey=True)
    for ax, (title, conds, ticks, xl) in zip(axes, panels):
        idx = [lab[c] for c in conds if c in lab]
        sub = rob.iloc[idx]
        ax.axhline(0, color="#0b0b0b", lw=0.9)
        for m in ms:
            ax.plot(range(len(sub)), sub[m], color=COLOR[m], marker=MARKER[m], label=m)
        ax.set_xticks(range(len(sub))); ax.set_xticklabels(ticks[:len(sub)]); ax.set_xlabel(xl); ax.set_title(title, fontsize=8.5)
    axes[0].set_ylabel("Makespan vs. HEFT (%)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(ms), bbox_to_anchor=(0.5, 1.08), fontsize=7.5)
    fig.tight_layout()
    fig.savefig(out.root / "figs" / "fig_robustness.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_tradeoff(res: str, out: Out):
    """SLA violation rate vs makespan (relative to HEFT) for kappa variants and baselines, genomic N=200."""
    r = load(f"{res}/ablation_genomic_N200")
    if r is None:
        return
    plt = _style()
    mk, sl = per_pair(r, "makespan"), per_pair(r, "sla_violation_rate") * 100
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    pts = {"HEFT": (0.0, sl["HEFT"].mean())}
    for v in mk.columns:
        if v.startswith("kappa") or v in ("GM-HRL", "GA"):
            pts[v] = (rel(mk, v), sl[v].mean())
    ks = sorted([(float(v[5:]), v) for v in pts if v.startswith("kappa")] + ([(1.0, "GM-HRL")] if "GM-HRL" in pts else []))
    ax.plot([pts[v][0] for _, v in ks], [pts[v][1] for _, v in ks], color=COLOR["GM-HRL"], marker="o", zorder=3)
    for k, v in ks:
        ax.annotate(f"$\\kappa$={k:g}", pts[v], textcoords="offset points", xytext=(4, 3), fontsize=6.5, color="#52514e")
    for v in ("HEFT", "GA"):
        if v in pts:
            ax.scatter(*pts[v], color=COLOR[v], marker="D" if v == "GA" else "s", s=22, zorder=3)
            ax.annotate(v, pts[v], textcoords="offset points", xytext=(4, -9), fontsize=7)
    ax.set_xlabel("Makespan vs. HEFT (%)"); ax.set_ylabel("SLA violation rate (%)")
    fig.tight_layout(); fig.savefig(out.root / "figs" / "fig_tradeoff.pdf", bbox_inches="tight"); plt.close(fig)


def fig_convergence(train: pd.DataFrame, runs: str, out: Out, domain="genomic", N=200):
    if train is None:
        return
    plt = _style()
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for meth, pref in [("PPO", "ppo"), ("GNN-PPO", "gnn_ppo"), ("GM-HRL", "gmhrl")]:
        curves = []
        for f in sorted(glob.glob(f"{runs}/{pref}_{domain}_N{N}_s*/train_log.jsonl")):
            cfg = json.load(open(Path(f).parent / "config.json"))["args"]
            epi = cfg.get("episodes_per_iter", 4); bc = cfg.get("bc_iters", 0) * epi
            L = [json.loads(x) for x in open(f) if '"val_makespan_over_lb"' in x and '"iter"' in x]
            curves.append(pd.Series([x["val_makespan_over_lb"] for x in L], index=[bc + (x["iter"] + 1) * epi for x in L]))
        if not curves:
            continue
        c = pd.concat(curves, axis=1).sort_index()
        ax.plot(c.index, c.mean(axis=1), color=COLOR[meth], label=meth)
        ax.fill_between(c.index, c.min(axis=1), c.max(axis=1), color=COLOR[meth], alpha=0.15, lw=0)
    ax.set_xlabel("Training episodes"); ax.set_ylabel("Validation makespan / LB")
    ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(out.root / "figs" / f"fig_convergence_{domain}_N{N}.pdf", bbox_inches="tight"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="results_v2")
    ap.add_argument("--runs", default="runs_v2")
    ap.add_argument("--bounds", default="results", help="directory with first-round bounds.csv files")
    ap.add_argument("--out", default="paper_out")
    a = ap.parse_args()
    out = Out(a.out)
    main_df = tab_main(a.res, out)
    tab_gap(a.res, out, a.bounds)
    tab_service(a.res, out)
    tab_ablation(a.res, out)
    rob = tab_robust(a.res, out)
    tab_transfer(a.res, out)
    train = tab_training(a.runs, out)
    tab_latency(a.res, out)
    fig_scale(main_df, out)
    fig_robust(rob, out)
    fig_tradeoff(a.res, out)
    fig_convergence(train, a.runs, out)
    (out.root / "numbers.md").write_text("# Numbers for the text of Section 6\n\n" + "\n\n".join(f"- {n}" for n in out.notes) + "\n", encoding="utf-8")
    print("wrote", out.root)


if __name__ == "__main__":
    main()
