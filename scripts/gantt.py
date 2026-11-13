"""Fig.: real-pipeline Gantt charts (batch 1, run 2) from the dispatcher logs."""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

root = sys.argv[1]; out = sys.argv[2]
POL = [("heft", "HEFT"), ("gmhrl", "GM-HRL"), ("fifo", "FIFO")]
GROUP = {"fastqc": "FastQC", "bwa": "Alignment (BWA, sort, index)", "sort": "Alignment (BWA, sort, index)",
         "index": "Alignment (BWA, sort, index)", "gatk": "GATK (revealed)", "freebayes": "FreeBayes (revealed)",
         "bcftools": "Merge and report", "multiqc": "Merge and report"}
COL = {"FastQC": "#2a78d6", "Alignment (BWA, sort, index)": "#eb6834", "GATK (revealed)": "#1baf7a",
       "FreeBayes (revealed)": "#6250d6", "Merge and report": "#8a8984"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def load(p):
    L = [json.loads(l) for l in open(f"{root}/rep2_{p}/events.jsonl")]
    head = L[0]
    arr = dict(head["samples"])
    cpu = {s["name"]: s["cpu"] for s in head["spec"]["stages"]}
    start, tasks = {}, []
    for e in L:
        if e["ev"] == "dispatch":
            start[e["task"]] = (e["t"], e["sample"], e["stage"])
        elif e["ev"] == "end":
            t0, s, st = start[e["task"]]
            tasks.append((s, st, t0 / 3600, e["t"] / 3600, cpu[st]))
    return arr, tasks


data = {p: load(p) for p, _ in POL}
arr = data["heft"][0]
order = sorted(arr, key=lambda s: arr[s])            # rows: samples in arrival order
row = {s: i for i, s in enumerate(order)}
tmax = max(max(t[3] for t in d[1]) for d in data.values())

plt.rcParams.update({"font.size": 8.5, "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "axes.edgecolor": INK2, "axes.linewidth": 0.8,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "xtick.direction": "out", "ytick.direction": "out"})
fig = plt.figure(figsize=(7.2, 8.6))
gs = fig.add_gridspec(4, 1, height_ratios=[1, 1, 1, 0.62], hspace=0.28)
for k, (p, name) in enumerate(POL):
    ax = fig.add_subplot(gs[k])
    _, tasks = data[p]
    for s, st, t0, t1, c in tasks:
        y = row[s]
        ax.barh(y, t1 - t0, left=t0, height=0.72, color=COL[GROUP[st]], edgecolor="white", linewidth=0.3)
    ax.scatter([arr[s] / 3600 for s in order], [row[s] for s in order], marker="|", s=14, color=INK, lw=0.8, zorder=3)
    done = {}
    for s, st, t0, t1, c in tasks:
        done[s] = max(done.get(s, 0), t1)
    resp = np.mean([done[s] - arr[s] / 3600 for s in order])
    mk = max(done.values())
    fb = sorted({row[s] for s, st, *_ in tasks if st == "freebayes"})
    ax.scatter([tmax * 1.012] * len(fb), fb, marker="<", s=16, color=COL["FreeBayes (revealed)"], clip_on=False, zorder=4)
    ax.set_xlim(0, tmax * 1.01); ax.set_ylim(len(order) - 0.4, -0.6)
    ax.set_yticks([0, 9, 19, 29, 39, 49]); ax.set_yticklabels(["1", "10", "20", "30", "40", "50"])
    ax.set_ylabel("Sample (arrival order)")
    ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.set_title(f"({'abc'[k]}) {name}: makespan {mk:.2f} h, mean sample response {resp:.2f} h",
                 loc="left", fontsize=8.5, color=INK)
    if k < 2: ax.set_xticklabels([])
# CPU in use
ax = fig.add_subplot(gs[3])
grid_t = np.linspace(0, tmax, 2000)
styles = {"heft": ("#2a78d6", "-"), "gmhrl": ("#eb6834", "-"), "fifo": ("#1baf7a", "--")}
for p, name in POL:
    _, tasks = data[p]
    u = np.zeros_like(grid_t)
    for s, st, t0, t1, c in tasks:
        u[(grid_t >= t0) & (grid_t < t1)] += c
    col, ls = styles[p]
    ax.plot(grid_t, u, color=col, lw=1.2, ls=ls, label=name)
ax.axhline(24, color=INK2, lw=0.8, ls=":")
ax.text(tmax * 0.995, 24.6, "capacity (24 threads)", ha="right", va="bottom", fontsize=7, color=INK2)
ax.set_xlim(0, tmax * 1.01); ax.set_ylim(0, 28)
ax.set_xlabel("Time since first arrival (h)"); ax.set_ylabel("Threads reserved")
ax.set_title("(d) Reserved CPU threads", loc="left", fontsize=8.5, color=INK)
ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
ax.legend(loc="lower left", frameon=False, ncol=3, fontsize=7.5)
handles = [Patch(color=COL[g], label=g) for g in COL if g != "Merge and report"] + \
          [Line2D([0], [0], marker="|", color=INK, lw=0, markersize=6, label="Sample arrival"),
           Line2D([0], [0], marker="<", color=COL["FreeBayes (revealed)"], lw=0, markersize=5, label="Sample with second caller")]
fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, 0.995))
fig.subplots_adjust(top=0.925, bottom=0.06, left=0.09, right=0.965)
fig.savefig(out + ".pdf"); fig.savefig(out + ".png", dpi=200)
