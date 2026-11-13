"""Schematic of the event-driven decision process of GM-HRL (Section 4.2)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d3"
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
plt.rcParams.update({"font.size": 8.5, "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix"})
fig, ax = plt.subplots(figsize=(7.2, 3.4))
ax.set_xlim(-0.9, 10.4); ax.set_ylim(-0.35, 3.95); ax.axis("off")

Y_EV, Y_SUB, Y_META = 2.75, 1.65, 0.55
for y, lab in ((Y_EV, "Events"), (Y_SUB, "Sub-Controller"), (Y_META, "Meta-Controller")):
    ax.plot([0, 10], [y, y], color=GRID, lw=1.0, zorder=0)
    ax.text(-0.15, y, lab, ha="right", va="center", fontsize=8, color=INK, weight="bold")
ax.annotate("", xy=(10.25, -0.12), xytext=(0, -0.12), arrowprops=dict(arrowstyle="->", color=INK2, lw=0.9))
ax.text(10.25, -0.27, "time", ha="right", va="top", fontsize=7.5, color=INK2)

# events: (t, kind)
ev = [(0.0, "arr"), (0.9, "arr"), (1.6, "done"), (2.5, "arr"), (3.3, "done"), (4.3, "rrd"),
      (5.4, "arr"), (6.2, "done"), (7.4, "done"), (8.6, "arr"), (9.3, "done")]
mk = {"arr": ("v", INK, "sample arrival"), "done": ("o", BLUE, "task completion"),
      "rrd": ("*", ORANGE, "completion with RRD event")}
seen = set()
for t, k in ev:
    m, c, lab = mk[k]
    ax.scatter(t, Y_EV, marker=m, s=70 if k != "rrd" else 170, color=c, zorder=3,
               edgecolor="white", linewidth=0.6, label=None if k in seen else lab)
    seen.add(k)
ax.annotate("revealed tasks and edges inserted into $\\mathcal{G}_t$;\nnode features updated",
            xy=(4.38, Y_EV + 0.08), xytext=(4.65, Y_EV + 0.42), ha="left", va="center", fontsize=7, color=INK2,
            arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))

# decision epochs: at events where a task is dispatchable; number of decisions at that time
epochs = [(0.0, 1), (0.9, 1), (1.6, 2), (2.5, 1), (3.3, 1), (4.3, 2), (6.2, 1), (7.4, 2), (8.6, 1), (9.3, 1)]
for t, n in epochs:
    for j in range(n):
        ax.plot([t, t], [Y_EV - 0.12, Y_SUB + 0.12], color=GRID, lw=0.6, ls=":", zorder=0)
        ax.scatter(t, Y_SUB + 0.17 * j, marker="s", s=30, color=AQUA, zorder=3, edgecolor="white", linewidth=0.5)
ax.plot([5.4, 5.4], [Y_EV - 0.12, Y_SUB + 0.12], color=GRID, lw=0.6, ls=":", zorder=0)
ax.scatter(5.4, Y_SUB, marker="s", s=26, facecolor="white", edgecolor=AQUA, linewidth=0.9, zorder=3)
ax.text(5.4, Y_SUB - 0.2, "no dispatchable task:\nno decision", ha="center", va="top", fontsize=7, color=INK2)
ax.annotate("two successive decisions\nat the same time", xy=(7.4, Y_SUB + 0.2), xytext=(8.15, Y_SUB + 0.62),
            fontsize=6.8, color=INK2, ha="left", va="center",
            arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))

# meta: first epoch, then first epoch after every Delta t_H
dt = 3.0
meta_t = [0.0]
nxt = dt
for t, _ in epochs:
    if t >= nxt:
        meta_t.append(t); nxt = t + dt
bounds = meta_t + [10.0]
for i, t in enumerate(meta_t):
    ax.add_patch(FancyBboxPatch((t + 0.03, Y_META - 0.17), bounds[i + 1] - t - 0.06, 0.34,
                                boxstyle="round,pad=0,rounding_size=0.05", facecolor="#efeefb", edgecolor="none", zorder=1))
    ax.scatter(t, Y_META, marker="D", s=34, color=VIOLET, zorder=3, edgecolor="white", linewidth=0.5)
    ax.text((t + bounds[i + 1]) / 2, Y_META, f"goal $\\mathbf{{g}}^{{({i+1})}}$ held", ha="center", va="center",
            fontsize=7, color=INK)
    ax.plot([t, t], [Y_META + 0.17, Y_SUB - 0.12], color=VIOLET, lw=0.6, ls=":", zorder=0)
for i in range(len(meta_t) - 1):
    a, b = meta_t[i], meta_t[i] + dt
    ax.annotate("", xy=(b, Y_META - 0.27), xytext=(a, Y_META - 0.27),
                arrowprops=dict(arrowstyle="<->", color=INK2, lw=0.6))
    ax.text((a + b) / 2, Y_META - 0.3, "$\\Delta t_H$", ha="center", va="top", fontsize=7, color=INK2)

h = [plt.Line2D([0], [0], marker="v", color=INK, lw=0, markersize=6, label="sample arrival"),
     plt.Line2D([0], [0], marker="o", color=BLUE, lw=0, markersize=6, label="task completion"),
     plt.Line2D([0], [0], marker="*", color=ORANGE, lw=0, markersize=10, label="completion with RRD event"),
     plt.Line2D([0], [0], marker="s", color=AQUA, lw=0, markersize=5.5, label="dispatch decision (task or WAIT)"),
     plt.Line2D([0], [0], marker="D", color=VIOLET, lw=0, markersize=5.5, label="goal written into $\\mathcal{G}_t$")]
ax.legend(handles=h, loc="upper center", ncol=3, frameon=False, fontsize=7.2, bbox_to_anchor=(0.5, 1.07))
fig.subplots_adjust(left=0.13, right=0.99, top=0.88, bottom=0.02)
fig.savefig("fig_timeline.pdf"); fig.savefig("fig_timeline.png", dpi=200)
print(meta_t)
