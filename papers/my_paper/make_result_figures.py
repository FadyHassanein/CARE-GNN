"""Generate the Results-section figures from the saved experiment JSONs:
  results_ablation.pdf     -- per-seed slope plot of the held-out ablation cube
  results_doseresponse.pdf -- text6 delta vs backbone strength (8 backbones)

    python papers/my_paper/make_result_figures.py
(run from repo root so results/experiments/ resolves)
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "extracted"
RES = ROOT / "results" / "experiments"

# Column-native sizing: the paper is two-column A4, so a figure lands at ~8.6cm.
# Drawing at 7in and letting LaTeX shrink it put the labels at ~5pt; draw at the
# final size instead and set type in real points. Serif to match the body text.
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman", "CMU Serif", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 7,
    "axes.labelsize": 7,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.01,
})

# Palette checked for colour-vision separation (the previous green/orange pair
# was indistinguishable under protanopia).
C_GNN, C_RL, C_LLM, C_OUT = "#1B6EC2", "#B5700C", "#0A9D5E", "#8E4FA8"


def load(name):
    with open(RES / name, encoding="utf-8") as f:
        return json.load(f)


# ====================================================================
# FIG A — per-seed slope plot, n=20 held-out ablation cube
# ====================================================================
seeds = list(range(100, 120))
r1 = {r["seed"]: r["auc"] for r in load("r1_holdout20.json")["per_seed"]}
conf2 = load("rl_clean_confirmation_n20.json")
r2 = dict(zip(conf2["seeds"], conf2["r2_no_priors_auc"]))
r3 = {r["seed"]: r["auc"] for r in load("r3_holdout20.json")["per_seed"]}
r4 = dict(zip(seeds, load("r4_clean_confirmation_n20.json")["r4_auc"]))

# The ablation is a 2x2 lattice (adaptive on/off x semantic on/off), not a
# ladder: R3 drops the adaptive pillar, so R2->R3 is a swap and not a valid
# single-pillar delta. Lay the four configurations out as the lattice and label
# only its four edges, each of which isolates exactly one pillar.
data = np.array([[r1[s], r2[s], r3[s], r4[s]] for s in seeds])  # (20, 4)
mean = data.mean(0)
# x position per configuration: R1 left, R2/R3 in the middle column, R4 right
px = np.array([0.0, 1.0, 1.0, 2.0])

fig, ax = plt.subplots(figsize=(3.4, 2.5))
for row in data:  # each seed contributes both routes through the lattice
    for mid in (1, 2):
        ax.plot([px[0], px[mid], px[3]], [row[0], row[mid], row[3]],
                color="#B0B8C4", lw=0.45, alpha=0.5, zorder=1)
    ax.scatter(px, row, s=3, color="#B0B8C4", alpha=0.5, zorder=1)

EDGES = [  # (from, to, colour, pillar added)
    (0, 1, C_RL, "adaptive"), (2, 3, C_RL, "adaptive"),
    (0, 2, C_LLM, "semantic"), (1, 3, C_LLM, "semantic"),
]
for i, j, colour, _ in EDGES:
    ax.plot([px[i], px[j]], [mean[i], mean[j]], color=colour, lw=1.6,
            zorder=3, solid_capstyle="round")
    d = data[:, j] - data[:, i]
    ax.annotate(f"$+{d.mean() * 100:.2f}$ pp\n${int((d > 0).sum())}/20$",
                xy=((px[i] + px[j]) / 2, (mean[i] + mean[j]) / 2),
                fontsize=6, ha="center", va="center", color="#333333",
                bbox=dict(facecolor="white", edgecolor=colour, lw=0.8,
                          boxstyle="round,pad=0.25"), zorder=5)
for xi, m, c in zip(px, mean, [C_GNN, C_RL, C_LLM, C_OUT]):
    ax.scatter([xi], [m], s=42, color=c, zorder=4, edgecolor="white", lw=0.8)

# Each node label is pushed away from the two edge boxes adjacent to its point,
# otherwise they overlap once the figure is drawn at column width.
for lbl, xi, m, off, ha in [("R1 structural", px[0], mean[0], (2, -10), "left"),
                            ("R2 +adaptive", px[1], mean[1], (9, -8), "left"),
                            ("R3 +semantic", px[2], mean[2], (-9, 8), "right"),
                            ("R4 full", px[3], mean[3], (2, 10), "right")]:
    ax.annotate(lbl, xy=(xi, m), xytext=off, textcoords="offset points",
                ha=ha, va="center", fontsize=6.5, fontweight="bold",
                color="#333333")

ax.plot([], [], color=C_RL, lw=1.6, label="adds the adaptive pillar")
ax.plot([], [], color=C_LLM, lw=1.6, label="adds the semantic pillar")
ax.plot([], [], color="#B0B8C4", lw=0.6, label="one of 20 held-out seeds")

ax.set_xticks([]); ax.set_xlim(-0.30, 2.30)
ax.set_ylim(data.min() - 0.0012, data.max() + 0.0016)
ax.set_ylabel("test AUC (val-selected checkpoint)", fontsize=7)
ax.tick_params(axis="y", labelsize=6.5)
ax.spines[["top", "right", "bottom"]].set_visible(False)
ax.legend(loc="upper left", fontsize=6, frameon=False)
fig.tight_layout()
fig.savefig(OUT / "results_ablation.pdf", bbox_inches="tight",
            facecolor="white")
plt.close(fig)

# ====================================================================
# FIG B — dose-response: text6 delta vs backbone strength
# ====================================================================
# (backbone, raw AUC, text6 delta pp, significant?) — numbers from
# tabular_baselines / xgb_graph_table / bwgnn_table / graphconsis_table /
# yelp_corrected_n5_summary; identical frozen 25/15/60 split.
# Label offsets are per point: at column width a uniform offset collides.
# (name, raw AUC, delta pp, significant, (dx,dy) in points, horizontal align)
points = [
    ("LR",           0.7694, 2.29, True,  (0, 8),   "center"),
    ("MLP",          0.8205, 0.78, False, (0, -11), "center"),
    ("HistGBM",      0.8620, 1.18, True,  (5, -10), "left"),
    ("XGBoost",      0.8541, 2.42, True,  (0, 8),   "center"),
    ("XGB-Graph",    0.9106, 0.70, True,  (0, 8),   "center"),
    ("CARE-GNN",     0.7648, 1.69, True,  (5, -9),  "left"),
    ("BWGNN-Homo",   0.8252, 1.66, True,  (0, 8),   "center"),
    ("BWGNN-Hetero", 0.8879, 0.62, True,  (-5, -10), "right"),
    ("GraphConsis",  0.7402, 6.58, True,  (7, 0),   "left"),
]

fig, ax = plt.subplots(figsize=(3.4, 2.5))
xs = np.array([p[1] for p in points]); ys = np.array([p[2] for p in points])
for name, xv, yv, sig, off, ha in points:
    ax.scatter([xv], [yv], s=34, zorder=3,
               facecolor=C_LLM if sig else "white",
               edgecolor=C_LLM, linewidth=1.0)
    ax.annotate(name, (xv, yv), xytext=off, textcoords="offset points",
                ha=ha, va="center", fontsize=6)

# Two fits: all nine points, and without the high-leverage GraphConsis point.
# Quoting one slope would overstate what these points support -- the gain is a
# difference containing the baseline it is regressed on, and GraphConsis alone
# moves the slope from -0.8 to -2.2 pp per +0.1 AUC.
xx = np.linspace(xs.min() - 0.01, xs.max() + 0.01, 50)
b, a = np.polyfit(xs, ys, 1)
ax.plot(xx, a + b * xx, color="#888888", lw=1.0, ls="--", zorder=2)
keep = np.array([nm != "GraphConsis" for nm, *_ in points])
b2, a2 = np.polyfit(xs[keep], ys[keep], 1)
ax.plot(xx, a2 + b2 * xx, color="#BBBBBB", lw=1.0, ls=":", zorder=2)
ax.text(0.7340, 0.55, "fit: all points (dashed)\nwithout GraphConsis (dotted)",
        fontsize=6, color="#666666", ha="left")

ax.scatter([], [], s=34, facecolor=C_LLM, edgecolor=C_LLM, label="significant")
ax.scatter([], [], s=34, facecolor="white", edgecolor=C_LLM,
           label="not significant")
ax.legend(loc="upper right", fontsize=6, frameon=False, handletextpad=0.4)
ax.set_xlim(0.730, 0.934); ax.set_ylim(-0.25, 7.6)
ax.set_xlabel("backbone AUC without the semantic signal", fontsize=7)
ax.set_ylabel("text6 gain (AUC pp)", fontsize=7)
ax.tick_params(labelsize=6.5)
ax.spines[["top", "right"]].set_visible(False)
ax.axhline(0, color="#DDDDDD", lw=1, zorder=0)
fig.tight_layout()
fig.savefig(OUT / "results_doseresponse.pdf", bbox_inches="tight",
            facecolor="white")
plt.close(fig)
print(f"Wrote: {OUT/'results_ablation.pdf'} and {OUT/'results_doseresponse.pdf'}")
