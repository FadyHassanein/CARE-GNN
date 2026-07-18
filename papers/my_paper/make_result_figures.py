"""Generate the Results-section figures from the saved experiment JSONs:
  results_ablation.png     -- per-seed slope plot of the held-out ablation cube
  results_doseresponse.png -- text6 delta vs backbone strength (8 backbones)

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

C_GNN, C_RL, C_LLM, C_OUT = "#1F77B4", "#EB811B", "#2CA02C", "#2B3A55"


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

configs = ["R1\nstructural", "R2$'$\n+adaptive", "R3\n+semantic", "R4$'$\nfull"]
data = np.array([[r1[s], r2[s], r3[s], r4[s]] for s in seeds])  # (20, 4)

fig, ax = plt.subplots(figsize=(7.2, 4.6))
x = np.arange(4)
for row in data:
    ax.plot(x, row, color="#B0B8C4", lw=0.9, alpha=0.75, zorder=1)
    ax.scatter(x, row, s=10, color="#B0B8C4", alpha=0.75, zorder=1)
mean = data.mean(0)
ax.plot(x, mean, color=C_OUT, lw=3.0, marker="o", ms=8, zorder=3,
        label="mean over 20 held-out seeds")
for xi, m, c in zip(x, mean, [C_GNN, C_RL, C_LLM, C_OUT]):
    ax.scatter([xi], [m], s=90, color=c, zorder=4)

# annotate the consecutive paired deltas, computed from the data itself
ann = []
for i in range(3):
    d = data[:, i + 1] - data[:, i]
    ann.append((f"$+{d.mean() * 100:.2f}$ pp\n${int((d > 0).sum())}/20$",
                i + 0.5))
for (txt, xp), ym in zip(ann, (mean[:-1] + mean[1:]) / 2):
    ax.annotate(txt, xy=(xp, ym), fontsize=8.5, ha="center", va="center",
                color="#444444",
                bbox=dict(facecolor="white", edgecolor="#CCCCCC",
                          boxstyle="round,pad=0.25"))
ax.text(3.0, data[:, 0].min() + 0.0005,
        "full vs. base: $+2.70$ pp\n$20/20$, $p<10^{-16}$",
        ha="center", fontsize=9, color=C_OUT, fontweight="bold")

ax.set_xticks(x); ax.set_xticklabels(configs, fontsize=10)
ax.set_ylabel("test AUC (val-selected checkpoint)", fontsize=10)
ax.tick_params(axis="y", labelsize=9)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(loc="upper left", fontsize=9, frameon=False)
fig.tight_layout()
fig.savefig(OUT / "results_ablation.png", dpi=220, bbox_inches="tight",
            facecolor="white")
plt.close(fig)

# ====================================================================
# FIG B — dose-response: text6 delta vs backbone strength
# ====================================================================
# (backbone, raw AUC, text6 delta pp, significant?) — numbers from
# tabular_baselines / xgb_graph_table / bwgnn_table / graphconsis_table /
# yelp_corrected_n5_summary; identical frozen 25/15/60 split.
points = [
    ("LR",           0.7694, 2.29, True),
    ("MLP",          0.8205, 0.78, False),
    ("HistGBM",      0.8620, 1.18, True),
    ("XGBoost",      0.8541, 2.42, True),
    ("XGB-Graph",    0.9106, 0.70, True),
    ("CARE-GNN",     0.7648, 1.69, True),
    ("BWGNN-Homo",   0.8252, 1.66, True),
    ("BWGNN-Hetero", 0.8879, 0.62, True),
    ("GraphConsis",  0.7840, 4.16, False),
]

fig, ax = plt.subplots(figsize=(7.2, 4.6))
xs = np.array([p[1] for p in points]); ys = np.array([p[2] for p in points])
for name, xv, yv, sig in points:
    ax.scatter([xv], [yv], s=95, zorder=3,
               facecolor=C_LLM if sig else "white",
               edgecolor=C_LLM, linewidth=1.8)
    dy = 0.22 if name != "XGBoost" else -0.38
    ha = "center"
    ax.annotate(name, (xv, yv), xytext=(0, 14 if dy > 0 else -20),
                textcoords="offset points", ha=ha, fontsize=9)

# least-squares trend line
b, a = np.polyfit(xs, ys, 1)
xx = np.linspace(xs.min() - 0.01, xs.max() + 0.01, 50)
ax.plot(xx, a + b * xx, color="#888888", lw=1.6, ls="--", zorder=2)
ax.text(0.895, 3.4, f"trend: {b / 10:.1f} pp per\n$+0.1$ AUC of backbone",
        fontsize=9, color="#666666", ha="center")

ax.scatter([], [], s=95, facecolor=C_LLM, edgecolor=C_LLM, label="significant")
ax.scatter([], [], s=95, facecolor="white", edgecolor=C_LLM,
           label="not significant")
ax.legend(loc="upper right", fontsize=9, frameon=False)
ax.set_xlabel("backbone AUC without the semantic signal", fontsize=10)
ax.set_ylabel("text6 gain (AUC pp)", fontsize=10)
ax.tick_params(labelsize=9)
ax.spines[["top", "right"]].set_visible(False)
ax.axhline(0, color="#DDDDDD", lw=1, zorder=0)
fig.tight_layout()
fig.savefig(OUT / "results_doseresponse.png", dpi=220, bbox_inches="tight",
            facecolor="white")
plt.close(fig)
print(f"Wrote: {OUT/'results_ablation.png'} and {OUT/'results_doseresponse.png'}")
