"""Generate the paper's figures:
  architecture.png       -- Fig 1, clean high-level 3-pillar overview
  architecture_capn.png  -- Fig 2, CAPN mechanism breakdown

    python papers/my_paper/make_architecture.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

DIR = Path(__file__).resolve().parent / "extracted"
C_GNN, C_RL, C_LLM = "#1F77B4", "#EB811B", "#2CA02C"
BG_GNN, BG_RL, BG_LLM = "#E3EEF8", "#FDECD7", "#E4F4E2"
C_IN, C_OUT = "#555555", "#2B3A55"


def box(ax, x, y, w, h, text, *, edge, face="white", fs=11, bold=False,
        header=None, tc="black"):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.03,rounding_size=0.10",
                 lw=1.7, edgecolor=edge, facecolor=face))
    ty = y + h / 2
    if header:
        ax.text(x + w / 2, y + h - 0.16, header, ha="center", va="top",
                fontsize=fs - 1.5, fontweight="bold", color=edge)
        ty = y + h * 0.40
    ax.text(x + w / 2, ty, text, ha="center", va="center", fontsize=fs,
            fontweight=("bold" if bold else "normal"), color=tc)


def arrow(ax, x1, y1, x2, y2, *, color, lw=2.0, rad=0.0, label=None,
          off=(0, 0.28), lc=None, fs=9):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                 mutation_scale=18, lw=lw, color=color,
                 connectionstyle=f"arc3,rad={rad}"))
    if label:
        ax.text((x1 + x2) / 2 + off[0], (y1 + y2) / 2 + off[1], label,
                ha="center", va="center", fontsize=fs, color=lc or color,
                fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="none",
                          boxstyle="round,pad=0.12"))


# ====================================================================
# FIG 1 — high-level overview (no crossing arrows)
# ====================================================================
fig, ax = plt.subplots(figsize=(10.5, 5.6))
ax.set_xlim(0, 11); ax.set_ylim(0, 6); ax.axis("off")

# main horizontal pipeline
box(ax, 0.35, 2.45, 2.45, 1.1, "Multi-relation\nreview graph", edge=C_IN, fs=11)
box(ax, 3.85, 2.30, 3.35, 1.4, "filter\n$+$ multi-relation aggregation",
    edge=C_GNN, fs=11, header="Structural pillar  (CARE-GNN)", face=BG_GNN)
box(ax, 8.25, 2.45, 2.4, 1.1, "Fraud probability\n$\\hat{y}_v$",
    edge=C_OUT, face=C_OUT, tc="white", bold=True, fs=11)
arrow(ax, 2.80, 3.0, 3.85, 3.0, color=C_IN)
arrow(ax, 7.20, 3.0, 8.25, 3.0, color=C_OUT)

# semantic pillar on top -> straight down into backbone
box(ax, 3.55, 4.55, 3.95, 1.1,
    "structural risk $+$ textual risk scores", edge=C_LLM, face=BG_LLM, fs=10.5,
    header="Semantic pillar  (LLM, cached)")
arrow(ax, 5.525, 4.55, 5.525, 3.70, color=C_LLM, lw=2.2,
      label="scores $\\to$ policy state $+$ node features", off=(0, 0.0),
      fs=9)

# adaptive pillar on bottom -> straight up into backbone
box(ax, 3.55, 0.35, 3.95, 1.1, "actor-critic filtering policy",
    edge=C_RL, face=BG_RL, fs=10.5, header="Adaptive pillar  (CAPN)")
arrow(ax, 5.525, 1.45, 5.525, 2.30, color=C_RL, lw=2.2,
      label="per-node, per-relation thresholds", off=(0, 0.0), fs=9)

fig.savefig(DIR / "architecture.png", dpi=220, bbox_inches="tight",
            facecolor="white")
plt.close(fig)

# ====================================================================
# FIG 2 — CAPN mechanism breakdown
# ====================================================================
fig, ax = plt.subplots(figsize=(11.5, 5.2))
ax.set_xlim(0, 13); ax.set_ylim(0, 6); ax.axis("off")

# inputs to the state
box(ax, 0.3, 3.05, 2.7, 1.7,
    "node features\nlabel confidence\nstructural stats", edge=C_IN, fs=10,
    header="per-node inputs")
box(ax, 0.3, 1.15, 2.7, 1.0, "LLM risk scores $z_v$", edge=C_LLM,
    face=BG_LLM, fs=10)

# state
box(ax, 3.7, 2.35, 2.0, 1.5, "state\n$s_{v,r}$", edge=C_RL, face=BG_RL,
    bold=True, fs=12, header="concat")
arrow(ax, 3.0, 3.6, 3.7, 3.35, color=C_IN)
arrow(ax, 3.0, 1.65, 3.7, 2.7, color=C_LLM, rad=0.12,
      label="(1) into state", off=(0.15, -0.35), fs=8.5)

# policy (actor + critic)
box(ax, 6.4, 2.35, 2.7, 1.5,
    "Beta policy $\\pi_\\psi$\n$\\theta_{v,r}=\\alpha/(\\alpha{+}\\beta)$",
    edge=C_RL, bold=True, fs=11, header="actor  (critic $V_\\eta$, advantage $R{-}V$)")
arrow(ax, 5.7, 3.1, 6.4, 3.1, color=C_RL)

# the dual use of one threshold
box(ax, 9.8, 3.35, 2.9, 1.05, "keep $\\lceil\\theta_{v,r}|\\mathcal{N}_r(v)|\\rceil$\nnearest neighbours",
    edge=C_GNN, face=BG_GNN, fs=9.5, header="neighbour filter")
box(ax, 9.8, 1.55, 2.9, 1.05, "relation weight\n$\\omega_r=\\bar{\\theta}_{\\cdot,r}$",
    edge=C_GNN, face=BG_GNN, fs=9.5, header="inter-relation mixing")
arrow(ax, 9.1, 3.4, 9.8, 3.85, color=C_RL, rad=-0.12, label="$\\theta_{v,r}$",
      off=(0.0, 0.3), fs=9)
arrow(ax, 9.1, 2.8, 9.8, 2.05, color=C_RL, rad=0.12, label="mean $\\theta$",
      off=(0.0, -0.3), fs=9)

ax.text(11.25, 0.95, "one learned threshold, two uses",
        ha="center", va="center", fontsize=9, style="italic", color=C_RL)
ax.text(6.5, 0.45,
        "Trained end-to-end by actor-critic: "
        "$A_{v,r}=R-V_\\eta(s_{v,r})$,  "
        "$R=w_1\\mathrm{clip}(\\Delta\\bar d)+w_2(\\mathrm{acc}-\\overline{\\mathrm{acc}})-w_3\\overline{|\\theta-\\frac{1}{2}|}$",
        ha="center", va="center", fontsize=8.5, color="#444444")

fig.savefig(DIR / "architecture_capn.png", dpi=220, bbox_inches="tight",
            facecolor="white")
plt.close(fig)
print(f"Wrote: {DIR/'architecture.png'} and {DIR/'architecture_capn.png'}")
