"""Generate the paper's architecture figures (final framing):
  architecture.png       -- Fig 1, 3-pillar overview + held-out-confirmed deltas
  architecture_capn.png  -- Fig 2, CAPN mechanism (clean config: neutral init,
                            z_v channel marked as measured-null)

    python papers/my_paper/make_architecture.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

DIR = Path(__file__).resolve().parent / "extracted"
C_GNN, C_RL, C_LLM = "#1F77B4", "#EB811B", "#2CA02C"
BG_GNN, BG_RL, BG_LLM = "#E3EEF8", "#FDECD7", "#E4F4E2"
C_IN, C_OUT, C_NULL = "#555555", "#2B3A55", "#999999"


def box(ax, x, y, w, h, text, *, edge, face="white", fs=11, bold=False,
        header=None, tc="black", ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.03,rounding_size=0.10",
                 lw=1.7, edgecolor=edge, facecolor=face, linestyle=ls))
    ty = y + h / 2
    if header:
        ax.text(x + w / 2, y + h - 0.16, header, ha="center", va="top",
                fontsize=fs - 1.5, fontweight="bold", color=edge)
        ty = y + h * 0.40
    ax.text(x + w / 2, ty, text, ha="center", va="center", fontsize=fs,
            fontweight=("bold" if bold else "normal"), color=tc)


def arrow(ax, x1, y1, x2, y2, *, color, lw=2.0, rad=0.0, label=None,
          off=(0, 0.28), lc=None, fs=9, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                 mutation_scale=18, lw=lw, color=color, linestyle=ls,
                 connectionstyle=f"arc3,rad={rad}"))
    if label:
        ax.text((x1 + x2) / 2 + off[0], (y1 + y2) / 2 + off[1], label,
                ha="center", va="center", fontsize=fs, color=lc or color,
                fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="none",
                          boxstyle="round,pad=0.12"))


# ====================================================================
# FIG 1 — high-level overview, annotated with held-out-confirmed deltas
# ====================================================================
fig, ax = plt.subplots(figsize=(10.5, 6.2))
ax.set_xlim(0, 11); ax.set_ylim(-0.5, 6.6); ax.axis("off")

# main horizontal pipeline
box(ax, 0.35, 2.45, 2.45, 1.1, "Multi-relation\nreview graph", edge=C_IN, fs=11)
box(ax, 3.85, 2.30, 3.35, 1.4, "filter\n$+$ multi-relation aggregation",
    edge=C_GNN, fs=11, header="Structural pillar  (CARE-GNN)", face=BG_GNN)
box(ax, 8.25, 2.45, 2.4, 1.1, "Fraud probability\n$\\hat{y}_v$",
    edge=C_OUT, face=C_OUT, tc="white", bold=True, fs=11)
arrow(ax, 2.80, 3.0, 3.85, 3.0, color=C_IN)
arrow(ax, 7.20, 3.0, 8.25, 3.0, color=C_OUT)

# semantic pillar on top -> gated residual into node features (the live channel)
box(ax, 3.30, 4.95, 4.45, 1.25,
    "six fraud-specific text scores\n(LLM \\emph{or} zero-cost heuristic scorer)".replace("\\emph{or}", "or"),
    edge=C_LLM, face=BG_LLM, fs=10.5,
    header="Semantic pillar  (cached, scorer-agnostic)")
arrow(ax, 5.525, 4.95, 5.525, 3.70, color=C_LLM, lw=2.2,
      label="gated residual $\\to$ node features   ($+1.97$ pp)", off=(0, 0.0),
      fs=9)

# adaptive pillar on bottom -> per-node thresholds (clean, neutral init)
box(ax, 3.30, 0.35, 4.45, 1.25, "actor-critic filtering policy\n(neutral init)",
    edge=C_RL, face=BG_RL, fs=10.5, header="Adaptive pillar  (CAPN)")
arrow(ax, 5.525, 1.60, 5.525, 2.30, color=C_RL, lw=2.2,
      label="per-node, per-relation thresholds   ($+0.58$ pp)", off=(0, 0.0),
      fs=9)

# ablated coupling: z_v -> policy state (measured ~0), dashed, routed left of
# the backbone but right of the input box (rad kept small to avoid overlap)
arrow(ax, 3.30, 5.35, 3.30, 1.25, color=C_NULL, lw=1.4, rad=0.10, ls="--")
ax.text(2.25, 4.42, "scores $\\to$ policy state\n(measured $\\approx 0$)",
        ha="center", va="center", fontsize=8, color=C_NULL, style="italic",
        bbox=dict(facecolor="white", edgecolor="none", boxstyle="round,pad=0.1"))

# held-out verdict strip
ax.text(5.5, -0.28,
        "held-out confirmed (20 fresh seeds): semantic $+1.97$ pp ($20/20$) · "
        "adaptive $+0.58$ pp ($20/20$) · compose $+0.74$ pp ($19/20$) · "
        "full framework $+2.70$ pp ($p<10^{-16}$)",
        ha="center", va="center", fontsize=9, color=C_OUT, fontweight="bold")
ax.text(5.5, 6.45,
        "ablated couplings (dashed / removed): LLM-prior init suppresses the "
        "policy ($-0.21$ pp); scores-in-state $\\approx 0$",
        ha="center", va="center", fontsize=8.5, color=C_NULL, style="italic")

fig.savefig(DIR / "architecture.png", dpi=220, bbox_inches="tight",
            facecolor="white")
plt.close(fig)

# ====================================================================
# FIG 2 — CAPN mechanism breakdown (clean configuration)
# ====================================================================
fig, ax = plt.subplots(figsize=(11.5, 5.4))
ax.set_xlim(0, 13); ax.set_ylim(-0.3, 6); ax.axis("off")

# inputs to the state
box(ax, 0.3, 3.05, 2.7, 1.7,
    "node features\nlabel confidence\nstructural stats", edge=C_IN, fs=10,
    header="per-node inputs")
box(ax, 0.3, 1.15, 2.7, 1.0, "LLM scores $z_v$\n(measured $\\approx 0$)",
    edge=C_NULL, face="white", fs=9, ls="--", tc=C_NULL)

# state
box(ax, 3.7, 2.35, 2.0, 1.5, "state\n$s_{v,r}$", edge=C_RL, face=BG_RL,
    bold=True, fs=12, header="concat")
arrow(ax, 3.0, 3.6, 3.7, 3.35, color=C_IN)
arrow(ax, 3.0, 1.65, 3.7, 2.7, color=C_NULL, rad=0.12, ls="--", lw=1.4)

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
ax.text(6.5, -0.15,
        "Clean configuration: policy initialised neutrally --- LLM-prior "
        "initialisation ablated (it suppresses the policy, $-0.21$ pp); "
        "clean policy $+0.58$ pp on 20/20 held-out seeds",
        ha="center", va="center", fontsize=8.5, color=C_RL, style="italic")

fig.savefig(DIR / "architecture_capn.png", dpi=220, bbox_inches="tight",
            facecolor="white")
plt.close(fig)
print(f"Wrote: {DIR/'architecture.png'} and {DIR/'architecture_capn.png'}")
