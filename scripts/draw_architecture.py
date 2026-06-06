"""Draw the full GNN + RL + LLM framework architecture.

Renders a single high-resolution PNG that visualises the three pillars,
their data flow, and the four losses used for joint training. Run from
the repo root:

    python scripts/draw_architecture.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "scripts" / "presentation_figures" / "full_architecture.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# Colour palette
# ------------------------------------------------------------------
C_GNN   = "#1F77B4"   # blue
C_RL    = "#EB811B"   # orange
C_LLM   = "#2CA02C"   # green
C_IN    = "#555555"   # dark grey (inputs)
C_OUT   = "#2B3A55"   # dark slate (output)
C_LOSS  = "#D62728"   # red (losses)
C_EDGE  = "#8C8C8C"
BG_GNN  = "#E3EEF8"
BG_RL   = "#FDECD7"
BG_LLM  = "#E4F4E2"
BG_IN   = "#F3F3F3"

fig, ax = plt.subplots(figsize=(18, 12))
ax.set_xlim(0, 20)
ax.set_ylim(0, 14)
ax.axis("off")

# ------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------
def draw_box(x, y, w, h, text, *, face, edge, fontsize=11, bold=False,
             text_color="black", sub_text=None, sub_fontsize=9,
             header=None, header_color=None):
    patch = FancyBboxPatch((x, y), w, h,
                           boxstyle="round,pad=0.04,rounding_size=0.12",
                           linewidth=1.6, edgecolor=edge, facecolor=face)
    ax.add_patch(patch)
    ty = y + h * 0.55 if sub_text else y + h * 0.5
    if header is not None:
        ax.text(x + w / 2, y + h - 0.25, header,
                ha="center", va="top", fontsize=fontsize - 1,
                fontweight="bold", color=header_color or edge)
        ty = y + h * 0.42
    ax.text(x + w / 2, ty, text, ha="center", va="center",
            fontsize=fontsize,
            fontweight=("bold" if bold else "normal"),
            color=text_color)
    if sub_text:
        ax.text(x + w / 2, y + h * 0.2, sub_text,
                ha="center", va="center",
                fontsize=sub_fontsize, color="#444444", style="italic")
    return patch

def draw_arrow(x1, y1, x2, y2, *, color=C_EDGE, lw=1.6, style="->",
               label=None, label_color=None, label_fs=9, rad=0.0,
               dashed=False, offset=(0.1, 0.1)):
    cs = f"arc3,rad={rad}"
    ls = "--" if dashed else "-"
    ar = FancyArrowPatch((x1, y1), (x2, y2),
                         arrowstyle=style, mutation_scale=14,
                         linewidth=lw, color=color,
                         connectionstyle=cs, linestyle=ls)
    ax.add_patch(ar)
    if label:
        mx, my = (x1 + x2) / 2 + offset[0], (y1 + y2) / 2 + offset[1]
        ax.text(mx, my, label, ha="center", va="center",
                fontsize=label_fs, color=label_color or color,
                fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="none",
                          boxstyle="round,pad=0.15"))

def pillar_backdrop(x, y, w, h, face, title, title_color):
    """Translucent backdrop that groups a pillar's components."""
    rect = Rectangle((x, y), w, h, linewidth=0,
                     facecolor=face, alpha=0.55, zorder=0)
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h + 0.15, title,
            ha="center", va="bottom", fontsize=14,
            fontweight="bold", color=title_color)

# ==================================================================
# TITLE
# ==================================================================
ax.text(10, 13.55, "Unified GNN + RL + LLM Framework — Full Architecture",
        ha="center", va="center", fontsize=18, fontweight="bold",
        color="#222222")
ax.text(10, 13.15,
        "data flows bottom → top  ·  green = LLM pillar  ·  orange = RL pillar  ·  blue = GNN pillar  ·  red = training losses",
        ha="center", va="center", fontsize=10, style="italic", color="#666666")

# ==================================================================
# RAW INPUTS (bottom row)
# ==================================================================
draw_box(0.6, 0.4, 3.2, 0.9,
         "Multi-relation\nadjacency",
         face=BG_IN, edge=C_IN, fontsize=10,
         sub_text="UPU/USU/UVU  or  RUR/RTR/RSR", sub_fontsize=8)
draw_box(4.3, 0.4, 3.2, 0.9,
         "Node features  $x_v$",
         face=BG_IN, edge=C_IN, fontsize=10,
         sub_text="25 (Amazon) / 32 (YelpChi)", sub_fontsize=8)
draw_box(8.0, 0.4, 3.2, 0.9,
         "26 graph statistics",
         face=BG_IN, edge=C_IN, fontsize=10,
         sub_text="degree, overlap, ego-density, …", sub_fontsize=8)
draw_box(11.7, 0.4, 3.2, 0.9,
         "Review text",
         face=BG_IN, edge=C_IN, fontsize=10,
         sub_text="YelpChi only", sub_fontsize=8)

# ==================================================================
# LLM PILLAR (right)
# ==================================================================
pillar_backdrop(7.7, 2.0, 7.5, 2.1, BG_LLM, "LLM PILLAR  (offline, precomputed)", C_LLM)

draw_box(8.0, 2.15, 3.2, 1.7,
         "Part 1 reasoning\n$t^{(1)}_v \\in \\mathbb{R}^{6}$",
         face="white", edge=C_LLM, fontsize=10, bold=True,
         header="Claude Haiku 4.5", header_color=C_LLM,
         sub_text="from 26 graph stats", sub_fontsize=8)

draw_box(11.7, 2.15, 3.2, 1.7,
         "Part 2 text risk\n$t^{(2)}_v \\in \\mathbb{R}^{6}$",
         face="white", edge=C_LLM, fontsize=10, bold=True,
         header="Claude Haiku 4.5", header_color=C_LLM,
         sub_text="from review text", sub_fontsize=8)

# arrows: inputs → LLM pillar
draw_arrow(9.6, 1.30, 9.6, 2.15, color=C_IN, lw=1.2)
draw_arrow(13.3, 1.30, 13.3, 2.15, color=C_IN, lw=1.2)

# ==================================================================
# TEXT-FEATURE GATE (between LLM Part 2 and GNN feature path)
# ==================================================================
draw_box(4.1, 4.8, 3.4, 1.2,
         "$x'_v = x_v + \\sigma(g) \\odot W_{\\text{text}}\\, t^{(2)}_v$",
         face="#FFF5EC", edge=C_LLM, fontsize=11, bold=True,
         header="TextFeatureGate", header_color=C_LLM,
         sub_text="gate $g$ init $= -2$ so $\\sigma(g) \\approx 0.12$",
         sub_fontsize=8)

# x_v → gate
draw_arrow(5.9, 1.30, 5.9, 4.8, color=C_IN, lw=1.2, rad=0.0)
# t(2) → gate
draw_arrow(13.3, 3.85, 7.5, 5.4, color=C_LLM, lw=1.5, rad=-0.25,
           label="$t^{(2)}_v$  (6-dim)", label_fs=9, offset=(-1.2, 0.4))

# ==================================================================
# RL PILLAR (left-middle column)
# ==================================================================
pillar_backdrop(0.6, 4.6, 3.0, 6.9, BG_RL, "RL PILLAR  (CAPN)", C_RL)

# State Constructor
draw_box(0.8, 5.0, 2.6, 1.4,
         "$\\mathbf{s}_v^{r} = [\\,x_v,\\,\\mathrm{conf},\\,\\hat{d},\\,\\bar{\\delta},\\,J,\\,\\sigma,\\,t^{(1)}_v\\,]$",
         face="white", edge=C_RL, fontsize=9, bold=True,
         header="State constructor", header_color=C_RL,
         sub_text="$D + 5 + 6 = D + 11$  dims", sub_fontsize=8)

# Policy Network
draw_box(0.8, 6.7, 2.6, 1.4,
         "$\\pi_\\theta(\\tau_r \\mid \\mathbf{s}_v^{r})$\nBeta distribution",
         face="white", edge=C_RL, fontsize=10, bold=True,
         header="Policy network (actor)", header_color=C_RL,
         sub_text="per-node $p_r = \\tau_r$", sub_fontsize=8)

# Value Network (critic)
draw_box(0.8, 8.4, 2.6, 1.2,
         "$V_\\phi(\\mathbf{s}_v^{r}) \\in \\mathbb{R}$",
         face="white", edge=C_RL, fontsize=11, bold=True,
         header="Value network (critic)", header_color=C_RL,
         sub_text="3-layer MLP  ·  baseline", sub_fontsize=8)

# Reward block
draw_box(0.8, 9.9, 2.6, 1.3,
         "$R = w_1\\Delta d + w_2\\, \\mathrm{acc} + w_3\\, \\mathrm{stab}$",
         face="white", edge=C_RL, fontsize=10, bold=True,
         header="Shaped reward", header_color=C_RL,
         sub_text="$w_1{=}0.5,\\ w_2{=}0.3,\\ w_3{=}0.2$", sub_fontsize=8)

# Arrows inside RL pillar
draw_arrow(2.1, 6.4, 2.1, 6.7, color=C_RL, lw=1.2)        # state → policy
draw_arrow(2.1, 6.4, 2.1, 8.4, color=C_RL, lw=1.0, rad=-0.3)  # state → critic (curved)
# LLM Part 1 → State Constructor
draw_arrow(9.6, 2.15, 3.4, 5.6, color=C_LLM, lw=1.5, rad=0.20,
           label="$t^{(1)}_v$ into state", label_fs=9, offset=(-0.5, 0.5))

# Node features → state constructor (x_v part)
draw_arrow(4.3, 0.85, 1.3, 5.0, color=C_IN, lw=1.0, rad=-0.30)

# ==================================================================
# GNN PILLAR (right-middle column)
# ==================================================================
pillar_backdrop(8.1, 4.6, 7.4, 6.9, BG_GNN, "GNN PILLAR  (CARE-GNN backbone)", C_GNN)

# Label Predictor
draw_box(8.4, 5.0, 3.2, 1.4,
         "$S(v) = \\mathrm{MLP}(x'_v) \\in \\mathbb{R}^{2}$",
         face="white", edge=C_GNN, fontsize=11, bold=True,
         header="Enhanced label predictor", header_color=C_GNN,
         sub_text="produces confidence + distance", sub_fontsize=8)

# Neighbour filter
draw_box(12.0, 5.0, 3.2, 1.4,
         "top-$p_r$ nbrs by\n$D(v,v') = |S(v)[0]-S(v')[0]|$",
         face="white", edge=C_GNN, fontsize=10, bold=True,
         header="Neighbour filter", header_color=C_GNN,
         sub_text="per relation $r$", sub_fontsize=8)

# Intra/Inter aggregation
draw_box(8.4, 6.9, 6.8, 1.6,
         "$h_v = \\mathrm{ReLU}\\!\\left(W\\,x'_v + \\sum_r p_r\\, W\\, h^{(r)}_{\\mathcal{N}(v)}\\right)$",
         face="white", edge=C_GNN, fontsize=11, bold=True,
         header="Multi-relation aggregation (intra + inter)", header_color=C_GNN,
         sub_text="mean-pool within relation,  weighted sum across relations",
         sub_fontsize=8)

# Classifier
draw_box(9.7, 9.0, 4.2, 1.4,
         "$\\hat{y}_v = \\mathrm{softmax}(W_{\\text{cls}}\\, h_v)$",
         face="white", edge=C_GNN, fontsize=11, bold=True,
         header="Binary classifier", header_color=C_GNN,
         sub_text="fraud probability", sub_fontsize=8)

# Gated features → label predictor + aggregation
draw_arrow(5.8, 6.0, 8.4, 5.6, color=C_LLM, lw=1.5, rad=-0.15,
           label="gated  $x'_v$", label_fs=9, offset=(-0.1, 0.45))

# Label predictor → filter (distance)
draw_arrow(11.6, 5.7, 12.0, 5.7, color=C_GNN, lw=1.5)

# Filter → aggregation
draw_arrow(13.6, 6.4, 13.6, 6.9, color=C_GNN, lw=1.5)
# Label predictor → aggregation (direct)
draw_arrow(10.0, 6.4, 10.0, 6.9, color=C_GNN, lw=1.5)

# RL thresholds → aggregation (ORANGE, the key cross-pillar arrow)
draw_arrow(3.4, 7.4, 8.4, 7.6, color=C_RL, lw=2.0, rad=-0.10,
           label="per-node thresholds  $p_r$", label_fs=10,
           label_color=C_RL, offset=(0.3, 0.5))

# aggregation → classifier
draw_arrow(11.8, 8.5, 11.8, 9.0, color=C_GNN, lw=1.5)

# ==================================================================
# OUTPUT + LOSSES (top row)
# ==================================================================
draw_box(8.4, 11.15, 6.8, 0.9,
         "Fraud probability  $\\hat{y}_v$",
         face=C_OUT, edge=C_OUT, fontsize=14, bold=True, text_color="white")
draw_arrow(11.8, 10.4, 11.8, 11.15, color=C_OUT, lw=2.0)

# Losses (right column, vertical stack)
loss_x = 16.0
for i, (name, expr) in enumerate([
    ("$\\mathcal{L}_{\\text{GNN}}$", "CE($\\hat{y}_v$, $y_v$)"),
    ("$\\mathcal{L}_{\\text{sim}}$", "CE($S(v)$, $y_v$)"),
    ("$\\mathcal{L}_{\\text{policy}}$", "$-\\hat{A}\\log\\pi_\\theta$"),
    ("$\\mathcal{L}_{\\text{critic}}$", "$(V_\\phi - R)^2$"),
]):
    y = 10.5 - i * 1.1
    draw_box(loss_x, y, 3.3, 0.85, f"{name}  =  {expr}",
             face="#FEEAEA", edge=C_LOSS, fontsize=10, bold=True, text_color=C_LOSS)

# Unified loss at the top
draw_box(loss_x, 11.55, 3.3, 0.7,
         "$\\mathcal{L} = \\mathcal{L}_{\\text{GNN}} + \\lambda_1 \\mathcal{L}_{\\text{sim}} + \\lambda_\\pi(t)\\mathcal{L}_{\\text{policy}} + \\lambda_c\\mathcal{L}_{\\text{critic}}$",
         face=C_LOSS, edge=C_LOSS, fontsize=9, bold=True, text_color="white")

# gradient arrows from losses to components (dashed)
draw_arrow(loss_x, 10.92, 14.5, 9.6,  color=C_LOSS, lw=1.1, dashed=True)  # L_GNN → classifier
draw_arrow(loss_x, 9.82,  11.6, 5.7,  color=C_LOSS, lw=1.1, dashed=True)  # L_sim → label predictor
draw_arrow(loss_x, 8.72,  3.4,  7.4,  color=C_LOSS, lw=1.1, dashed=True, rad=0.25)  # L_policy → policy
draw_arrow(loss_x, 7.62,  3.4,  9.0,  color=C_LOSS, lw=1.1, dashed=True, rad=0.15)  # L_critic → critic

# Legend / notes at very top
ax.text(loss_x + 1.65, 12.55,
        "loss components  ▲",
        ha="center", va="bottom", fontsize=9, color=C_LOSS, style="italic")

# ==================================================================
# FOOTER — training schedule
# ==================================================================
ax.text(10, 0.15,
        "Training schedule:  epochs 0–5 GNN warmup ($\\lambda_\\pi{=}0$)  ·  5–10 policy ramp ($\\lambda_\\pi: 0 \\to 0.3$)  ·  10–31 full actor–critic",
        ha="center", va="bottom", fontsize=10, style="italic", color="#444444")

# Save
fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Wrote: {OUT}")
