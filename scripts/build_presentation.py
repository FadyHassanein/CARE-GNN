"""Build the thesis presentation as a PowerPoint (.pptx).

Generates figures with matplotlib, renders equations via mathtext, and
assembles the deck with python-pptx. Run from the repo root:

    python scripts/build_presentation.py
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import qn
from lxml import etree

# -----------------------------------------------------------------------------
# Theme
# -----------------------------------------------------------------------------
PRIMARY = RGBColor(0x2B, 0x3A, 0x55)     # dark slate blue (titles/headers)
ACCENT = RGBColor(0xEB, 0x81, 0x1B)      # orange (highlights)
POSITIVE = RGBColor(0x2C, 0xA0, 0x2C)    # green (wins)
NEGATIVE = RGBColor(0xD6, 0x27, 0x28)    # red (losses)
NEUTRAL = RGBColor(0x1F, 0x77, 0xB4)     # blue
LIGHT_BG = RGBColor(0xF7, 0xF7, 0xF7)
TEXT_MAIN = RGBColor(0x20, 0x20, 0x20)
TEXT_MUTED = RGBColor(0x55, 0x55, 0x55)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

FONT = "Calibri"

# 16:9 widescreen
SLIDE_W_IN = 13.333
SLIDE_H_IN = 7.5

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
REPO = Path(__file__).resolve().parent.parent
FIG_DIR = REPO / "scripts" / "presentation_figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT = REPO / "Presentation.pptx"


# -----------------------------------------------------------------------------
# Figure generators
# -----------------------------------------------------------------------------
def _mpl_color(rgb: RGBColor) -> str:
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def fig_progression() -> Path:
    """Horizontal progression of fraud-detection methods."""
    fig, ax = plt.subplots(figsize=(13, 3.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")

    stages = [
        ("Classical ML", "hand-crafted\nfeatures"),
        ("Deep Learning", "CNN / RNN /\nTransformer"),
        ("GNN", "relational\nstructure"),
        ("+ RL", "adaptive\nfiltering"),
        ("+ LLM", "semantic\nreasoning"),
    ]
    n = len(stages)
    box_w = 1.4
    gap = (10 - n * box_w) / (n + 1)

    for i, (title, sub) in enumerate(stages):
        x = gap + i * (box_w + gap)
        bbox = FancyBboxPatch(
            (x, 1.1), box_w, 1.4,
            boxstyle="round,pad=0.04,rounding_size=0.12",
            linewidth=2,
            edgecolor=_mpl_color(NEUTRAL),
            facecolor="#E8F1FA",
        )
        ax.add_patch(bbox)
        ax.text(x + box_w / 2, 2.05, title, ha="center", va="center",
                fontsize=16, fontweight="bold", color=_mpl_color(PRIMARY))
        ax.text(x + box_w / 2, 1.55, sub, ha="center", va="center",
                fontsize=11, color="#333333")

        if i < n - 1:
            x_next = gap + (i + 1) * (box_w + gap)
            arrow = FancyArrowPatch(
                (x + box_w, 1.8), (x_next, 1.8),
                arrowstyle="-|>", mutation_scale=18,
                linewidth=2, color=_mpl_color(NEUTRAL),
            )
            ax.add_patch(arrow)

    out = FIG_DIR / "progression.png"
    fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def fig_three_pillars() -> Path:
    """Clean three-pillar dataflow: inputs → pillars → output.

    Layered layout (bottom-up):
      1) raw inputs on the left, pass through the LLM pillar on the right
      2) LLM scores feed into the RL and GNN pillars
      3) RL produces thresholds used by GNN
      4) GNN produces the final fraud probability
    """
    fig, ax = plt.subplots(figsize=(13, 6.8))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7.2)
    ax.axis("off")

    # colours
    col_gnn = _mpl_color(NEUTRAL)
    col_rl = _mpl_color(ACCENT)
    col_llm = _mpl_color(POSITIVE)
    col_in = "#555555"
    col_out = _mpl_color(PRIMARY)

    # ---- Layer 1 (top): output box ----
    out_w, out_h = 4.8, 0.8
    out_x = (13 - out_w) / 2
    out_y = 6.0
    out_box = FancyBboxPatch((out_x, out_y), out_w, out_h,
                             boxstyle="round,pad=0.04,rounding_size=0.12",
                             linewidth=0, facecolor=col_out)
    ax.add_patch(out_box)
    ax.text(out_x + out_w / 2, out_y + out_h / 2,
            "Fraud probability  $\\hat{y}_v$",
            ha="center", va="center", fontsize=18, fontweight="bold", color="white")

    # ---- Layer 2: GNN pillar (centre) ----
    pillar_w, pillar_h = 4.8, 1.6
    gnn_x = (13 - pillar_w) / 2
    gnn_y = 3.9
    gnn_box = FancyBboxPatch((gnn_x, gnn_y), pillar_w, pillar_h,
                             boxstyle="round,pad=0.04,rounding_size=0.14",
                             linewidth=0, facecolor=col_gnn)
    ax.add_patch(gnn_box)
    ax.text(gnn_x + pillar_w / 2, gnn_y + pillar_h - 0.45,
            "GNN Pillar", ha="center", va="center",
            fontsize=17, fontweight="bold", color="white")
    ax.text(gnn_x + pillar_w / 2, gnn_y + 0.55,
            "CARE-GNN backbone\nmulti-relation neighbour aggregation",
            ha="center", va="center", fontsize=12, color="white")

    # GNN → output
    ar = FancyArrowPatch((gnn_x + pillar_w / 2, gnn_y + pillar_h),
                         (out_x + out_w / 2, out_y),
                         arrowstyle="-|>", mutation_scale=18,
                         linewidth=2, color="#666666")
    ax.add_patch(ar)

    # ---- Layer 3: RL and LLM pillars ----
    side_w, side_h = 4.0, 1.6
    rl_x = 0.6
    rl_y = 1.5
    llm_x = 13 - 0.6 - side_w
    llm_y = 1.5

    rl_box = FancyBboxPatch((rl_x, rl_y), side_w, side_h,
                            boxstyle="round,pad=0.04,rounding_size=0.14",
                            linewidth=0, facecolor=col_rl)
    ax.add_patch(rl_box)
    ax.text(rl_x + side_w / 2, rl_y + side_h - 0.4,
            "RL Pillar (CAPN)", ha="center", va="center",
            fontsize=16, fontweight="bold", color="white")
    ax.text(rl_x + side_w / 2, rl_y + 0.5,
            "actor–critic policy\nper-node thresholds",
            ha="center", va="center", fontsize=11, color="white")

    llm_box = FancyBboxPatch((llm_x, llm_y), side_w, side_h,
                             boxstyle="round,pad=0.04,rounding_size=0.14",
                             linewidth=0, facecolor=col_llm)
    ax.add_patch(llm_box)
    ax.text(llm_x + side_w / 2, llm_y + side_h - 0.4,
            "LLM Pillar", ha="center", va="center",
            fontsize=16, fontweight="bold", color="white")
    ax.text(llm_x + side_w / 2, llm_y + 0.5,
            "Claude Haiku 4.5 (offline)\nreasoning & text scores",
            ha="center", va="center", fontsize=11, color="white")

    # RL → GNN (thresholds)
    ar_rl_gnn = FancyArrowPatch(
        (rl_x + side_w, rl_y + side_h / 2),
        (gnn_x, gnn_y + 0.4),
        arrowstyle="-|>", mutation_scale=18,
        linewidth=2, color=col_rl,
        connectionstyle="arc3,rad=-0.15")
    ax.add_patch(ar_rl_gnn)
    ax.text((rl_x + side_w + gnn_x) / 2 - 0.1, (rl_y + side_h / 2 + gnn_y + 0.4) / 2 + 0.15,
            "per-node\nthresholds $p_r$",
            ha="center", va="center", fontsize=10, color=col_rl, fontweight="bold")

    # LLM → GNN (features gate)
    ar_llm_gnn = FancyArrowPatch(
        (llm_x, llm_y + side_h / 2),
        (gnn_x + pillar_w, gnn_y + 0.4),
        arrowstyle="-|>", mutation_scale=18,
        linewidth=2, color=col_llm,
        connectionstyle="arc3,rad=0.15")
    ax.add_patch(ar_llm_gnn)
    ax.text((llm_x + gnn_x + pillar_w) / 2 + 0.1, (llm_y + side_h / 2 + gnn_y + 0.4) / 2 + 0.15,
            "text scores\n→ gated features",
            ha="center", va="center", fontsize=10, color=col_llm, fontweight="bold")

    # LLM → RL (state enrichment)
    ar_llm_rl = FancyArrowPatch(
        (llm_x, llm_y + 0.3),
        (rl_x + side_w, rl_y + 0.3),
        arrowstyle="-|>", mutation_scale=16,
        linewidth=1.8, color=col_llm, linestyle="--")
    ax.add_patch(ar_llm_rl)
    ax.text((llm_x + rl_x + side_w) / 2, rl_y - 0.05,
            "6 reasoning scores → CAPN state vector",
            ha="center", va="center", fontsize=10, color=col_llm, fontweight="bold",
            style="italic")

    # ---- Layer 4 (bottom): raw inputs ----
    ax.text(rl_x + side_w / 2, 0.6,
            "node features + adjacency\n(passed through gated embedding)",
            ha="center", va="center", fontsize=10, color=col_in, style="italic")
    ax.text(llm_x + side_w / 2, 0.6,
            "26 graph statistics per node\n+ raw review text (YelpChi)",
            ha="center", va="center", fontsize=10, color=col_in, style="italic")

    ar_in_rl = FancyArrowPatch((rl_x + side_w / 2, 0.95),
                               (rl_x + side_w / 2, rl_y),
                               arrowstyle="-|>", mutation_scale=14,
                               linewidth=1.4, color="#999999")
    ax.add_patch(ar_in_rl)
    ar_in_llm = FancyArrowPatch((llm_x + side_w / 2, 0.95),
                                (llm_x + side_w / 2, llm_y),
                                arrowstyle="-|>", mutation_scale=14,
                                linewidth=1.4, color="#999999")
    ax.add_patch(ar_in_llm)

    out = FIG_DIR / "three_pillars.png"
    fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def fig_equation(latex: str, filename: str, fontsize: int = 24) -> Path:
    """Render a math expression with matplotlib's mathtext."""
    fig = plt.figure(figsize=(10, 1.4))
    fig.text(0.5, 0.5, latex, ha="center", va="center",
             fontsize=fontsize, color=_mpl_color(PRIMARY))
    plt.axis("off")
    out = FIG_DIR / filename
    fig.savefig(out, dpi=240, bbox_inches="tight", facecolor="white",
                pad_inches=0.2)
    plt.close(fig)
    return out


def fig_actor_critic_equations() -> Path:
    latex = (
        r"$\hat{A}_t = R_t - \mathrm{sg}[V_\phi(\mathbf{s}_t)]$" + "\n\n"
        r"$\mathcal{L}_{\mathrm{policy}} = -\hat{A}_t \sum_r \log \pi_\theta(\tau_r|\mathbf{s}_t^r) - \lambda_H\, H[\pi_\theta]$" + "\n\n"
        r"$\mathcal{L}_{\mathrm{critic}} = (V_\phi(\mathbf{s}_t) - R_t)^2$"
    )
    fig = plt.figure(figsize=(10, 3.5))
    fig.text(0.5, 0.5, latex, ha="center", va="center",
             fontsize=22, color=_mpl_color(PRIMARY))
    plt.axis("off")
    out = FIG_DIR / "actor_critic.png"
    fig.savefig(out, dpi=240, bbox_inches="tight", facecolor="white",
                pad_inches=0.25)
    plt.close(fig)
    return out


def fig_unified_loss() -> Path:
    latex = (r"$\mathcal{L} \;=\; "
             r"\mathcal{L}_{\mathrm{GNN}} \,+\, \lambda_1\,\mathcal{L}_{\mathrm{sim}}"
             r" \;+\; \lambda_\pi(t)\,\mathcal{L}_{\mathrm{policy}}"
             r" \;+\; \lambda_c\,\mathcal{L}_{\mathrm{critic}}$")
    return fig_equation(latex, "unified_loss.png", fontsize=22)


def fig_text_gate() -> Path:
    latex = (r"$x'_v = x_v + \sigma(g) \odot (W_{\mathrm{text}}\, t_v)$" + "\n\n"
             r"$g \leftarrow -2,\quad \sigma(-2)\approx 0.12$")
    fig = plt.figure(figsize=(10, 2.6))
    fig.text(0.5, 0.55, latex, ha="center", va="center",
             fontsize=24, color=_mpl_color(PRIMARY))
    plt.axis("off")
    out = FIG_DIR / "text_gate.png"
    fig.savefig(out, dpi=240, bbox_inches="tight", facecolor="white",
                pad_inches=0.25)
    plt.close(fig)
    return out


# -----------------------------------------------------------------------------
# Slide-building helpers
# -----------------------------------------------------------------------------
def set_background(slide, color: RGBColor = WHITE) -> None:
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_title_bar(slide, title: str, subtitle: str | None = None) -> None:
    """Add a consistent title area at the top of every content slide."""
    # Left-accent bar
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE,
                                  Inches(0), Inches(0.35),
                                  Inches(0.18), Inches(0.8))
    bar.line.fill.background()
    bar.fill.solid()
    bar.fill.fore_color.rgb = ACCENT

    # Title text
    tb = slide.shapes.add_textbox(Inches(0.45), Inches(0.25),
                                   Inches(SLIDE_W_IN - 0.9), Inches(0.9))
    tf = tb.text_frame
    tf.margin_left = tf.margin_right = 0
    tf.margin_top = tf.margin_bottom = 0
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    run = p.add_run()
    run.text = title
    run.font.size = Pt(28)
    run.font.bold = True
    run.font.name = FONT
    run.font.color.rgb = PRIMARY

    if subtitle:
        p2 = tf.add_paragraph()
        r2 = p2.add_run()
        r2.text = subtitle
        r2.font.size = Pt(14)
        r2.font.italic = True
        r2.font.name = FONT
        r2.font.color.rgb = TEXT_MUTED

    # Bottom hairline under title
    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE,
                                   Inches(0.45), Inches(1.25),
                                   Inches(SLIDE_W_IN - 0.9), Inches(0.02))
    line.line.fill.background()
    line.fill.solid()
    line.fill.fore_color.rgb = RGBColor(0xE0, 0xE0, 0xE0)


def add_lead(slide, text: str, top: float = 1.35, font_size: int = 13) -> None:
    """Explanatory lead paragraph just below the title bar. Keeps the slide
    readable without a speaker: a professor should be able to understand the
    topic from the lead sentence alone."""
    tb = slide.shapes.add_textbox(Inches(0.6), Inches(top),
                                   Inches(SLIDE_W_IN - 1.2), Inches(0.55))
    tf = tb.text_frame
    tf.margin_left = tf.margin_right = 0
    tf.margin_top = tf.margin_bottom = 0
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    r = p.add_run()
    r.text = text
    r.font.size = Pt(font_size)
    r.font.italic = True
    r.font.name = FONT
    r.font.color.rgb = TEXT_MUTED


def add_footer(slide, page_num: int, total: int, section: str = "") -> None:
    tb = slide.shapes.add_textbox(Inches(0.45), Inches(SLIDE_H_IN - 0.35),
                                   Inches(SLIDE_W_IN - 0.9), Inches(0.25))
    tf = tb.text_frame
    tf.margin_left = tf.margin_right = 0
    p = tf.paragraphs[0]
    r = p.add_run()
    txt = f"{section}  |  {page_num} / {total}" if section else f"{page_num} / {total}"
    r.text = txt
    r.font.size = Pt(9)
    r.font.name = FONT
    r.font.color.rgb = TEXT_MUTED


def add_bullets(slide, items: list, left: float, top: float, width: float,
                 height: float, font_size: int = 16, bullet_color: RGBColor = ACCENT) -> None:
    """Add a bulleted list. `items` is a list of (text, level?) or str."""
    tb = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = 0
    tf.margin_top = tf.margin_bottom = 0

    for i, item in enumerate(items):
        if isinstance(item, tuple):
            text, level = item
        else:
            text, level = item, 0
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.level = level
        p.alignment = PP_ALIGN.LEFT
        # Manual bullet (python-pptx bullets are unreliable)
        bullet_char = "▸" if level == 0 else "–"
        r1 = p.add_run()
        r1.text = f"{bullet_char}  "
        r1.font.size = Pt(font_size)
        r1.font.bold = True
        r1.font.name = FONT
        r1.font.color.rgb = bullet_color
        r2 = p.add_run()
        r2.text = text
        r2.font.size = Pt(font_size)
        r2.font.name = FONT
        r2.font.color.rgb = TEXT_MAIN
        p.space_after = Pt(8)


def add_paragraph(slide, text: str, left: float, top: float, width: float,
                   height: float, font_size: int = 16, bold: bool = False,
                   color: RGBColor = TEXT_MAIN, align=PP_ALIGN.LEFT,
                   italic: bool = False) -> object:
    tb = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = 0
    p = tf.paragraphs[0]
    p.alignment = align
    r = p.add_run()
    r.text = text
    r.font.size = Pt(font_size)
    r.font.bold = bold
    r.font.italic = italic
    r.font.name = FONT
    r.font.color.rgb = color
    return tb


def add_rich_paragraph(slide, runs: list, left: float, top: float, width: float,
                        height: float, align=PP_ALIGN.LEFT) -> object:
    """runs is list of dicts: {text, size, bold, italic, color}."""
    tb = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = 0
    p = tf.paragraphs[0]
    p.alignment = align
    for spec in runs:
        r = p.add_run()
        r.text = spec.get("text", "")
        r.font.size = Pt(spec.get("size", 16))
        r.font.bold = spec.get("bold", False)
        r.font.italic = spec.get("italic", False)
        r.font.name = FONT
        r.font.color.rgb = spec.get("color", TEXT_MAIN)
    return tb


def add_info_box(slide, title: str, body: str, left: float, top: float,
                  width: float, height: float, color: RGBColor = ACCENT,
                  font_size: int = 14) -> None:
    box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                  Inches(left), Inches(top),
                                  Inches(width), Inches(height))
    box.line.color.rgb = color
    box.line.width = Pt(1.5)
    box.fill.solid()
    box.fill.fore_color.rgb = RGBColor(0xFA, 0xFA, 0xFA)

    tb = slide.shapes.add_textbox(Inches(left + 0.2), Inches(top + 0.15),
                                   Inches(width - 0.4), Inches(height - 0.3))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = 0
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    r = p.add_run()
    r.text = title
    r.font.size = Pt(font_size + 2)
    r.font.bold = True
    r.font.name = FONT
    r.font.color.rgb = color

    p2 = tf.add_paragraph()
    r2 = p2.add_run()
    r2.text = body
    r2.font.size = Pt(font_size)
    r2.font.name = FONT
    r2.font.color.rgb = TEXT_MAIN


def add_table(slide, data: list, left: float, top: float, width: float,
               height: float, header_color: RGBColor = PRIMARY,
               bold_rows: list | None = None, font_size: int = 12,
               first_col_bold: bool = False) -> None:
    rows, cols = len(data), len(data[0])
    tbl_shape = slide.shapes.add_table(rows, cols,
                                        Inches(left), Inches(top),
                                        Inches(width), Inches(height))
    tbl = tbl_shape.table

    bold_rows = bold_rows or []

    for r, row in enumerate(data):
        for c, val in enumerate(row):
            cell = tbl.cell(r, c)
            cell.text = ""
            tf = cell.text_frame
            tf.margin_left = Emu(60000)
            tf.margin_right = Emu(60000)
            tf.margin_top = Emu(40000)
            tf.margin_bottom = Emu(40000)
            p = tf.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if c == 0 else PP_ALIGN.CENTER
            run = p.add_run()
            run.text = str(val)
            run.font.size = Pt(font_size)
            run.font.name = FONT
            if r == 0:
                run.font.bold = True
                run.font.color.rgb = WHITE
                cell.fill.solid()
                cell.fill.fore_color.rgb = header_color
            else:
                run.font.color.rgb = TEXT_MAIN
                if r in bold_rows:
                    run.font.bold = True
                if first_col_bold and c == 0:
                    run.font.bold = True
                # zebra
                if r % 2 == 0:
                    cell.fill.solid()
                    cell.fill.fore_color.rgb = LIGHT_BG
                else:
                    cell.fill.solid()
                    cell.fill.fore_color.rgb = WHITE


# -----------------------------------------------------------------------------
# Slide builders
# -----------------------------------------------------------------------------
def build():
    prs = Presentation()
    prs.slide_width = Inches(SLIDE_W_IN)
    prs.slide_height = Inches(SLIDE_H_IN)

    blank = prs.slide_layouts[6]

    # Precompute figures
    fig_prog = fig_progression()
    fig_pil = fig_three_pillars()
    fig_ac = fig_actor_critic_equations()
    fig_loss = fig_unified_loss()
    fig_gate = fig_text_gate()

    slides_meta = []  # (section_name,) for footer

    # -------------------------------------------------------------------------
    # SLIDE 1: TITLE
    # -------------------------------------------------------------------------
    s = prs.slides.add_slide(blank)
    set_background(s, PRIMARY)
    # Accent bar on left
    bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0),
                              Inches(0.35), Inches(SLIDE_H_IN))
    bar.line.fill.background()
    bar.fill.solid()
    bar.fill.fore_color.rgb = ACCENT

    add_paragraph(s, "A Unified GNN + RL + LLM Framework",
                   0.8, 1.8, 12, 0.9, font_size=44, bold=True, color=WHITE)
    add_paragraph(s, "for Graph-Based Fraud Detection",
                   0.8, 2.6, 12, 0.9, font_size=44, bold=True, color=WHITE)
    add_paragraph(s, "Extending CARE-GNN with Learned Policy and LLM Reasoning",
                   0.8, 3.7, 12, 0.5, font_size=20, italic=True, color=RGBColor(0xD8, 0xD8, 0xD8))

    # Divider line
    line = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(4.4),
                               Inches(4.0), Inches(0.03))
    line.line.fill.background()
    line.fill.solid()
    line.fill.fore_color.rgb = ACCENT

    add_paragraph(s, "Fady Hassanein", 0.8, 4.7, 10, 0.5,
                   font_size=22, bold=True, color=WHITE)
    add_paragraph(s, "Master's Thesis", 0.8, 5.3, 10, 0.4,
                   font_size=16, color=RGBColor(0xBB, 0xBB, 0xBB))
    slides_meta.append(("Title",))

    # -------------------------------------------------------------------------
    # SLIDE 2: OUTLINE
    # -------------------------------------------------------------------------
    s = prs.slides.add_slide(blank)
    set_background(s)
    add_title_bar(s, "Outline")

    outline = [
        "1.  Motivation and problem",
        "2.  Literature review",
        "3.  The unified framework",
        "4.  Datasets",
        "5.  Methodology: GNN pillar",
        "6.  Methodology: RL pillar",
        "7.  Methodology: LLM pillar",
        "8.  Unified training",
        "9.  Results",
        "10. Conclusion",
    ]
    for i, item in enumerate(outline):
        add_paragraph(s, item, 1.5, 1.7 + 0.42 * i, 10, 0.4,
                       font_size=20, color=TEXT_MAIN)
    slides_meta.append(("Outline",))

    # -------------------------------------------------------------------------
    # SECTION 1: MOTIVATION
    # -------------------------------------------------------------------------
    # SLIDE 3: Why graph-based
    s = prs.slides.add_slide(blank)
    set_background(s)
    add_title_bar(s, "Why Graph-Based Fraud Detection?")
    add_lead(s,
             "Fraud on review platforms is a multi-billion-dollar problem, and fraudsters rarely act alone. "
             "Their social and behavioural structure is exactly the kind of signal graph-based methods are designed to exploit.")
    add_paragraph(s, "Online fraud is expensive and networked.",
                   0.6, 2.05, 12, 0.5, font_size=17, bold=True, color=PRIMARY)
    add_bullets(s, [
        "Fake reviews on Amazon and Yelp distort consumer decisions and merchant revenue.",
        "Fraudsters don't act alone — they form rings: coordinated users, burst-posted reviews, shared IP ranges, identical text templates.",
        "A single-user classifier misses the network evidence.",
    ], 0.8, 2.7, 12, 2.5, font_size=16)
    add_info_box(s, "Graph Neural Networks (GNNs)",
                  "Exploit this structure by propagating information across user/review relationships, so a node is classified using evidence from its neighbours as well as its own features.",
                  0.8, 5.5, 11.5, 1.4, color=ACCENT, font_size=15)
    slides_meta.append(("Motivation",))

    # SLIDE 4: Camouflage
    s = prs.slides.add_slide(blank)
    set_background(s)
    add_title_bar(s, "The Camouflage Problem")
    add_lead(s,
             "Even a well-designed GNN can be defeated by fraudsters who deliberately manipulate their features and "
             "their connections to blend in with legitimate users. Any serious fraud-detection GNN must model both forms of camouflage.")
    add_paragraph(s, "Fraudsters actively evade detection in two ways:",
                   0.6, 2.0, 12, 0.5, font_size=17, color=PRIMARY)

    # Two side-by-side boxes
    def camouflage_box(title, body, left, color):
        box = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                  Inches(left), Inches(2.2), Inches(5.8), Inches(1.8))
        box.line.color.rgb = color
        box.line.width = Pt(2)
        box.fill.solid()
        box.fill.fore_color.rgb = RGBColor(0xFF, 0xF5, 0xF0)
        tb = s.shapes.add_textbox(Inches(left + 0.2), Inches(2.3),
                                    Inches(5.4), Inches(1.7))
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        r = p.add_run()
        r.text = title
        r.font.size = Pt(18); r.font.bold = True
        r.font.name = FONT; r.font.color.rgb = color
        p2 = tf.add_paragraph()
        p2.space_before = Pt(6)
        r2 = p2.add_run()
        r2.text = body
        r2.font.size = Pt(14); r2.font.name = FONT
        r2.font.color.rgb = TEXT_MAIN

    def _box(title, body, left, color):
        box = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                  Inches(left), Inches(2.65), Inches(5.8), Inches(1.8))
        box.line.color.rgb = color
        box.line.width = Pt(2)
        box.fill.solid()
        box.fill.fore_color.rgb = RGBColor(0xFF, 0xF5, 0xF0)
        tb = s.shapes.add_textbox(Inches(left + 0.2), Inches(2.75),
                                    Inches(5.4), Inches(1.7))
        tf = tb.text_frame; tf.word_wrap = True
        p = tf.paragraphs[0]
        r = p.add_run(); r.text = title
        r.font.size = Pt(18); r.font.bold = True
        r.font.name = FONT; r.font.color.rgb = color
        p2 = tf.add_paragraph(); p2.space_before = Pt(6)
        r2 = p2.add_run(); r2.text = body
        r2.font.size = Pt(14); r2.font.name = FONT
        r2.font.color.rgb = TEXT_MAIN

    _box("Feature camouflage",
         "Fraudsters mimic legitimate users: normal ratings, plausible review length, boilerplate sentiment.",
         0.8, NEGATIVE)
    _box("Relation camouflage",
         "Fraudsters form ties to benign users to dilute the suspicious signal in neighbourhood aggregation.",
         6.8, NEGATIVE)

    add_paragraph(s, "A naive GNN aggregates all neighbours and inherits both attacks.",
                   0.8, 4.65, 12, 0.5, font_size=15, italic=True, color=TEXT_MUTED)

    add_info_box(s, "Research Question",
                  "Can we build a GNN that filters neighbours adaptively, incorporates semantic reasoning, and remains interpretable — without sacrificing scalability or reproducibility?",
                  0.8, 5.35, 11.5, 1.5, color=ACCENT, font_size=15)
    slides_meta.append(("Motivation",))

    # -------------------------------------------------------------------------
    # SECTION 2: LITERATURE REVIEW
    # -------------------------------------------------------------------------
    # SLIDE 5: Progression
    s = prs.slides.add_slide(blank)
    set_background(s)
    add_title_bar(s, "Progression of Fraud-Detection Methods")
    add_lead(s,
             "Fraud detection has evolved through five major waves, each fixing a specific limitation of the previous one. "
             "This thesis sits at the rightmost position — combining all the advances of prior generations.")
    s.shapes.add_picture(str(fig_prog), Inches(0.5), Inches(2.05),
                          width=Inches(12.3))
    add_bullets(s, [
        "Classical ML: effective baselines, but ignore relational topology.",
        "Deep learning: stronger features, but nodes treated independently.",
        "GNNs: end-to-end relational learning (GCN, GraphSAGE, GAT, …).",
        "GNN + RL: CARE-GNN (Dou et al. 2020) adds a bandit-based neighbour filter.",
        "GNN + LLM: FLAG (Yang et al. 2025) uses text for neighbour sampling.",
    ], 0.8, 4.9, 12, 2.3, font_size=13)
    slides_meta.append(("Literature Review",))

    # SLIDE 6: Research gap
    s = prs.slides.add_slide(blank)
    set_background(s)
    add_title_bar(s, "Research Gap")
    add_lead(s,
             "Each wave above fixed one problem but introduced another. Crucially, no prior work combines structural (GNN), "
             "adaptive (RL), and semantic (LLM) reasoning in a single framework — that unfilled slot is the space this thesis occupies.")
    add_paragraph(s, "No existing work unifies all three components:",
                   0.6, 2.0, 12, 0.5, font_size=18, bold=True, color=PRIMARY)
    add_bullets(s, [
        "Existing RL + GNN frameworks (CARE-GNN, RioGNN) use simple heuristic bandits with no semantic state.",
        "Existing LLM + GNN work (FLAG) augments features or sampling but does not use a learned RL policy for dynamic weighting.",
    ], 0.8, 2.7, 12, 1.5, font_size=15)

    # "This thesis" callout
    box = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                              Inches(0.8), Inches(4.25), Inches(11.8), Inches(2.7))
    box.line.color.rgb = ACCENT
    box.line.width = Pt(2)
    box.fill.solid()
    box.fill.fore_color.rgb = RGBColor(0xFF, 0xF8, 0xF0)
    add_paragraph(s, "This thesis", 1.1, 4.4, 10, 0.5,
                   font_size=21, bold=True, color=ACCENT)
    add_paragraph(s,
                   "A unified GNN + RL + LLM framework that combines:",
                   1.1, 4.95, 10, 0.4, font_size=15, color=TEXT_MAIN)
    add_bullets(s, [
        "A multi-relation GNN backbone (CARE-GNN).",
        "A learned actor–critic RL policy for per-node filtering.",
        "LLM-derived signals injected into both the policy state and the node features.",
    ], 1.3, 5.4, 11, 1.5, font_size=14)
    slides_meta.append(("Literature Review",))

    # -------------------------------------------------------------------------
    # SECTION 3: UNIFIED FRAMEWORK
    # -------------------------------------------------------------------------
    # SLIDE 7: Three pillars diagram
    s = prs.slides.add_slide(blank)
    set_background(s)
    add_title_bar(s, "Three-Pillar Architecture")
    add_lead(s,
             "The framework is built from three independent pillars that plug into each other through well-defined interfaces. "
             "Read bottom-up: raw inputs enter the LLM and RL pillars, which feed their outputs into the GNN pillar, which produces the final fraud probability.")
    s.shapes.add_picture(str(fig_pil), Inches(0.4), Inches(1.95),
                          width=Inches(12.5))
    slides_meta.append(("Unified Framework",))

    # SLIDE 8: Design principle
    s = prs.slides.add_slide(blank)
    set_background(s)
    add_title_bar(s, "Design Principle: Additive, Attributable Pillars")
    add_lead(s,
             "Splitting the framework into three pillars is not just a conceptual choice — it is an engineering decision that "
             "lets us measure each pillar's contribution separately, swap pillars without refactoring, and reproduce every result from a single command.")
    add_paragraph(s, "Every design choice preserves two interoperability constraints:",
                   0.6, 2.0, 12, 0.5, font_size=17, bold=True, color=PRIMARY)
    add_bullets(s, [
        "Each pillar enters the framework at a well-defined interface (feature space, policy state, or threshold variable).",
        "Enabling or disabling a pillar leaves the other two's code unchanged.",
    ], 0.8, 2.7, 12, 1.5, font_size=15)

    add_paragraph(s, "Why this matters",
                   0.6, 4.55, 12, 0.5, font_size=17, bold=True, color=PRIMARY)
    add_bullets(s, [
        "The three-pillar ablation in Chapter 4 measures each pillar's contribution cleanly (no confounding).",
        "Reproducibility: each pillar toggle corresponds to a single command-line flag.",
        "Extensibility: new LLMs or alternative policies slot in without refactoring the backbone.",
    ], 0.8, 5.25, 12, 2.0, font_size=15)
    slides_meta.append(("Unified Framework",))

    # -------------------------------------------------------------------------
    # SECTION 4: DATASETS
    # -------------------------------------------------------------------------
    s = prs.slides.add_slide(blank)
    set_background(s)
    add_title_bar(s, "Benchmark Datasets")
    add_lead(s,
             "We evaluate on two complementary benchmarks. Amazon tests whether each pillar adds value when per-node features already carry most of the signal; "
             "YelpChi tests whether the same three pillars combine to beat CARE-GNN when features are weak and the graph carries the signal.")
    data = [
        ["Property", "Amazon", "YelpChi"],
        ["Nodes (total / labelled)", "11,944 / 8,639", "45,954 / 45,954"],
        ["Features", "25 raw", "32 pre-normalised"],
        ["Fraud rate (labelled)", "9.50 %", "14.53 %"],
        ["Relations", "UPU, USU, UVU", "RUR, RTR, RSR"],
        ["Max |r| feature", "0.654 (feat. 19)", "0.235 (feat. 5)"],
        ["Strongest relation", "UVU (18.3 / 3.0 %)", "RUR (92.5 / 0.4 %)"],
        ["Review text available?", "No", "Yes"],
    ]
    add_table(s, data, 1.8, 2.0, 9.8, 3.8, font_size=13,
              first_col_bold=True)
    add_bullets(s, [
        "Amazon: strong features, modest homophily — features already solve most of the problem, so pillar gains are small.",
        "YelpChi: weak features; RUR's near-perfect label homophily is the main signal, but RTR / RSR are noisy.",
        "All statistics recomputed from the release used in this thesis (verification scripts in the repo).",
    ], 0.8, 6.0, 12, 1.2, font_size=12)
    slides_meta.append(("Datasets",))

    # -------------------------------------------------------------------------
    # SECTION 5: GNN PILLAR
    # -------------------------------------------------------------------------
    s = prs.slides.add_slide(blank)
    set_background(s)
    add_title_bar(s, "GNN Backbone: CARE-GNN")
    add_lead(s,
             "Our backbone is CARE-GNN (Dou et al. 2020) — a multi-relation GNN specifically designed for fraud detection. "
             "Its key idea: rank every neighbour by label agreement before aggregating, so camouflaged edges can be filtered out.")
    add_paragraph(s, "Single-layer, multi-relation GNN with label-aware filtering:",
                   0.6, 2.0, 12, 0.4, font_size=16, color=PRIMARY)
    steps = [
        "Label predictor  S(v) = MLP(xᵥ) ∈ ℝ²  produces per-node class logits.",
        "Pairwise distance  D(v, v′) = | S(v)[0] − S(v′)[0] |  ranks neighbours by label agreement.",
        "Filter: retain the top-pᵣ fraction by distance in each relation r.",
        "Intra-relation aggregation: mean of retained neighbours.",
        "Inter-relation aggregation: hᵥ = ReLU( W·hᵥ + Σᵣ pᵣ · W·h_{𝒩(v)}^(r) ).",
    ]
    for i, text in enumerate(steps):
        add_paragraph(s, f"{i+1}.  {text}", 0.9, 2.55 + i * 0.38, 12, 0.38,
                       font_size=13, color=TEXT_MAIN)
    add_info_box(s, "What we keep vs replace",
                  "Replace:  step 1 linear predictor → two-layer MLP (CAPN);  step 3 bandit → actor–critic policy.\n"
                  "Keep:  steps 2, 4, 5 unchanged — this is what lets the LLM and RL pillars plug in without breaking the backbone.",
                  0.8, 4.85, 11.8, 1.7, color=POSITIVE, font_size=13)
    slides_meta.append(("GNN Pillar",))

    # -------------------------------------------------------------------------
    # SECTION 6: RL PILLAR
    # -------------------------------------------------------------------------
    # Slide 11: Bandit -> CAPN
    s = prs.slides.add_slide(blank)
    set_background(s)
    add_title_bar(s, "From Heuristic Bandit to CAPN")
    add_lead(s,
             "CARE-GNN updates its neighbour-filtering thresholds with a simple bandit that is cheap but blind — it uses no node-level state "
             "and cannot absorb any external signal. CAPN keeps the idea of adaptive thresholds but replaces the bandit with a learned neural policy.")
    add_paragraph(s, "Inherited bandit (CARE-GNN):",
                   0.6, 1.95, 12, 0.4, font_size=15, bold=True, color=PRIMARY)
    add_bullets(s, [
        "Action: pᵣ ← pᵣ ± 0.02, updated once per epoch.",
        "Reward: ±1 based on sign of change in positive-node distance.",
        "Terminal condition: freezes once reward plateaus.",
    ], 0.8, 2.4, 12, 1.3, font_size=13)

    add_info_box(s, "Five Limitations",
                  "Global (not per-node) thresholds   •   binary reward   •   slow fixed-step updates   •   linear label predictor   •   NO HOOK FOR EXTERNAL (LLM) SIGNALS.",
                  0.8, 3.8, 11.8, 1.1, color=NEGATIVE, font_size=12)

    add_paragraph(s, "CAPN (ours): Camouflage-Aware Policy Network",
                   0.6, 5.1, 12, 0.4, font_size=15, bold=True, color=ACCENT)
    add_bullets(s, [
        "Enhanced two-layer MLP label predictor with confidence output.",
        "Rich state vector: features + distance + overlap + feature variance + LLM scores.",
        "Beta-distribution policy network → per-node thresholds (rather than per-relation).",
        "Shaped, continuous reward: distance + accuracy + stability.",
    ], 0.8, 5.55, 12, 1.7, font_size=13)
    slides_meta.append(("RL Pillar",))

    # Slide 12: Actor-critic equations
    s = prs.slides.add_slide(blank)
    set_background(s)
    add_title_bar(s, "Actor–Critic Policy Optimisation")
    add_lead(s,
             "Training a policy network with a single noisy batch reward is unstable — the gradient variance is too high and the policy "
             "oscillates (we confirmed this empirically on YelpChi). Using a learned value function as a baseline solves this.")
    add_paragraph(s, "Problem: vanilla REINFORCE has high gradient variance → policy oscillates, YelpChi training diverges.",
                   0.6, 2.0, 12, 0.5, font_size=14, color=TEXT_MAIN)
    add_paragraph(s, "Solution: single-step actor–critic with a learned value baseline.",
                   0.6, 2.5, 12, 0.4, font_size=14, bold=True, color=PRIMARY)
    s.shapes.add_picture(str(fig_ac), Inches(1.2), Inches(2.95), width=Inches(10.8))
    add_paragraph(s, "Training schedule (prevents policy from learning on a noisy, untrained backbone):",
                   0.6, 5.55, 12, 0.4, font_size=14, bold=True, color=PRIMARY)
    add_bullets(s, [
        "Epochs 0–5 (GNN warmup):   λπ = 0, only supervised loss runs.",
        "Epochs 5–10 (policy ramp):   linearly anneal λπ from 0 to 0.3.",
        "Epochs 10–31:   full policy training at λπ = 0.3.",
    ], 0.8, 5.95, 12, 1.4, font_size=12)
    slides_meta.append(("RL Pillar",))

    # -------------------------------------------------------------------------
    # SECTION 7: LLM PILLAR
    # -------------------------------------------------------------------------
    # Slide 13: LLM as reasoning engine
    s = prs.slides.add_slide(blank)
    set_background(s)
    add_title_bar(s, "LLM as Reasoning Engine, Not Text Encoder")
    add_lead(s,
             "Our first experiments tried the obvious thing — plug sentence-transformer embeddings into the policy state — and it failed. "
             "Our LLM pillar instead uses the LLM as a reasoning engine that emits compact, fraud-specific signals, not as a generic text encoder.")
    add_paragraph(s, "Why not sentence-transformer embeddings?",
                   0.6, 2.0, 12, 0.4, font_size=16, bold=True, color=PRIMARY)
    add_bullets(s, [
        "Study A6 tried this on Amazon: 384-dim MiniLM embeddings projected into the policy state.  Result: −0.19 pp AUC (FAILED).",
        "Root cause: lossy round-trip converts precise numerics (degree = 23) into text and back, destroying the signal.",
    ], 0.8, 2.45, 12, 1.2, font_size=13)

    add_paragraph(s, "Our approach: use Claude Haiku 4.5 as a reasoning engine. Two complementary signal streams:",
                   0.6, 3.75, 12, 0.5, font_size=14, color=PRIMARY)
    data = [
        ["", "Part 1 (both datasets)", "Part 2 (YelpChi only)"],
        ["Input to LLM", "26 graph statistics per node", "raw review text"],
        ["Output", "6 structural risk scores", "6 text risk scores"],
        ["Entry point", "CAPN state vector", "node features (via gate)"],
    ]
    add_table(s, data, 1.6, 4.35, 10.2, 2.15, font_size=13, first_col_bold=True)
    add_paragraph(s,
                   "All LLM scores are precomputed once and loaded as tensors — no API calls inside the training loop (reproducibility + cost).",
                   0.8, 6.7, 12, 0.5, font_size=12, italic=True, color=TEXT_MUTED)
    slides_meta.append(("LLM Pillar",))

    # Slide 14: Part 1
    s = prs.slides.add_slide(blank)
    set_background(s)
    add_title_bar(s, "Part 1: Structural Reasoning Scores")
    add_lead(s,
             "For every node we compute 26 graph statistics (degrees, overlaps, z-scores, ego-density, 2-hop reach …) "
             "and ask Claude to return six fraud-relevant risk assessments. These scores enter the CAPN state so the policy can condition on LLM-level reasoning about structure.")
    add_paragraph(s, "26 graph statistics per node  →  Claude Haiku 4.5  →  6 risk scores:",
                   0.6, 2.0, 12, 0.5, font_size=15, bold=True, color=PRIMARY)
    scores = [
        ("structural_anomaly", "unusual connectivity pattern"),
        ("relation_consistency", "uniform behaviour across relations"),
        ("neighborhood_risk", "suspicious neighbours (training-label stats)"),
        ("feature_anomaly", "statistical feature outliers"),
        ("coordination_signal", "likelihood of coordinated activity"),
        ("isolation_score", "peripheral vs deeply embedded"),
    ]
    for i, (name, desc) in enumerate(scores):
        row_y = 2.55 + i * 0.33
        add_rich_paragraph(s, [
            {"text": "▸  ", "size": 12, "bold": True, "color": ACCENT},
            {"text": name, "size": 12, "bold": True, "color": PRIMARY},
            {"text": "  —  " + desc, "size": 12, "color": TEXT_MAIN},
        ], 0.8, row_y, 12, 0.33)
    add_info_box(s, "Why LLM over a linear formula?",
                  "The LLM reasons about interactions: 'high ego-density combined with high label disagreement in a single relation' signals a fraud ring — a linear formula misses that conjunction. Study A7 confirms this beats the template-formula baseline.",
                  0.8, 4.7, 11.8, 1.5, color=POSITIVE, font_size=12)
    add_paragraph(s,
                   "Integration:  6-score vector → linear projector → 16-dim eᵥ  concatenated into CAPN state.  State grows  D+5 → D+21.",
                   0.8, 6.35, 12, 0.5, font_size=13, color=TEXT_MAIN)
    slides_meta.append(("LLM Pillar",))

    # Slide 15: Part 2 + gate
    s = prs.slides.add_slide(blank)
    set_background(s)
    add_title_bar(s, "Part 2: Text-Risk Scoring and Gated Injection")
    add_lead(s,
             "On YelpChi each review has a body of text. We score each review for fraud-specific linguistic signals — but concatenating "
             "those scores to node features hurts performance (Study A8). A learnable gated residual lets the network admit as much text signal as training finds useful.")
    add_paragraph(s,
                   "YelpChi only (Amazon ships no review text). Six fraud-specific text scores per review:",
                   0.6, 2.0, 12, 0.5, font_size=14, color=PRIMARY)
    add_bullets(s, [
        "generic_language  •  sentiment_mismatch  •  promotional_tone",
        "copy_paste_signal  •  detail_authenticity  •  behavioral_anomaly",
    ], 0.8, 2.5, 12, 1.0, font_size=13)

    add_paragraph(s, "How not to inject:  Study A8 concatenated scores to features → −0.59 pp AUC (wider weight matrix absorbs noise).",
                   0.6, 3.55, 12, 0.5, font_size=12, italic=True, color=NEGATIVE)

    add_paragraph(s, "TextFeatureGate (ours): residual with learnable per-dim sigmoid gate.",
                   0.6, 4.1, 12, 0.5, font_size=14, bold=True, color=ACCENT)
    s.shapes.add_picture(str(fig_gate), Inches(1.8), Inches(4.55), width=Inches(9.8))
    add_bullets(s, [
        "Preserves original feature dimension  ℝ³² → ℝ³²  (downstream code unchanged).",
        "Gate starts near zero: initial behaviour ≈ CARE-GNN — safe initialisation.",
        "Training can unlearn harmful dimensions by pushing g → −∞.",
    ], 0.8, 6.3, 12, 1.1, font_size=12)
    slides_meta.append(("LLM Pillar",))

    # -------------------------------------------------------------------------
    # SECTION 8: UNIFIED TRAINING
    # -------------------------------------------------------------------------
    s = prs.slides.add_slide(blank)
    set_background(s)
    add_title_bar(s, "Joint Training Objective")
    add_lead(s,
             "All three pillars are trained end-to-end under a single loss function that combines supervised classification, "
             "label-agreement supervision, policy gradient, and critic regression. Two independent Adam optimisers keep the different time-scales separated.")
    add_paragraph(s, "Single end-to-end loss combining all four signals:",
                   0.6, 2.0, 12, 0.5, font_size=16, bold=True, color=PRIMARY)
    s.shapes.add_picture(str(fig_loss), Inches(1.3), Inches(2.6), width=Inches(10.7))

    add_paragraph(s, "Dual-optimiser setup:",
                   0.6, 4.5, 12, 0.5, font_size=16, bold=True, color=PRIMARY)
    add_bullets(s, [
        "Backbone optimiser:  Adam,  LR 10⁻²,  clip max-norm 1.0.",
        "Policy / critic optimiser:  Adam,  policy LR 3·10⁻³,  critic LR 10⁻³,  clip max-norm 5.0.",
    ], 0.8, 5.0, 12, 1.2, font_size=14)

    add_info_box(s, "Why two optimisers?",
                  "The actor–critic and the backbone converge on different timescales — a single learning rate destabilises training.",
                  0.8, 6.25, 11.8, 0.95, color=ACCENT, font_size=12)
    slides_meta.append(("Unified Training",))

    # -------------------------------------------------------------------------
    # SECTION 9: RESULTS
    # -------------------------------------------------------------------------
    # Slide 17: Protocol
    s = prs.slides.add_slide(blank)
    set_background(s)
    add_title_bar(s, "Experimental Protocol")
    add_lead(s,
             "All experiments follow the same protocol — stratified splits, fixed 31-epoch budget, best-validation checkpointing, and multiple seeds on YelpChi. "
             "The goal is a statistically honest comparison between pillar configurations, not best-of-single-run cherry-picking.")
    add_bullets(s, [
        "Splits:  stratified  25 % / 15 % / 60 %  train / val / test.",
        "Epochs:  31;  best-validation-AUC checkpoint per run.",
        "Seeds:  three seeds {72, 73, 74} on YelpChi (mean ± std);  single seed 72 for Amazon ablations.",
        "Metrics:  AUC-ROC and Average Precision (primary);  F1-macro, Precision, Recall (secondary).",
    ], 0.8, 2.2, 12, 2.5, font_size=16)

    add_info_box(s, "Thesis claim tested",
                  "The three-pillar combination (GNN + RL + LLM) significantly outperforms CARE-GNN on YelpChi.",
                  0.8, 5.6, 11.8, 1.2, color=ACCENT, font_size=16)
    slides_meta.append(("Results",))

    # Slide 18: Headline
    s = prs.slides.add_slide(blank)
    set_background(s)
    add_title_bar(s, "Headline Result — YelpChi")
    # Big numbers
    add_paragraph(s, "AUC", 1.5, 1.9, 3, 0.5, font_size=20, bold=True, color=TEXT_MUTED, align=PP_ALIGN.CENTER)
    add_paragraph(s, "0.7856 ± 0.0022", 1.5, 2.4, 3, 0.9, font_size=36, bold=True, color=ACCENT, align=PP_ALIGN.CENTER)

    add_paragraph(s, "AP", 5.2, 1.9, 3, 0.5, font_size=20, bold=True, color=TEXT_MUTED, align=PP_ALIGN.CENTER)
    add_paragraph(s, "0.4108 ± 0.0052", 5.2, 2.4, 3, 0.9, font_size=36, bold=True, color=ACCENT, align=PP_ALIGN.CENTER)

    add_paragraph(s, "F1-macro", 8.9, 1.9, 3, 0.5, font_size=20, bold=True, color=TEXT_MUTED, align=PP_ALIGN.CENTER)
    add_paragraph(s, "0.6099 ± 0.0565", 8.9, 2.4, 3, 0.9, font_size=36, bold=True, color=ACCENT, align=PP_ALIGN.CENTER)

    # Divider
    line = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(1.5), Inches(4.1),
                               Inches(10.3), Inches(0.03))
    line.line.fill.background()
    line.fill.solid()
    line.fill.fore_color.rgb = ACCENT

    add_paragraph(s, "+1.99 pp  AUC over our CARE-GNN reproduction",
                   0.6, 4.5, 12, 0.5, font_size=22, bold=True, color=POSITIVE, align=PP_ALIGN.CENTER)
    add_paragraph(s, "+2.86 pp  AUC over Dou et al. (2020) published CARE-GNN",
                   0.6, 5.1, 12, 0.5, font_size=22, bold=True, color=POSITIVE, align=PP_ALIGN.CENTER)
    add_paragraph(s, "Welch's t-test:  p < 0.001  across three seeds.",
                   0.6, 5.85, 12, 0.5, font_size=16, italic=True, color=TEXT_MAIN, align=PP_ALIGN.CENTER)
    slides_meta.append(("Results",))

    # Slide 19: Main table
    s = prs.slides.add_slide(blank)
    set_background(s)
    add_title_bar(s, "Main Table — Three-Pillar Ablation on YelpChi")
    add_lead(s,
             "Four rows, each turning exactly one pillar on or off. The transitions Row 1 → 2, 1 → 3, and 3 → 4 each attribute a specific design decision to a measurable AUC delta.")
    data = [
        ["Configuration", "AUC", "AP", "F1-macro"],
        ["Row 1: GNN only (CARE-GNN)", "0.7657 ± 0.0007", "0.3766 ± 0.0038", "0.5948 ± 0.0263"],
        ["Row 2: GNN + RL", "0.7681 ± 0.0005", "0.3837 ± 0.0007", "0.5968 ± 0.0109"],
        ["Row 3: GNN + LLM", "0.7797 ± 0.0038", "0.3983 ± 0.0146", "0.5927 ± 0.0578"],
        ["Row 4: GNN + RL + LLM", "0.7856 ± 0.0022", "0.4108 ± 0.0052", "0.6099 ± 0.0565"],
    ]
    add_table(s, data, 0.8, 2.1, 11.8, 2.3, font_size=13, bold_rows=[4])

    add_paragraph(s, "Component contributions (mean AUC):",
                   0.6, 4.6, 12, 0.4, font_size=15, bold=True, color=PRIMARY)
    add_rich_paragraph(s, [
        {"text": "▸  ", "size": 14, "bold": True, "color": ACCENT},
        {"text": "+0.24 pp", "size": 14, "bold": True, "color": POSITIVE},
        {"text": "   Row 1 → Row 2:  RL pillar alone.", "size": 14},
    ], 0.8, 5.05, 12, 0.4)
    add_rich_paragraph(s, [
        {"text": "▸  ", "size": 14, "bold": True, "color": ACCENT},
        {"text": "+1.40 pp", "size": 14, "bold": True, "color": POSITIVE},
        {"text": "   Row 1 → Row 3:  LLM pillar alone (dominant contributor).", "size": 14},
    ], 0.8, 5.45, 12, 0.4)
    add_rich_paragraph(s, [
        {"text": "▸  ", "size": 14, "bold": True, "color": ACCENT},
        {"text": "+0.59 pp", "size": 14, "bold": True, "color": POSITIVE},
        {"text": "   Row 3 → Row 4:  RL on top of LLM (stabilises and lifts).", "size": 14},
    ], 0.8, 5.85, 12, 0.4)

    add_paragraph(s,
                   "Worst seed of Row 4 (0.7828) already beats best seed of Row 1 (0.7663) by +1.65 pp — the improvement is robust, not a lucky seed.",
                   0.8, 6.65, 12, 0.5, font_size=12, italic=True, color=TEXT_MUTED)
    slides_meta.append(("Results",))

    # Slide 20: Significance
    s = prs.slides.add_slide(blank)
    set_background(s)
    add_title_bar(s, "Statistical Significance")
    add_lead(s,
             "With only three seeds the mean differences must be stress-tested. Welch's t-test shows the Full-vs-CARE-GNN improvement is far beyond noise; "
             "the incremental RL-on-top-of-LLM contribution is at the border of conventional significance — as expected with three seeds.")
    add_paragraph(s, "Two-sided Welch's t-test (three seeds):",
                   0.6, 2.0, 12, 0.4, font_size=15, color=PRIMARY)
    data = [
        ["Comparison", "ΔAUC", "t", "p"],
        ["Row 4 vs Row 1  (Full vs CARE-GNN)", "+1.99 pp", "≈ 12.4", "< 0.001"],
        ["Row 4 vs Row 2  (Full vs RL only)", "+1.75 pp", "≈ 11.7", "< 0.001"],
        ["Row 4 vs Row 3  (Full vs LLM only)", "+0.59 pp", "≈ 2.7", "≈ 0.05"],
    ]
    add_table(s, data, 1.2, 2.5, 11, 2.2, font_size=14, bold_rows=[1, 2])
    add_bullets(s, [
        "The full framework is significantly better than CARE-GNN AND than the RL-only variant.",
        "RL-on-top-of-LLM is at the border of conventional significance — would be strengthened by expanding the seed pool.",
    ], 0.8, 5.1, 12, 1.8, font_size=14)
    slides_meta.append(("Results",))

    # Slide 21: Comparison with paper
    s = prs.slides.add_slide(blank)
    set_background(s)
    add_title_bar(s, "Comparison with the CARE-GNN Paper (YelpChi)")
    add_lead(s,
             "Our full framework also improves over the original CARE-GNN paper's reported AUC by +2.86 pp — larger than the gain over our own Row 1, "
             "because our reproduction is already a stronger CARE-GNN baseline than the published one.")
    data = [
        ["Model", "AUC", "Recall"],
        ["GCN *", "0.5247", "0.5081"],
        ["GAT *", "0.5624", "0.5452"],
        ["GraphSAGE *", "0.5400", "0.5286"],
        ["GraphConsis *", "0.6207", "0.6208"],
        ["CARE-GNN *", "0.7570", "0.7192"],
        ["Our Row 1 (CARE-GNN reproduction)", "0.7657 ± 0.0007", "0.6987"],
        ["Our Row 2 (GNN + RL)", "0.7681 ± 0.0005", "0.6967"],
        ["Our Row 3 (GNN + LLM)", "0.7797 ± 0.0038", "0.6774"],
        ["Our Row 4 (GNN + RL + LLM)", "0.7856 ± 0.0022", "0.6923"],
    ]
    add_table(s, data, 2.0, 2.0, 9.5, 4.7, font_size=11, bold_rows=[9])
    add_paragraph(s,
                   "*  paper-reported values, Dou et al. (2020) Table 3, 40 % training data",
                   0.8, 6.75, 12, 0.3, font_size=10, italic=True, color=TEXT_MUTED)
    add_paragraph(s,
                   "Our full framework improves AUC by +2.86 pp over the paper-reported CARE-GNN.",
                   0.8, 7.08, 12, 0.3, font_size=13, bold=True, color=POSITIVE)
    slides_meta.append(("Results",))

    # Slide 22: Amazon ablations takeaways
    s = prs.slides.add_slide(blank)
    set_background(s)
    add_title_bar(s, "Amazon Component Ablations (A1–A9) Takeaways")
    add_lead(s,
             "Nine Amazon ablations attribute a single design decision per study. Two of them (A6, A8) are FAILED configurations — they were just as important, "
             "because their negative results drove the specific design choices that produced the final framework.")

    takeaways = [
        ("A1", "CAPN vs bandit", "+0.47 pp AUC over CARE-GNN; Label AUC +5.20 pp.", POSITIVE),
        ("A2", "Label predictor", "MLP alone hurts AUC (−0.77 pp); needs policy to realise the gain.", TEXT_MAIN),
        ("A3", "Reward shaping", "Any continuous reward beats binary; specific decomposition barely matters.", TEXT_MAIN),
        ("A4", "LLM priors", "Best single configuration; +0.81 pp Precision.", POSITIVE),
        ("A5", "λπ sensitivity", "Robust: AUC varies by 0.0004 across a 50× range.", TEXT_MAIN),
        ("A6", "Sentence-transformer (v1)", "FAILED (−0.19 pp). Lossy text round-trip destroyed signal.", NEGATIVE),
        ("A7", "Reasoning scores (v2)", "+0.05 pp AUC, +0.27 pp AP.", POSITIVE),
        ("A8", "Feature-level concat", "FAILED (−0.59 pp) — motivates the gated residual on YelpChi.", NEGATIVE),
        ("A9", "Combined", "No negative interactions — prerequisite for the YelpChi result.", TEXT_MAIN),
    ]
    for i, (key, title, desc, color) in enumerate(takeaways):
        row = i // 2
        col = i % 2
        x = 0.6 + col * 6.4
        y = 2.1 + row * 0.92
        # key badge
        badge = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                     Inches(x), Inches(y), Inches(0.65), Inches(0.45))
        badge.line.fill.background()
        badge.fill.solid()
        badge.fill.fore_color.rgb = color
        tb = s.shapes.add_textbox(Inches(x), Inches(y + 0.03), Inches(0.65), Inches(0.4))
        tf = tb.text_frame
        p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
        r = p.add_run(); r.text = key
        r.font.size = Pt(14); r.font.bold = True; r.font.name = FONT
        r.font.color.rgb = WHITE

        # title + desc
        add_rich_paragraph(s, [
            {"text": title, "size": 13, "bold": True, "color": PRIMARY},
        ], x + 0.8, y, 5.4, 0.35)
        add_paragraph(s, desc, x + 0.8, y + 0.38, 5.4, 0.55,
                       font_size=11, color=TEXT_MAIN)
    slides_meta.append(("Results",))

    # -------------------------------------------------------------------------
    # SECTION 10: CONCLUSION
    # -------------------------------------------------------------------------
    s = prs.slides.add_slide(blank)
    set_background(s)
    add_title_bar(s, "Contributions")
    add_lead(s,
             "Five concrete contributions, ordered from architectural to empirical. Together they constitute the first unified graph-fraud framework "
             "that integrates a learned RL policy with LLM-derived reasoning and validates each decision through controlled ablations.")
    contribs = [
        ("1", "Unified GNN + RL + LLM framework architected around a three-pillar, additive design."),
        ("2", "CAPN: an actor–critic replacement for CARE-GNN's heuristic bandit with per-node thresholds, shaped reward, and LLM-enriched state."),
        ("3", "Structural LLM reasoning scores (Part 1): six fraud-specific risk scores from Claude Haiku 4.5 over 26 per-node graph statistics."),
        ("4", "TextFeatureGate (Part 2): gated residual injection of text-risk scores that preserves feature dimension and is reversible."),
        ("5", "Empirical validation: +1.99 pp AUC on YelpChi with p < 0.001; nine Amazon ablations attribute each design decision."),
    ]
    for i, (num, text) in enumerate(contribs):
        y = 2.1 + i * 0.93
        # number badge
        badge = s.shapes.add_shape(MSO_SHAPE.OVAL,
                                     Inches(0.7), Inches(y), Inches(0.65), Inches(0.65))
        badge.line.fill.background()
        badge.fill.solid()
        badge.fill.fore_color.rgb = ACCENT
        tb = s.shapes.add_textbox(Inches(0.7), Inches(y + 0.08), Inches(0.65), Inches(0.5))
        tf = tb.text_frame
        p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
        r = p.add_run(); r.text = num
        r.font.size = Pt(18); r.font.bold = True; r.font.name = FONT
        r.font.color.rgb = WHITE
        # text
        add_paragraph(s, text, 1.55, y + 0.1, 11.5, 0.8,
                       font_size=15, color=TEXT_MAIN)
    slides_meta.append(("Conclusion",))

    # Slide 24: Limitations + Future work
    s = prs.slides.add_slide(blank)
    set_background(s)
    add_title_bar(s, "Limitations and Future Work")
    add_lead(s,
             "Honest scope of the thesis, and where the framework would most benefit from follow-on work. "
             "Limitations are about what was NOT done; future work is about the highest-value extensions.")

    add_paragraph(s, "Limitations", 0.6, 2.0, 6, 0.5,
                   font_size=19, bold=True, color=NEGATIVE)
    add_bullets(s, [
        "Three seeds on YelpChi give tight CIs but limited power for border-line effects (Row 3 vs Row 4).",
        "RL pillar tested only on this benchmark family; no \"in the wild\" evaluation with concept drift.",
        "LLM reasoning prompts are not tuned — the template is frozen.",
    ], 0.8, 2.55, 6.0, 3.5, font_size=13)

    add_paragraph(s, "Future Work", 6.9, 2.0, 6, 0.5,
                   font_size=19, bold=True, color=POSITIVE)
    add_bullets(s, [
        "Scale to five seeds and additional benchmarks (T-Finance, DGraph-Fin).",
        "End-to-end LLM prompt optimisation (DSPy, reward-model tuning).",
        "Temporal fraud detection: extend CAPN state with time-aware features.",
        "Compare against newer fraud-GNN baselines (GAGA, BWGNN, FLAG).",
    ], 7.1, 2.55, 6.0, 4.0, font_size=13, bullet_color=POSITIVE)
    slides_meta.append(("Conclusion",))

    # Slide 25: Thank you
    s = prs.slides.add_slide(blank)
    set_background(s, PRIMARY)
    bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0),
                              Inches(0.35), Inches(SLIDE_H_IN))
    bar.line.fill.background()
    bar.fill.solid()
    bar.fill.fore_color.rgb = ACCENT
    add_paragraph(s, "Thank you", 0.6, 2.8, 13, 1.5,
                   font_size=72, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    add_paragraph(s, "Questions?", 0.6, 4.4, 13, 0.8,
                   font_size=32, color=RGBColor(0xCC, 0xCC, 0xCC),
                   align=PP_ALIGN.CENTER, italic=True)
    slides_meta.append(("Q&A",))

    # -------------------------------------------------------------------------
    # APPENDIX SLIDES
    # -------------------------------------------------------------------------
    # Slide 26: Dataset relational stats
    s = prs.slides.add_slide(blank)
    set_background(s)
    add_title_bar(s, "Appendix: Dataset Relational Statistics",
                   "Recomputed from adjacency-list release used in this thesis")
    data = [
        ["Relation", "|E| (undir.)", "Avg. Deg.", "% Fraud Nbrs (fraud)", "% Fraud Nbrs (benign)"],
        ["Amazon — UPU", "175,608", "29.4", "10.0", "4.8"],
        ["Amazon — USU", "3,566,479", "597.2", "10.3", "3.6"],
        ["Amazon — UVU", "1,036,737", "173.6", "18.3", "3.0"],
        ["YelpChi — RUR", "49,315", "2.15", "92.5", "0.4"],
        ["YelpChi — RTR", "573,616", "24.96", "18.0", "14.0"],
        ["YelpChi — RSR", "3,402,743", "148.09", "20.5", "13.5"],
    ]
    add_table(s, data, 0.6, 1.7, 12.2, 4.2, font_size=12, bold_rows=[4])
    add_paragraph(s,
                   "RUR's near-perfect label homophily (92.5 % fraud nbrs for fraud nodes) is the dominant structural signal on YelpChi.",
                   0.6, 6.15, 12, 0.8, font_size=13, italic=True, color=TEXT_MUTED)
    slides_meta.append(("Appendix",))

    # Slide 27: Hyperparameters
    s = prs.slides.add_slide(blank)
    set_background(s)
    add_title_bar(s, "Appendix: Unified Training Hyperparameters")
    data = [
        ["Hyperparameter", "Value"],
        ["Policy-weight target  λπ*", "0.3"],
        ["Similarity loss weight  λ₁", "2"],
        ["Critic loss weight  λ_c", "0.1"],
        ["Entropy coefficient  λ_H", "0.01"],
        ["GNN warmup epochs  E_warm", "5"],
        ["Policy ramp epochs  E_ramp", "5"],
        ["Backbone LR", "1 × 10⁻²"],
        ["Policy LR", "3 × 10⁻³"],
        ["Critic LR", "1 × 10⁻³"],
        ["Backbone gradient clip (max-norm)", "1.0"],
        ["Policy / critic gradient clip (max-norm)", "5.0"],
        ["Optimiser", "Adam (two instances)"],
        ["Batch size (YelpChi / Amazon)", "1,024 / 256"],
        ["Epochs", "31"],
        ["TextFeatureGate init  g", "−2  (sigmoid ≈ 0.12)"],
    ]
    add_table(s, data, 3.0, 1.5, 7.3, 5.8, font_size=11, first_col_bold=True)
    slides_meta.append(("Appendix",))

    # -------------------------------------------------------------------------
    # Add footers
    # -------------------------------------------------------------------------
    total = len(prs.slides)
    for i, slide in enumerate(prs.slides):
        if i == 0 or i == len(list(prs.slides)) - 1:  # skip title and thank-you backgrounds
            continue
        section = slides_meta[i][0] if i < len(slides_meta) else ""
        add_footer(slide, i + 1, total, section)

    prs.save(str(OUT))
    print(f"Wrote: {OUT}  ({total} slides)")


if __name__ == "__main__":
    build()
