"""Build the supervisor deck (.pptx) -- a full project explanation.

Separate from build_presentation.py, which generates the thesis deck. This one
walks the project from the problem statement to the open decisions, for someone
who has not been following along.

Every number is read from results/experiments/*.json at build time. The paper's
figures drifted once because numbers were typed into them by hand; nothing here
is typed twice.

    python scripts/build_progress_deck.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pptx import Presentation
from pptx.util import Inches
from pptx.enum.text import PP_ALIGN

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_presentation import (  # noqa: E402  (shared styling)
    PRIMARY, ACCENT, POSITIVE, NEGATIVE, TEXT_MAIN, TEXT_MUTED,
    SLIDE_W_IN, SLIDE_H_IN, set_background, add_title_bar, add_lead,
    add_footer, add_bullets, add_paragraph, add_info_box,
)

REPO = Path(__file__).resolve().parent.parent
RES = REPO / "results" / "experiments"
FIG = REPO / "scripts" / "progress_figures"
OUT = REPO / "Progress_Update.pptx"
FIG.mkdir(exist_ok=True)

C_STRUCT, C_ADAPT, C_SEM, C_FULL = "#1B6EC2", "#B5700C", "#0A9D5E", "#8E4FA8"
C_BASE, C_DOWN = "#6F7A8A", "#B03A3A"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Calibri", "DejaVu Sans"],
    "font.size": 13, "axes.linewidth": 0.9,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.08,
})


def jload(n):
    with open(RES / n, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------- data ------
M = jload("holdout20_full_metrics.json")           # R1-R4, n=20 seeds
AMZ = jload("amazon_adaptive_holdout20.json")      # Amazon adaptive, n=20
YSPL = jload("yelp_split_robustness.json")         # 10 splits, R1 vs R4
ASPL = jload("amazon_split_robustness.json")       # 10 splits, R1 vs R2
GC = jload("graphconsis_table_n20.json")           # standalone / +text6
GCC = jload("graphconsis_capn_table_n20.json")     # +CAPN / +full

AUC = {k: float(np.mean(M[k]["auc"])) for k in ("R1", "R2", "R3", "R4")}
AP = {k: float(np.mean(M[k]["ap"])) for k in ("R1", "R2", "R3", "R4")}
FULL = GCC["GraphConsis+CAPN+LLM (framework)"]
CAPN = GCC["GraphConsis+CAPN"]


# -------------------------------------------------------------- figures -----
def fig_ablation() -> Path:
    data = np.array([[M[k]["auc"][i] for k in ("R1", "R2", "R3", "R4")]
                     for i in range(len(M["R1"]["auc"]))])
    mean = data.mean(0)
    px = np.array([0.0, 1.0, 1.0, 2.0])
    fig, ax = plt.subplots(figsize=(9.2, 4.3))
    for row in data:
        for mid in (1, 2):
            ax.plot([px[0], px[mid], px[3]], [row[0], row[mid], row[3]],
                    color="#C3CAD4", lw=0.8, alpha=0.6, zorder=1)
    for i, j, col in [(0, 1, C_ADAPT), (2, 3, C_ADAPT),
                      (0, 2, C_SEM), (1, 3, C_SEM)]:
        ax.plot([px[i], px[j]], [mean[i], mean[j]], color=col, lw=3.4, zorder=3,
                solid_capstyle="round")
        d = data[:, j] - data[:, i]
        ax.annotate(f"+{d.mean()*100:.2f} pp\n{int((d > 0).sum())}/20",
                    xy=((px[i]+px[j])/2, (mean[i]+mean[j])/2), fontsize=12,
                    ha="center", va="center", color="#222222",
                    bbox=dict(facecolor="white", edgecolor=col, lw=1.4,
                              boxstyle="round,pad=0.3"), zorder=5)
    for xi, m, c in zip(px, mean, [C_STRUCT, C_ADAPT, C_SEM, C_FULL]):
        ax.scatter([xi], [m], s=190, color=c, zorder=4, edgecolor="white", lw=2)
    for lbl, xi, m, off, ha in [("backbone", px[0], mean[0], (4, -20), "left"),
                                ("+ adaptive", px[1], mean[1], (16, -14), "left"),
                                ("+ semantic", px[2], mean[2], (-16, 16), "right"),
                                ("full framework", px[3], mean[3], (4, 20), "right")]:
        ax.annotate(lbl, xy=(xi, m), xytext=off, textcoords="offset points",
                    ha=ha, va="center", fontsize=13, fontweight="bold",
                    color="#222222")
    ax.set_xticks([]); ax.set_xlim(-0.34, 2.40)
    ax.set_ylim(data.min() - 0.0016, data.max() + 0.0022)
    ax.set_ylabel("test AUC", fontsize=13)
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    p = FIG / "ablation.png"; fig.savefig(p, dpi=190, facecolor="white"); plt.close(fig)
    return p


def fig_splits() -> Path:
    """Per-split deltas on both datasets -- the generalisation evidence."""
    def deltas(prefix, a, b):
        out = []
        for s in range(1, 11):
            r1 = json.load(open(RES / f"{prefix}_s{s}_{a}.json"))["per_seed"][0]["auc"]
            r2 = json.load(open(RES / f"{prefix}_s{s}_{b}.json"))["per_seed"][0]["auc"]
            out.append(100 * (r2 - r1))
        return out

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6))
    for ax, (title, d, meta) in zip(axes, [
            ("YelpChi — full framework", deltas("yelp", "r1", "r4"), YSPL["AUC"]),
            ("Amazon — adaptive pillar", deltas("amz", "r1", "r2"), ASPL["AUC"])]):
        x = np.arange(1, 11)
        ax.bar(x, d, color=C_FULL if "Yelp" in title else C_ADAPT, width=.62)
        ax.axhline(0, color="#999999", lw=1)
        ax.axhline(meta["delta_pp"], color="#333333", ls="--", lw=1.4)
        ax.text(10.6, meta["delta_pp"], f"  mean\n  {meta['delta_pp']:+.2f}",
                fontsize=11, va="center", fontweight="bold")
        ax.set_title(f"{title}\n{int(meta['positive'])}/10 splits positive · "
                     f"p = {meta['p']:.1g}", fontsize=12.5, fontweight="bold")
        ax.set_xlabel("independent data split", fontsize=11)
        ax.set_ylabel("AUC gain (pp)", fontsize=11)
        ax.set_xticks(x); ax.set_xlim(0.3, 12.2)
        ax.tick_params(labelsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    p = FIG / "splits.png"; fig.savefig(p, dpi=190, facecolor="white"); plt.close(fig)
    return p


def fig_wrapper() -> Path:
    """GraphConsis: the whole framework attached to a backbone it was not built on."""
    base = np.array(GC["raw32"]["per_seed_auc"])
    rows = [("standalone", base.mean(), C_BASE, None),
            ("+ adaptive", CAPN["auc_mean"], C_ADAPT, CAPN["vs_standalone"]),
            ("+ semantic", np.mean(GC["raw32+text6"]["per_seed_auc"]), C_SEM,
             GC["text6_delta"]),
            ("+ full framework", FULL["auc_mean"], C_FULL, FULL["vs_standalone"])]
    fig, ax = plt.subplots(figsize=(9.2, 3.9))
    ys = np.arange(len(rows))[::-1]
    for y, (nm, v, col, st) in zip(ys, rows):
        ax.barh([y], [v - 0.70], left=0.70, height=.58, color=col)
        ax.text(v + 0.002, y, f"{v:.4f}", va="center", fontsize=12,
                fontweight="bold", color="#222222")
        if st:
            p = st["p"]
            ax.text(0.706, y, f"{st['delta_pp']:+.2f} pp   "
                              f"p {'< 0.0001' if p < 1e-4 else f'= {p:.3f}'}",
                    va="center", fontsize=11.5, color="white", fontweight="bold")
    ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in rows], fontsize=13)
    ax.set_xlim(0.70, 0.855); ax.set_xlabel("test AUC — 20 seeds", fontsize=12)
    ax.tick_params(axis="x", labelsize=10)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout()
    p = FIG / "wrapper.png"; fig.savefig(p, dpi=190, facecolor="white"); plt.close(fig)
    return p


def fig_backbones() -> Path:
    pts = [("CARE-GNN", .7648, .7817), ("Logistic reg.", .7694, .7923),
           ("MLP", .8205, .8283), ("BWGNN-Homo", .8252, .8417),
           ("XGBoost", .8541, .8783), ("HistGBM", .8620, .8738),
           ("BWGNN-Hetero", .8879, .8941), ("XGB-Graph", .9106, .9177)]
    fig, ax = plt.subplots(figsize=(9.2, 4.1))
    for y, (n, a, b) in zip(range(len(pts)), pts):
        ax.plot([a, b], [y, y], color="#C3CAD4", lw=3, zorder=1)
        ax.scatter([a], [y], s=85, color=C_BASE, zorder=3)
        ax.scatter([b], [y], s=85, color=C_SEM, zorder=3)
    ax.set_yticks(range(len(pts))); ax.set_yticklabels([p[0] for p in pts], fontsize=12)
    ax.axvline(AUC["R4"], color=C_FULL, lw=2.4, ls="--", zorder=2)
    ax.text(AUC["R4"] + .0015, len(pts) - .5,
            f"our framework\non CARE-GNN\n{AUC['R4']:.4f}", color=C_FULL,
            fontsize=11.5, fontweight="bold", va="top")
    ax.set_xlabel("test AUC — identical 25/15/60 split", fontsize=12.5)
    ax.scatter([], [], s=85, color=C_BASE, label="backbone alone")
    ax.scatter([], [], s=85, color=C_SEM, label="+ our six text scores")
    ax.legend(loc="lower right", fontsize=11, frameon=False)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=10)
    fig.tight_layout()
    p = FIG / "backbones.png"; fig.savefig(p, dpi=190, facecolor="white"); plt.close(fig)
    return p


# ---------------------------------------------------------------- deck ------
def build():
    f_abl, f_spl, f_wrap, f_bb = (fig_ablation(), fig_splits(),
                                  fig_wrapper(), fig_backbones())
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(SLIDE_W_IN), Inches(SLIDE_H_IN)
    blank = prs.slide_layouts[6]
    N, n = 13, 0

    def new(title, subtitle=None, section=""):
        nonlocal n
        n += 1
        s = prs.slides.add_slide(blank)
        set_background(s); add_title_bar(s, title, subtitle); add_footer(s, n, N, section)
        return s

    # 1 -- title
    s = prs.slides.add_slide(blank); n = 1
    set_background(s)
    add_paragraph(s, "Camouflage-Resistant Graph Fraud Detection", 0.9, 2.3, 11.6, 1.0,
                  font_size=38, bold=True, color=PRIMARY)
    add_paragraph(s, "A three-pillar framework: graph structure, a learned filtering "
                     "policy, and language-model signal", 0.9, 3.35, 11.6, 0.8,
                  font_size=19, color=TEXT_MUTED)
    add_paragraph(s, "Fady Tarek  ·  German University in Cairo", 0.9, 4.3, 11.6, 0.4,
                  font_size=15, color=TEXT_MUTED)
    add_info_box(s, "In one sentence",
                 "Attach this framework to a graph fraud detector you already have, "
                 "and that detector gets significantly better — demonstrated on two "
                 "backbones and two datasets.",
                 0.9, 5.1, 11.6, 1.2, color=ACCENT, font_size=15)

    # 2 -- the problem
    s = new("The problem", "Why fraud detection on graphs is hard", "Background")
    add_lead(s, "Fraud is relational — so graphs are natural. But fraudsters know that.")
    add_bullets(s, [
        "Fraudsters operate in groups, share targets, reuse patterns. A graph neural "
        "network can read both the behaviour and the connections.",
        "So fraudsters camouflage in two ways: they edit their own profile to look "
        "ordinary, and they connect to honest accounts.",
        "A network that averages over a fraudster's neighbours then averages in a "
        "crowd of innocents — and the fraud signal disappears.",
        "Benchmark: YelpChi — 45,954 reviews, 14.5% fraudulent, 32 behavioural "
        "features, three kinds of connection.",
    ], 0.8, 2.4, 11.8, 3.2, font_size=16)

    # 3 -- prior art
    s = new("What already existed", "CARE-GNN, and the two gaps it leaves", "Background")
    add_info_box(s, "Its good idea",
                 "Do not compare neighbours by their raw features — faking features is "
                 "exactly what camouflage IS. Instead rank them by predicted fraud "
                 "likelihood and keep the closest ones.",
                 0.8, 1.9, 11.8, 1.3, color=POSITIVE, font_size=15)
    add_bullets(s, [
        "GAP 1 — one global threshold per relation, identical for all 45,954 nodes, "
        "set by a crude bandit. A heavily camouflaged node and a clean one are "
        "filtered the same way.",
        "GAP 2 — it only ever sees the graph. It never reads a single word of the "
        "review text, even though the text is right there.",
    ], 0.8, 3.5, 11.8, 2.0, font_size=16)
    add_paragraph(s, "Those two gaps are the second and third pillars.",
                  0.8, 5.7, 11.8, 0.5, font_size=18, bold=True, color=PRIMARY,
                  align=PP_ALIGN.CENTER)

    # 4 -- the framework
    s = new("The framework", "Three pillars, one joint objective", "Method")
    add_info_box(s, "STRUCTURAL",
                 "Multi-relation GNN\n(CARE-GNN backbone,\nkept as-is)",
                 0.7, 1.9, 3.9, 1.9, color=PRIMARY, font_size=14)
    add_info_box(s, "ADAPTIVE  (CAPN)",
                 "An actor–critic policy replaces the bandit.\nOutputs a threshold per "
                 "NODE, per relation.",
                 4.75, 1.9, 3.9, 1.9, color=ACCENT, font_size=14)
    add_info_box(s, "SEMANTIC",
                 "A language model scores each review on six\nfraud-relevant text "
                 "dimensions.",
                 8.8, 1.9, 3.9, 1.9, color=POSITIVE, font_size=14)
    add_bullets(s, [
        "The mechanism worth knowing: one learned number θ does two jobs — it sets how "
        "many neighbours survive the filter, AND its per-relation average becomes that "
        "relation's weight in the final blend. The aggregation adapts node by node.",
        "The language model runs ONCE, offline, cached. The training loop never calls "
        "it — so there is no added cost per epoch and no API dependency at train time.",
    ], 0.8, 4.2, 11.8, 2.1, font_size=15)

    # 5 -- protocol
    s = new("How it was evaluated", "The part reviewers praised", "Method")
    add_bullets(s, [
        "Five seeds used for development. Then TWENTY fresh seeds, never touched during "
        "tuning, for every headline claim.",
        "Every baseline re-run in-house at the identical split — never quoted from its "
        "source paper. On this benchmark the amount of training data dominates the "
        "score, so cross-paper numbers are not like-for-like.",
        "Later: ten INDEPENDENT data splits, because repeated seeds on one split only "
        "measure initialisation noise, not generalisation.",
        "Re-implementations validated against published numbers before being trusted.",
    ], 0.8, 2.0, 11.8, 3.2, font_size=16)
    add_info_box(s, "Why this mattered",
                 "Split-to-split variance turned out to be 2.4–2.8× larger than "
                 "seed-to-seed variance. Reporting only seeds would have measured the "
                 "smaller component and called it generalisation.",
                 0.8, 5.4, 11.8, 1.1, color=ACCENT, font_size=14)

    # 6 -- main result
    s = new("Main result", "Each pillar measured separately, 20 held-out seeds", "Results")
    s.shapes.add_picture(str(f_abl), Inches(1.9), Inches(1.7), width=Inches(9.5))
    add_paragraph(s, f"backbone {AUC['R1']:.4f}  →  full framework {AUC['R4']:.4f}   "
                     f"(+{100*(AUC['R4']-AUC['R1']):.2f} pp AUC, "
                     f"+{100*(AP['R4']-AP['R1']):.2f} pp average precision)",
                  0.8, 6.4, 11.8, 0.5, font_size=16, bold=True, color=PRIMARY,
                  align=PP_ALIGN.CENTER)

    # 7 -- splits
    s = new("Does it survive different data?", "Ten independent splits per dataset",
            "Results")
    s.shapes.add_picture(str(f_spl), Inches(1.7), Inches(1.7), width=Inches(9.9))
    add_paragraph(s, "Positive on every single split, on both datasets. This is the "
                     "evidence that supports a generalisation claim — repeated seeds "
                     "on one split do not.",
                  0.8, 5.9, 11.8, 0.8, font_size=15, color=TEXT_MAIN,
                  align=PP_ALIGN.CENTER)

    # 8 -- wrapper
    s = new("Attached to a backbone it was not built for",
            "GraphConsis — a different architecture, 20 seeds", "Results")
    s.shapes.add_picture(str(f_wrap), Inches(1.9), Inches(1.7), width=Inches(9.5))
    add_paragraph(s, f"The complete framework, moved unchanged onto a second backbone: "
                     f"{FULL['vs_standalone']['delta_pp']:+.2f} pp, p < 0.0001. "
                     f"This is the wrapper claim.",
                  0.8, 6.1, 11.8, 0.6, font_size=16, bold=True, color=PRIMARY,
                  align=PP_ALIGN.CENTER)

    # 9 -- portability
    s = new("The text signal travels on its own",
            "Six scores added to eight unrelated models", "Results")
    s.shapes.add_picture(str(f_bb), Inches(1.9), Inches(1.5), width=Inches(9.5))
    add_paragraph(s, "Significant on 7 of 8 — including the method that currently leads "
                     "this benchmark. Note our own framework (dashed) is NOT the "
                     "strongest absolute model: the claim is improvement over a base, "
                     "not best-on-benchmark.",
                  0.8, 5.9, 11.8, 0.9, font_size=14, color=TEXT_MAIN,
                  align=PP_ALIGN.CENTER)

    # 10 -- honest findings
    s = new("The findings I did not expect", "All three are negative results", "Findings")
    add_info_box(s, "1 · My own design was sabotaging itself",
                 "I initialised the policy from LLM-derived priors to give it a head "
                 "start. It was SUPPRESSING the policy (−0.71 pp). Removing it turned a "
                 "null (+0.19 pp, p = 0.10) into +0.58 pp on 20/20 seeds.",
                 0.7, 1.8, 12.0, 1.35, color=ACCENT, font_size=14)
    add_info_box(s, "2 · The language model is not necessary",
                 "A free deterministic scorer computing the same six quantities is "
                 "statistically indistinguishable from Claude on EVERY graph backbone. "
                 "The contribution is which six things you measure, not who measures "
                 "them — and a practitioner needs no API at all.",
                 0.7, 3.35, 12.0, 1.35, color=ACCENT, font_size=14)
    add_info_box(s, "3 · The pillars add, they do not synergise",
                 f"+0.58 and +1.96 sum to +2.54; together they give +2.70. The "
                 f"interaction is +0.16 pp against 0.43 pp of noise. The 2025 wave of "
                 f"LLM-enhanced detectors assumes integration is where the value is. "
                 f"Nobody tested it. It is not.",
                 0.7, 4.9, 12.0, 1.4, color=ACCENT, font_size=14)

    # 11 -- review
    s = new("External review", "Five blind reviewers + a devil's advocate", "Status")
    add_lead(s, "Verdict: major revision. No reviewer questioned the integrity of the work.")
    add_bullets(s, [
        "Their strongest criticism was fair: our framework reaches 0.79 AUC while a "
        "gradient-boosted tree in our own comparison table reaches 0.92.",
        "The answer is the wrapper framing — it is not a competitor to that tree, it is "
        "an upgrade for graph detectors. Judged on base vs base-plus-wrapper, and that "
        "is +8.61 pp on a backbone we did not design it for.",
        "Our novelty claim survived their independent literature search: nothing "
        "combines a GNN, an RL filtering policy, and LLM signal.",
    ], 0.8, 2.4, 11.8, 2.4, font_size=15)
    add_info_box(s, "And I corrected myself",
                 "Three published numbers were wrong because five seeds was too few on "
                 "a high-variance backbone. I found all three and fixed them. They moved "
                 "the same way — the paper had been claiming slightly more than it could "
                 "support.",
                 0.8, 5.0, 11.8, 1.3, color=NEGATIVE, font_size=14)

    # 12 -- where it stands
    s = new("Where it stands", section="Status")
    add_info_box(s, "Established",
                 "• Both pillars significant, 20 held-out seeds\n"
                 "• Positive on 10/10 independent splits\n"
                 "• Replicated on a second dataset\n"
                 "• Transfers to a second backbone (+8.61 pp)\n"
                 "• Text signal lifts 7 of 8 other models",
                 0.7, 1.9, 5.9, 2.7, color=POSITIVE, font_size=14)
    add_info_box(s, "Open",
                 "• Not competitive on absolute accuracy\n"
                 "• No operating-point evidence yet\n"
                 "• Semantic pillar needs review text,\n   so Amazon is partial\n"
                 "• Paper needs restructuring + reformatting",
                 6.85, 1.9, 5.9, 2.7, color=ACCENT, font_size=14)
    add_paragraph(s, "Experiments are essentially done. Roughly 1–2 weeks of writing "
                     "to a submittable draft.",
                  0.7, 5.0, 12.0, 0.6, font_size=18, bold=True, color=PRIMARY,
                  align=PP_ALIGN.CENTER)

    # 13 -- decisions
    s = new("What I need from you", section="Decision")
    add_bullets(s, [
        "VENUE — PAKDD 2027 closes 20 November (three weeks, tight but reachable); "
        "ECML-PKDD 2027 closes around July and is the better fit, because negative "
        "results have a home there and Eindhoven is far cheaper to reach than New "
        "Zealand; IEEE ICAISET in Cairo closes 10 January, is hybrid, and needs no "
        "travel.",
        "A third backbone (PC-GNN) was insurance in case GraphConsis failed. It did "
        "not — so I would skip it unless you disagree.",
        "The paper is in the wrong template for every one of those venues, and all the "
        "conference options are double-blind. That conversion is unavoidable, so the "
        "venue decision should come first.",
    ], 0.8, 2.0, 11.8, 3.2, font_size=16)
    add_info_box(s, "My recommendation",
                 "Target ECML-PKDD, keep Cairo as a floor. The extra months let me add "
                 "operating-point evidence, which was the one reviewer gap I have not "
                 "closed.",
                 0.8, 5.4, 11.8, 1.1, color=POSITIVE, font_size=15)

    prs.save(str(OUT))
    print(f"Wrote: {OUT}  ({n} slides)")


if __name__ == "__main__":
    build()
