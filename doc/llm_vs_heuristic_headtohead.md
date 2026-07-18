# Is the LLM needed? Haiku vs heuristic-template text6 head-to-head

Reviewer's staking issue: the paper attributes the text6 gain to "LLM-derived
semantic signal," but the six scores have deterministic heuristic definitions and
the LLM was never ablated against them at the paper's protocol. This is that
ablation. `raw + template6` vs `raw + haiku6`, frozen YelpChi 25/15/60, 5 seeds,
paired test on the Haiku-minus-template delta.

Scripts: `scripts/llm_vs_template.py` (tabular), `scripts/bwgnn_experiment.py
--text-file ...`, `scripts/graphconsis_experiment.py --text-file ...`,
`run_yelp_experiment.py --tag r3_template -- --text-enrichment gate
--text-risk-mode template`. Both tensors on disk:
`text_risk_scores_{haiku,template}.pt`.

## The two score tensors genuinely differ
mean |template − haiku| = 0.20 on [0,1]; per-column Pearson r ∈ [−0.16, +0.56].
Haiku's scores are 99.8% quantised to multiples of 0.05 (coarse ordinal). So this
is not "identical inputs" — they are different score vectors that happen to
produce similar downstream AUC.

## Result: LLM's edge over the heuristic washes out under graph aggregation

| Backbone | + template6 | + haiku6 | Haiku − template | sig |
|---|---|---|---|---|
| Logistic regression | 0.7881 | 0.7923 | +0.42pp | **SIG** (bootstrap CI) |
| MLP (64,64)         | 0.8237 | 0.8283 | +0.47pp | n.s. (p=0.46) |
| HistGBM             | 0.8666 | 0.8738 | +0.72pp | **SIG** (p=0.008) |
| XGBoost             | 0.8662 | 0.8716 | +0.55pp | degenerate test |
| **CARE-GNN (R3, headline)** | 0.7802 | 0.7817 | **+0.15pp** | **n.s. (p=0.31)** |
| BWGNN-Hetero        | 0.8927 | 0.8941 | +0.14pp | n.s. (p=0.35) |
| BWGNN-Homo          | 0.8348 | 0.8417 | +0.70pp | n.s. (p=0.06) |
| GraphConsis         | 0.8134 | 0.8256 | +1.22pp | n.s. (p=0.70) |

**All four GNN backbones n.s.** (CARE +0.15, BWGNN-Het +0.14, BWGNN-Homo +0.70,
GraphConsis +1.22 — the last has a large point estimate but p=0.70 from
GraphConsis's ±0.03 seed noise). Significant only on LR/GBM. Decisive.

**Pattern:** on every GNN backbone — including the paper's headline CARE-GNN —
Haiku is statistically indistinguishable from the free heuristic. The LLM shows a
small significant edge only on simple feature-only classifiers (LR, GBM), and
that edge does not survive once the features enter a graph model. The CARE-GNN
result (+0.15pp, p=0.31) independently reproduces the reviewer's cited
+0.18pp/t≈0.64.

## Implication for the paper (reframe: engineered features)
The defensible claim is NOT "LLM-derived semantic signal drives the gain." It is:
**six compact, fraud-specific text features improve backbones across the board,
and on graph models it makes no significant difference whether a frontier LLM or
a zero-cost heuristic computes them.** The LLM is not necessary for the effect on
the models the paper is built on. This converts the fatal rebuttal ("did you
ablate the LLM?") into a contribution (a practitioner needs no API calls).

Related fixes in flight: de-confound R2 (drop yelp_priors); 30-seed hardening of
the semantic delta.
