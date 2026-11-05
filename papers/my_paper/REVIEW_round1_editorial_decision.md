# Editorial Decision Letter — Round 1

**Manuscript:** A Three-Pillar Framework Unifying Structural, Adaptive, and Semantic Reasoning for Graph Fraud Detection
**Panel:** 5 role-separated seats (Journal-Fit, Methodology, Domain, Perspective, Devil's Advocate), committed blind
**Date:** 2026-10-23 · Round 1
**Calibration status:** `NOT_CALIBRATED` · **Criteria binding:** `criteria_binding_unavailable` (no venue fixed)

---

## Decision: MAJOR REVISION

**Seat recommendations:** Journal-Fit *Major Revision* · Methodology *Major Revision* · Perspective *Major Revision* · Domain **Reject** (explicitly "reject this framing; I would willingly review a reframed paper built on the same experiments") · Devil's Advocate 3 CRITICAL findings.

Four of five seats converge on Major Revision. The dissenting Reject is not a judgement that the work is weak — that seat states plainly that the experimental programme contains "at least three results the field should have." It is a judgement that one finding (R1 below) cannot be repaired by rewriting alone. That constraint is adopted into the roadmap below rather than overridden.

**No seat questioned the integrity of the work.** Every seat independently praised the null-reporting discipline. The Devil's Advocate noted that two of its own critical findings were *constructible only because the authors reported the damaging results themselves* — and warned that a revision must not respond by making the paper quieter.

---

## Devil's Advocate CRITICAL adjudication (Iron Rule #4)

| # | Finding | Adjudication |
|---|---|---|
| **C1** | Necessity-of-unification thesis vs. pure additivity in Table 1 | **VALIDATED.** Verified against the paper's own means: R2−R1 = +0.58, R3−R1 = +1.96, sum = +2.54, joint R4−R1 = +2.70 → **interaction ≈ +0.16 pp against R3's own seed std of 0.43 pp.** Independently corroborated by the Domain seat, which reached the same conclusion by a different route ("the composition result is in fact the signature of *independence*"). Blocks any Accept. |
| **C2** | The architecture claimed as novel is not the architecture evaluated | **REFUTED AS TO THE CODE; VALIDATED AS TO THE MANUSCRIPT.** The R4 run flags are `--use-capn --use-actor-critic --text-enrichment gate --text-state-enrichment` — R4 **does** retain the scores-in-state pathway. But two independent expert seats read the submitted paper and concluded the opposite, from Figure 1's "ablated couplings … scores-in-state ≈ 0" banner and §4.2's "these are the clean configurations Table 1 reports." A manuscript that causes two expert readers to misidentify its own headline configuration has a defect of the first order. Recorded with rationale; does not veto on its own. |
| **C3** | All inference conditioned on one fixed split | **VALIDATED.** Independently and identically found by the Methodology seat (its W1, also rated Critical). Blocks any Accept. |

---

## Consensus findings (ranked by corroboration)

### F1 — The framework is outperformed by the paper's own baselines, and the comparison is never drawn
**4 of 5 seats** (Journal-Fit *Critical*, Domain *Critical*, Perspective *Major*, DA *Major*). The Methodology seat did not assess it (out of remit).

At the identical frozen split: R4 = **0.7909**; logistic regression + text6 = **0.7923**; MLP = 0.8283; XGBoost = 0.8783; BWGNN-Hetero = 0.8941; XGB-Graph = **0.9177**. Tables 1 and 2 never appear together, and §5 asserts "The framework delivers."

Two seats called this the single most consequential defect. The Perspective seat put the practitioner reading bluntly: *"build a gradient-boosted tree on two-hop aggregated features plus six text scores, skip the GNN, skip the RL policy, skip the LLM."* The Domain seat noted the paper borrows XGB-Graph's benchmark-leading status rhetorically while never admitting XGB-Graph alone far exceeds the framework.

### F2 — Every p-value estimates seed variance; the claims are stated over data
**3 of 5 seats** (Methodology *Critical*, DA *CRITICAL*, Perspective *Major* via the temporal-split angle).

`train.py` fixes `random_state=2`. The twenty "held-out seeds" resample initialisation on an identical partition. The Methodology seat: *"no reported p-value estimates the quantity the paper's central claims are about."* The DA added that the paper applies the field's split-variance standard to its baselines (§4.1's refusal to quote cross-ratio numbers) and then exempts itself.

### F3 — The dose–response is a statistical artifact and is used as load-bearing interpretation
**5 of 5 seats.** Unanimous.

Three independent objections: (i) Δ = AUC(with) − AUC(without) is regressed on AUC(without), so the same noisy quantity appears on both axes with opposite signs — this produces a negative slope *under the null*, compounded by a ceiling effect (Methodology, DA); (ii) the fit uses nine points while §4.3 says eight, the ninth being GraphConsis from a different experiment, non-significant, with the largest residual and ~10× the seed std (Journal-Fit, Domain, DA); (iii) no slope CI, R², or n is reported. The slope is then used to explain away the Amazon null.

### F4 — The unification claim is not supported by what the evidence attributes
**3 of 5 seats** (Domain *Critical*, DA *CRITICAL*, Journal-Fit *Major*).

What is uniquely attributable to unification is the +0.74 pp composition edge on one host. The semantic signal transfers to eight unrelated backbones by plain concatenation; the LLM is replaceable by a free heuristic; the adaptive pillar is undetectable on two of three hosts.

**Novelty check result (Domain seat, independent 2024–2026 search):** the literal claim **survives** — no work was found combining a GNN backbone, an RL-learned filtering policy, and LLM-derived semantic signal. Nearest neighbours each have exactly two of three. But the seat judges the claim "hollowed out": being first to combine three things is a contribution only if the combination does something, and the paper's own measurements say the interfaces contribute nothing.

### F5 — No metric at any deployable operating point
**4 of 5 seats.** F1-macro is flat and explicitly disclaimed; no precision@k, no recall at fixed FPR, no cost curve replaces it. The Perspective seat: a ranking gain concentrated in mid-ranks "is worth nothing to a queue that only ever sees the top few percent."

### F6 — Single-dataset evidence base
**3 of 5 seats.** *Partially answered since submission:* the Amazon R1-vs-R2 ablation now exists — **+0.51 pp AUC, 19/20 seeds, p = 6.4×10⁻⁷** (n = 20 held-out seeds, identical protocol). The DA had called this "the single most decisive generality test for the paper's one architecturally novel component." Note the same runs show **F1-macro −1.43 pp, 0/20 seeds** on Amazon, which sharpens F5 rather than resolving it.

### F7 — Camouflage is the organising concept and is never measured
**2 of 5 seats** (Domain *Major*, DA *Major*). No stratification by camouflage degree, no analysis of which edges the thresholds prune, no synthetic-camouflage injection. CARE-GNN itself operationalises both camouflage types with dataset-level statistics, so the lineage supplies a measurement precedent.

### F8 — No cost, reproducibility, or artifact statement
**4 of 5 seats.** No code/data availability, no hyperparameter values for λ₁/λ_π/λ_c/w₁₋₃, no tuning-parity statement, no LLM scoring cost, no serving-path analysis, no bootstrap specification. The DA added that **the six text scores are never fully enumerated anywhere in the paper** (§3.4 names three, §4.1 names three and says "similar cues") and the heuristic scorer is never defined — making the paper's most portable claim unreproducible precisely where it matters most.

### F9 — Statistical reporting
**Methodology seat, decision-bearing.** No multiplicity correction across ~50 tests while the abstract's headline is a *count* of significant results; no test statistics, df, seed counts, or CIs anywhere; two incompatible procedures (paired-t over seeds vs. bootstrap over nodes) pooled under one star notation and summed into "seven of eight"; `p < 10⁻¹⁶` from n = 20 exceeds what twenty paired observations can support (the sign-test floor is ~10⁻⁶).

### F10 — Asymmetric hedging
**3 of 5 seats.** §4.4 calls +4.16 pp at **p = 0.07** "decisive" while calling +1.20 pp at p = 0.63 a failure — same host, same seeds, same split.

### F11 — Baseline validation asymmetry
**2 of 5 seats.** BWGNN is validated against published numbers and matches — the right standard. GraphConsis is validated by *exceeding* published reproductions by 13–21 pp with no diagnosis. XGB-Graph, which carries the abstract's strongest generality claim, is never reference-validated at all while being described as "the method that currently leads this benchmark."

---

## Unresolved disagreement between seats

**Journal-Fit vs. Domain on disposition.** Journal-Fit holds that F1/F4 are repairable by rewriting alone at no experimental cost (reframe around the portable signal). Domain holds they are not: its route (a) removes the stated novelty and route (b) requires building a cross-pillar interface that currently does not work. *Arbitration:* Domain is correct that rewriting cannot make the unification claim true; Journal-Fit is correct that a reframed paper needs no new experiments. Both are satisfied by retiring the unification framing rather than defending it — reflected in R1 below.

---

## Alternative explanations the panel raised that the paper does not address

1. **Label provenance** (Perspective, DA). YelpChi labels are Yelp's own filter output, not adjudicated truth. If that filter keys on linguistic-style regularities, text6 may be recovering *the labelling system's heuristics* — which would also explain why a crude heuristic matches an LLM. The flattering reading is currently presented as the only one.
2. **Feature-count / regularisation** (DA M11). No generic text baseline (TF-IDF, embeddings) and no random-six-column null control. "Adding roughly any text signal to a 32-dim space helps" fits the data equally well.
3. **Neighbourhood sparsification, not camouflage awareness** (DA M12). No non-learned per-node threshold control. The comparison confounds granularity, stochasticity, and learning.
4. **Input-perturbation regularisation** (DA). The stabilisation effect may be the gate's small learned perturbation, independent of `t_v` content — testable by routing random values through the same gate.
5. **Checkpoint-selection noise** (DA). Best-val-AUC selection over epochs on a 15% partition carries an optimistic bias of the same order as the sub-1-pp effects claimed.

---

## Revision Roadmap

*Source-ordered, not work-ordered. No priority is inferred; the author adjudicates each item.*

### R1 — Resolve the contribution claim (F1, F4; DA C1, C2)
The panel's binding constraint. Either **(a)** retire the unification framing and rebuild the paper around what the evidence attributes — the portable six-score signal, the LLM-replaceability result, and the suppressive-coupling finding (no new experiments; Journal-Fit's preferred route), or **(b)** keep the claim and build a cross-pillar interface that beats the sum of the independent branches (new work; Domain's route). Route (a) additionally requires retitling and inverting the abstract's order.
**Mandatory regardless of route:** state explicitly in §3 and §4.2 which couplings are present in R4. Two expert seats misread this.

### R2 — Add the comparison the paper omits (F1)
One table placing R1–R4 beside XGBoost, HistGBM, BWGNN-Homo/Hetero and XGB-Graph at the frozen split, on AUC *and* AP. Then state the honest reading in a Discussion section. Retire "The framework delivers."

### R3 — Fix the inference layer (F2, F9)
Re-run headline comparisons over ≥10 independent splits; report the across-split paired CI as primary and the seed test as a secondary reproducibility statement. Declare alpha; apply Holm/BH within each test family; report *t*(df), exact p, paired mean difference and 95% CI; separate the two significance procedures into labelled columns and stop summing across them.

### R4 — Repair or retire the dose–response (F3)
Re-estimate free of the shared-error artifact (regress on a baseline from disjoint seeds, or model gain as a fraction of remaining headroom); report slope CI and n; show the fit with and without GraphConsis; disclose the fit's composition in the caption. If it does not survive, soften to a qualitative statement.

### R5 — Add operating-point evidence (F5)
From existing score vectors: precision@{1,5,10}%, recall at FPR ∈ {0.1,1,5}%, partial AUC at FPR ≤ 0.05, and F1 at a *validation-tuned* threshold rather than a fixed 0.5 cut. Report for R1–R4 across the held-out seeds, and on Amazon where F1 moved −1.43 pp.

### R6 — Fold in the Amazon results (F6)
Tabulate R1 vs R2 on Amazon (n = 20: AUC +0.51 pp, 19/20, p = 6.4×10⁻⁷; AP +0.48 pp, 18/20; F1 −1.43 pp, 0/20). Report the seed-variance reduction (±0.0027 → ±0.0011). State explicitly that the Amazon scope limit applies to the *textual* pathway only.

### R7 — Measure camouflage, or stop organising the paper around it (F7)
Stratify test nodes by a CARE-GNN-style camouflage statistic and report per-stratum AUC/AP for R1–R4. If the gain is uniform across strata, adopt the simpler account the panel offers.

### R8 — Add the missing controls (panel alternatives 2–4)
A generic text representation (TF-IDF or sentence embeddings), a random/shuffled six-column null through the same gate, and a non-learned per-node threshold control.

### R9 — Diagnostics for the central mechanism (Perspective W5)
θ distribution at convergence per relation: mean, spread, across-node SD, saturation fraction. If θ is near-constant within a relation, the adaptive pillar's interpretation changes materially. Values are already in the training logs.

### R10 — Reproducibility and cost (F8)
Enumerate all six score definitions and the heuristic scorer formulae. Add a hyperparameter table with tuning-parity statement, a code/data availability statement, bootstrap specification, and a cost ledger (one-time scoring cost, per-epoch overhead, serving path, parameter counts).

### R11 — Baseline validation parity (F11)
Validate XGB-Graph against the published GADBench number as BWGNN was validated, or soften "currently leads this benchmark." Diagnose the GraphConsis divergence or rename the host.

### R12 — Literature and attribution
Cite Rayana & Akoglu (KDD 2015) as the YelpChi source and state where raw review text was obtained. Engage the heterophily/spectral account on its merits — §2.1 dismisses it on a supervision-ratio argument the paper's own BWGNN run (0.8879 at *this* split) contradicts. Correct ref [18], which is a GNN survey cited for an RL state-representation claim. Discuss the semantic-camouflage line (CAMERA, IJCAI 2026) as the boundary condition on a text-cue defence. Note L2IR (arXiv:2605.26040) and the LLM-GNN soft-prompt line (arXiv:2605.28524) as likely-concurrent work that narrows §2.3's exclusivity statement.

### R13 — Scoped claims and copyedits
Scope "every pillar … held-out-confirmed" to its host. Apply one alpha uniformly — remove "decisive" from any p > alpha result. Replace "frontier language model (Claude Haiku)" with the accurate tier. Fix Figure 2's caption cross-reference (§4.2 → §3.3). Flag Figure 3's axis truncation. Add a consolidated limitations section and a false-positive/recourse paragraph. Replace the CIKM header and the literal "X" page markers; anonymise if targeting a double-blind venue.

---

## Routing guidance (Journal-Fit seat)

Route to **ECML-PKDD** (first choice) or **PAKDD**. Rationale: the paper's most valuable results are deflationary, and that has a home at those venues but not at a method-performance journal. **ICDM short** is a real alternative — §4.3 alone (zero-cost scorer matches the LLM, with the dose–response characterisation) is a tight self-contained short paper that sidesteps F1 and F4 entirely by not claiming a framework. **Not recommended** for ESWA or KBS in current shape: single-dataset positive evidence plus a method trailing its own baselines draws desk-level scepticism. Applied Intelligence is the most tolerant of the three journals if a journal is required, after R2, R5 and R10 are closed.
