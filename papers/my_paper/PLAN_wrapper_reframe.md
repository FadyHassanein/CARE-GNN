# Plan — reframe as a backbone wrapper, and close the evidence gaps

**Thesis of the revised paper:**
> The three-pillar framework is a *wrapper* that upgrades selection-based GNN fraud detectors.
> Attach it to a detector you already have and it gets better. It is not a competitor to XGB-Graph;
> it is an upgrade path for GNN detectors, with a stated scope limit.

**Why this framing:** it defuses the panel's biggest finding (4/5 seats: "the framework loses to
your own baselines"). A wrapper is judged on *base vs. base+wrapper*, not on absolute rank.
Section 4.4 already is this paper — it is just buried as a secondary check.

**Scope limit, stated up front:** CAPN needs a neighbour-selection step to actuate on and gradients
to learn through. That excludes XGB-Graph (no gradients) and BWGNN/GHRN (spectral, no neighbour
selection). The claim is **"any selection-based GNN detector,"** not "any model." The semantic half
travels further, which is what Table 2 now demonstrates.

---

## Evidence inventory

### Have (strong)
| Host | Dataset | Result | Status |
|---|---|---|---|
| CARE-GNN | YelpChi | base 0.7639 → full 0.7909, **+2.70 pp**, 20/20, p=2.4e-17 | solid |
| CARE-GNN | Amazon | base 0.9378 → +adaptive 0.9429, **+0.51 pp**, 19/20, p=6.4e-7 | solid (adaptive only; no text on Amazon) |
| 8 backbones | YelpChi | text6 improves 7 of 8 | solid — now reframed as *scope boundary*, not competition |

### Have (weak — this is the obstacle)
| Host | Result | Problem |
|---|---|---|
| GraphConsis | standalone 0.7840±0.0357 · +CAPN +1.20 (p=0.63) · +text6 +4.16 (p=0.07) · +full +2.71 (p=0.38) | **all n.s. at n=5**, and +full (0.8110) lands below +text6 alone (0.8256) |

Diagnosis: a *power* problem, not a null. Standalone seed-std is ±0.0357 while +text6 is ±0.0100 —
the largest text6 effect in the paper fails significance because the **baseline** is noisy. n=20
should resolve it.

### Missing
- A third host (two hosts is thin for "any selection-based GNN")
- Split-level evidence (every p-value is seed-only on one frozen split — `train.py:268`, `random_state=2`)

---

## Decisions needed from you

**D1 — Confirm the wrapper reframe.** Everything below assumes it.

**D2 — Third host: PC-GNN, yes or no?** ~2–3 days of implementation (scoped in
`doc/framework_improvement_plan.md`, July). Three hosts makes "selection-based GNNs" credible;
two makes it a pair of case studies. *Recommendation: yes if the timeline allows — it is the
difference between a generality claim and an anecdote.*

**D3 — Venue.** Affects template and length. Panel routing: ECML-PKDD (1st), PAKDD (2nd),
ICDM short (viable alternative). Not ESWA/KBS in current shape.

---

## Phase 1 — Start the long runs (tonight, unattended)

GPU is 6 GB → strictly serial. GraphConsis is CPU-only → runs in parallel.

| # | Job | Device | Est. | Why |
|---|---|---|---|---|
| 1.1 | **GraphConsis n=20**: standalone vs. +full framework | CPU | ~13 h | Decides whether the wrapper claim has one host or two. Highest value single job. |
| 1.2 | **Amazon split-robustness**: 10 splits × (R1, R2) | GPU | ~2.5 h | Cheap now (Jaccard fix: 11.8×). First split-level evidence in the project. |
| 1.3 | **Yelp split-robustness**: 10 splits × (R1, R4) | GPU | ~18 h | The headline claim gets split-level evidence. |

**Design note on 1.2/1.3:** vary `random_state` in the split, 1 seed each → 10 paired observations
*across splits*. Report the across-split paired CI as the primary inferential statistic; keep the
seed-level test as a secondary reproducibility statement. R2/R3 attribution stays seed-only with
that limitation stated explicitly, rather than silently.

**If 1.1 comes back still non-significant:** the wrapper claim rests on one host (CARE-GNN × 2
datasets). That is reportable but much weaker, and D2 becomes mandatory rather than optional.

---

## Phase 2 — Safe fixes (no rework risk, correct under any framing)

Can be done while Phase 1 runs.

1. **R4 configuration sentence.** Two expert reviewers concluded R4 lacks the semantic→adaptive
   interface, from Figure 1's "scores-in-state ≈ 0" banner and §4.2's "these are the clean
   configurations." The run flags show R4 *does* include `--text-state-enrichment`. State
   explicitly which couplings each of R1–R4 contains.
2. **Enumerate all six text scores** and the heuristic scorer formulae. They currently appear
   nowhere in the paper — §3.4 names three, §4.1 names three and says "similar cues."
3. **Citations:** add Rayana & Akoglu (KDD 2015) as the YelpChi source (and where raw review text
   comes from); fix ref [18] (a GNN survey cited for an RL state-representation claim); add CAMERA
   (IJCAI 2026) on semantic camouflage; note L2IR (arXiv:2605.26040) and arXiv:2605.28524 as
   likely-concurrent work narrowing §2.3's exclusivity statement.
4. **Copyedits:** "frontier language model (Claude Haiku)" → accurate tier; Figure 2 caption
   cross-ref (§4.2 → §3.3); remove "decisive" from the p=0.07 result; flag Figure 3's axis
   truncation; replace the CIKM header and the literal "X" page markers.
5. **Engage the spectral account.** §2.1 dismisses BWGNN/GHRN/GAGA on a supervision-ratio argument
   that your own BWGNN run (0.8879 at *this* split) contradicts. Either delete the clause or
   engage the heterophily theory on its merits.

---

## Phase 3 — Analysis from artifacts already on disk

No new training; needs score vectors or logs.

6. **Operating-point metrics** (4/5 seats). precision@{1,5,10}%, recall at FPR ∈ {0.1,1,5}%,
   partial AUC at FPR ≤ 0.05, and F1 at a *validation-tuned* threshold rather than a fixed 0.5 cut.
   Run for R1–R4 on Yelp and for R1/R2 on Amazon — where F1-macro moved **−1.43 pp, 0/20 seeds**.
   This is the threshold-calibration check from earlier: does the drop vanish at a tuned cut?
7. **Repair or retire the dose–response** (flagged by **all five seats**). Δ is currently regressed
   on its own baseline term, which manufactures a negative slope under the null. Re-estimate
   against a baseline from disjoint seeds, or model gain as a fraction of remaining headroom.
   Report slope CI and n; show the fit with and without GraphConsis; disclose that the fit uses
   nine points while the text says eight.
8. **θ diagnostics.** Per-relation distribution of the learned thresholds at convergence: mean,
   spread, across-node SD, saturation fraction. If θ is near-constant within a relation, CAPN is an
   expensive way to learn a global threshold and the +0.58 pp has a duller explanation. Values are
   already in the training logs.
9. **Statistical reporting.** Declare alpha; Holm/BH within each test family; report *t*(df), exact
   p, paired mean difference and 95% CI; split the two significance procedures (paired-t over seeds
   vs. bootstrap over nodes) into labelled columns and stop summing across them.

---

## Phase 4 — New experiments, in priority order

10. **PC-GNN as third host** (D2). ~2–3 days build + ~1 day runs.
11. **Camouflage stratification.** Partition test nodes by a CARE-GNN-style camouflage statistic;
    report per-stratum AUC/AP for R1–R4. Camouflage motivates the entire paper and is measured
    nowhere. If the gain is uniform across strata, the framing must change to match.
12. **Controls** (Devil's Advocate). A generic text baseline (TF-IDF or sentence embeddings) and a
    random/shuffled six-column null through the same gate — otherwise "adding roughly any text
    signal helps" fits the data equally well. Plus a non-learned per-node threshold control, which
    separates granularity from learning.
13. **Heuristic-scored R4.** Table 3 establishes scorer-equivalence for *concatenated* text6, not
    for the gated framework. One re-run makes the headline vendor-independent and removes a
    dependency on a model snapshot that will be deprecated.

---

## Phase 5 — Rewrite (after Phase 1 lands)

14. **Restructure around the wrapper:**
    - §4.4 (wrapper) → **main result**
    - §4.2 (pillar ablation) → mechanism study: *why* the wrapper works
    - §4.3 (8 backbones) → scope boundary: the semantic half travels even where CAPN cannot go
    - New: the comparison table placing R1–R4 beside the strong baselines, with the honest reading
15. **Drop "unification is necessary."** The interaction is **+0.16 pp against R3's own seed-std of
    0.43 pp** — the pillars are additive. Replace with: each pillar contributes, they stack
    additively, attach them to your selection-based GNN. Report the null interaction as a *finding*
    — it tells the 2025 GNN+LLM wave something it assumes without testing.
16. **Scope every claim to its host.** "Every pillar … held-out-confirmed" is true on CARE-GNN, not
    on GraphConsis.
17. **Add a Discussion section** (there is none) plus consolidated limitations, false-positive cost,
    and recourse. Add code/data availability, hyperparameter table with tuning-parity statement,
    and a cost ledger.

---

## Critical path

```
D1 (confirm reframe)
  └─> Phase 1 runs (overnight: 1.1 CPU ∥ 1.2 then 1.3 GPU)
        └─> 1.1 result decides: two hosts, or one host + mandatory PC-GNN
              └─> Phase 5 rewrite
Phase 2 + 3 run in parallel throughout — nothing in them depends on D1.
```

**Realistic timeline:** ~2–3 weeks to a submittable draft, assuming PC-GNN is in scope.
Without PC-GNN, ~1–1.5 weeks to a weaker but honest two-host paper.

**The one thing that would sink it:** if GraphConsis at n=20 stays non-significant *and* PC-GNN is
out of scope, there is no wrapper claim — only a single-host framework paper, which is the paper
the panel just rejected. That is why 1.1 runs first.
