# Framework improvement plan — making all three pillars earn their place

**Objective.** The paper's main contribution is a unified three-pillar framework
(GNN + LLM + RL) for camouflage-resistant fraud detection. For that contribution
to survive review, each pillar needs at least one setting where it demonstrably
and significantly earns its place. Today, only the structural pillar and the
*feature content* of the semantic pillar clear that bar. This plan lays out the
concrete, mechanism-level work to change that — plus the comparison tables the
paper draft still needs.

**Discipline (non-negotiable).** Every direction below has a mechanism
hypothesis, a decisive pre-registered test, and an abandon criterion. Explore on
seeds 72–76; confirm anything that will be claimed significant on held-out seeds
(100+) at n≥20. All configs tried get logged (multiple-comparisons record). We
do not tune until something crosses p<0.05 and report only the winner — that
cost us once already (RL quest: n=5 +1.39pp → n=30 +0.41pp, p=0.56).

---

## 0. Current evidence baseline (do not re-litigate; reproduce before extending)

| Claim | Status | Evidence |
|---|---|---|
| Full framework beats CARE-GNN base | **PROVEN** +1.90pp p=0.002 (std), +2.94pp p=0.0019 (v3 backbone) | `yelp_corrected_n5_summary.json`, `yelp_v3_n5_summary.json` |
| text6 features improve 6/8 backbones | **PROVEN** (LR +2.29 … XGB-Graph +0.70, all at frozen 25/15/60) | `tabular_baselines.json`, `xgb_graph_table.json`, `bwgnn_table.json` |
| text6 stabilises fragile backbones | replicated 2× (v3 seed-collapse cured; GraphConsis ±0.036→±0.010) | `yelp_v3_n5_summary.json`, `graphconsis_table.json` |
| LLM (Haiku) beats heuristic template scorer | **NULL on all 4 GNN backbones** (CARE +0.15 p=0.31; BWGNN-Het +0.14; BWGNN-Homo +0.70; GraphConsis p=0.70); sig only on LR/GBM | `doc/llm_vs_heuristic_headtohead.md` |
| RL improves mean AUC | **NULL at n=30 held-out** (+0.41pp p=0.56, GraphConsis best-host) | `doc/rl_significance_quest.md`, `rl_quest/v3_stab_soft_n30.json` |
| RL reduces variance | trend, not proven (Pitman–Morgan p=0.079; worst-10 seeds +3.75pp, RTM-confounded) | same |
| R2 "RL-alone" free of LLM priors | **DONE — clean RL +0.39pp p=0.039 SIG at n=5** (priors were hurting, −0.21pp); coherent w/ GraphConsis null (same ~+0.4pp effect, CARE's ±0.002 noise makes it detectable). **n=20 held-out confirmation running** (seeds 100–119) — claim nothing until it lands | `r2_no_priors.json`; confirm → `r{1,2_no_priors}_holdout20.json` |
| Amazon: text pathway | structurally absent (no review text); structural scores ≈ 0 / negative | `doc/gnn_rl_llm_ablation_results.md` |

Protocol constants: YelpChi frozen 25/15/60 split (`random_state=2`), val-AUC
checkpoint, seeds 72–76 explore / 100+ confirm, sample std, paired t-tests.
Hardware: 6 GB GPU (strictly serial), 32 GB RAM, CPU runs thread-capped
(`OMP_NUM_THREADS=6`), `_prevent_sleep()` active for long runs, per-seed caches
under `results/experiments/rl_quest/` pattern.

---

## Workstream G — structural pillar (GNN): raise the framework's ceiling

The backbone caps what the whole framework can show. CARE-GNN at 0.78 makes the
framework look weak next to in-repo baselines (BWGNN-Het 0.894, XGB-Graph 0.918).

### G1. BWGNN as the structural pillar (highest value / moderate effort)
- **Mechanism:** the framework is pillar-modular; nothing in the LLM pillar needs
  CARE-GNN. `scripts/bwgnn_experiment.py` already implements BWGNN + gate.
- **Do:** define "framework-on-BWGNN" = BWGNN-Hetero + gated text6 (+ RL per
  workstream R2 below when ready). Report the 4-row ablation on this backbone:
  BWGNN / +RL / +text6-gate / full. 5 seeds explore, 20 confirm for the headline.
- **Commands:** extend `bwgnn_experiment.py` with a `--framework-table` mode
  reusing `GatedBWGNN`; RL row lands after R2.
- **Success:** framework headline moves 0.78 → ~0.89 with the same honest
  attribution. **Abandon:** never — this is table-stakes modularity evidence;
  even a partial table (no RL row) is publishable.
- Est: 1 day + ~4 h compute.

### G2. Rigor: n=30 for the headline semantic delta (reviewer point 4)
- **Do:** R3-vs-R1 (CARE) and BWGNN±text6 on held-out seeds 100–129. Scores are
  cached → cheap. CARE rows ~10 min/seed GPU; BWGNN ~3 min/seed.
- **Success criterion:** semantic delta holds at n=30 (expected: yes, effect is
  3–8× the RL effect). This makes the paper's strongest number bulletproof.
- Est: overnight GPU queue, zero new code.

---

## Workstream L — semantic pillar (LLM): make the LLM itself necessary

Head-to-head verdict: the six *features* work, but a free heuristic matches
Haiku on every GNN backbone. Two honest ways forward — make the LLM's marginal
value real, or reframe. These give the LLM its fair shot first.

### L1. Text embeddings instead of 6 quantised scores (highest probability)
- **Mechanism:** Haiku's output is 6 numbers, 99.8% quantised to 0.05 — a
  ~7-level ordinal per dimension. Massive information bottleneck. A text
  *embedding* (384–1024 dims) carries signal no 6-number heuristic can encode.
  If the LLM pathway wins anywhere, it's here.
- **Do:** (a) sentence-transformer embeddings of review text (local, free) →
  PCA/projector to 8–32 dims → gate injection (dims >32 need `llm/projector.py`
  MLP mode); (b) optional stronger variant: Haiku/Claude embeddings or
  score-with-rationale → embed the rationale. Compare four arms on 2 backbones
  (CARE R3-style + BWGNN-Het): raw / +template6 / +haiku6 / +emb.
- **Success (pre-registered):** emb beats template6 with p<0.05 on ≥2 GNN
  backbones at n=5, confirmed at n=20 on the better backbone. Then the semantic
  pillar's claim upgrades to "learned text representations beat engineered
  features," and the LLM/encoder is genuinely necessary.
- **Abandon:** if emb ≤ template on both, the reframe stands: engineered
  features, LLM optional. Report either way (it is a finding either way).
- Est: 1–2 days (embedding generation is the long pole; cache like text6).

### L2. LLM-only signals (judgments a heuristic cannot compute)
- **Mechanism:** the current 6 scores have deterministic definitions, so a
  template can match them *by construction*. Prompt instead for judgments with
  no closed-form: cross-review consistency for the same product, plausibility of
  detail given the product category, persona coherence across a user's reviews.
- **Do:** 3–4 new scores, one Haiku pass over cached review groups (cost ≈ one
  text6 generation run), pin artifacts with sha256 sidecars as before, then the
  same ±ablation as text6.
- **Success:** new scores add ≥+0.5pp over raw+template6 on ≥2 backbones,
  p<0.05. **Abandon:** if not, don't stack prompts — one iteration only.
- Est: 1 day + API cost. Do after L1 (L1's verdict shapes the prompt design).

### L3. LLM in the policy state, isolated (reviewer point 3a)
- **Do:** R4 with and without `z_v`/text6 in the CAPN state (flag exists:
  `--text-state-enrichment`), 5 seeds. Currently the novel routing is bundled
  and untested. This is a cheap must-run regardless of outcome.
- Est: ~4 h GPU, zero new code.

---

## Workstream R — adaptive pillar (RL): a mechanism change, not more tuning

Tuning is exhausted (7 configs logged in `doc/rl_significance_quest.md`). The
n=30 null says the current action (per-node filtering threshold over an
already-good similarity ranking) has ~+0.4pp of true headroom. The pillar
becomes important only if the *action space* or the *objective* changes.

### R1. Reward realignment: optimise the actual metric (do first — 1 day)
- **Mechanism:** current reward = Δ mean neighbour distance + balanced-acc EMA —
  a proxy chain. Reward the policy with Δ validation-AUC (or batch AP) computed
  on a held-out mini-fold each epoch, so the policy optimises what we report.
- **Do:** add `'reward': 'val_auc'` arm to `RL_QUEST_CONFIGS` in
  `graphconsis_experiment.py` (infrastructure exists; ~30 lines: score a fixed
  1k-node val subsample per epoch, reward = clipped ΔAUC vs EMA).
- **Gate:** n=5 explore; proceed to n=20 confirm only if delta ≥ +1.5pp AND
  per-seed positive ≥ 4/5. **Abandon:** otherwise (logged).

### R2. Edge-level action space on BWGNN (the real bet — 3–5 days)
- **Mechanism:** GHRN showed pruning heterophilic edges lifts BWGNN ~+0.2–1pp
  with a *global* prune ratio. Make CAPN output a **per-node prune threshold
  over incident-edge heterophily scores** (post-hoc label-disagreement proxy
  from the label predictor, as GHRN's `E = L·Ŷ·Ŷᵀ·Lᵀ`). Richer, genuinely
  useful decisions on a strong backbone — and it merges R with G1 so the RL row
  exists on the competitive backbone.
- **Do:** implement in `bwgnn_experiment.py`: precompute per-edge scores each
  epoch-k, policy = per-node Beta over prune fraction (reuse capn.py machinery
  as GraphConsis integration did), reward from R1's val-AUC design.
- **Gate:** BWGNN+policy-prune vs BWGNN and vs BWGNN+global-prune-sweep (the
  GHRN-style fixed-ratio control — the policy must beat the *tuned global*
  ratio, not just the backbone). n=5 explore → n=20 confirm.
- **Abandon:** if the policy can't beat a tuned global ratio, the honest
  conclusion is "per-node adaptivity adds nothing over one global knob here."

### R3. Variance/robustness as the RL pillar's formal contribution (cheap, parallel)
- **Mechanism:** the n=30 data already shows CAPN compresses seed variance
  (0.042→0.031, Pitman–Morgan p=0.079) and rescues worst seeds (+3.75pp on the
  bottom 10). That is a *deployment-relevant* property — reframe the pillar's
  role: the policy is a robustness device, not an accuracy device.
- **Do:** (a) n=50 (add seeds 130–149; caches make this ~7 h CPU) to power the
  Pitman–Morgan test properly; (b) report worst-decile AUC as a headline metric
  alongside mean.
- **Success:** variance reduction p<0.05 at n=50 → the RL pillar has a proven,
  honest, *different* contribution: "the adaptive pillar does not raise mean
  accuracy but significantly reduces run-to-run risk." That IS an important
  pillar — arguably more useful in production than +0.4pp.
- **Abandon:** if p>0.10 at n=50, drop the robustness claim too.

### R4. PC-GNN host (only if R1–R3 all fail)
- Imbalance-aware Pick-Choose selection = richest untested action space. Port
  per `doc` backbone-integration pattern (validate at 40/20/40 vs published
  0.6983±0.03 / our-protocol band first). Est: 2–3 days. Last resort because
  reimplementation cost is high and the GraphConsis lesson says host richness
  alone didn't rescue the mean effect.

**Priority order:** R1 → R3 (parallel) → R2 → R4. If R2 succeeds the pillar has
an accuracy story on the competitive backbone; if only R3 succeeds it has a
robustness story; if all fail, the paper keeps the rigorous null (already
written) — which is honest but weaker than the user's goal, so R1–R3 get real
effort before concluding.

---

## Comparisons to add to the paper draft (with status)

1. **Haiku-vs-template head-to-head table** (7 backbones) — DONE, must go in
   §semantic regardless of framing. `doc/llm_vs_heuristic_headtohead.md`.
2. **Amazon results + honest discussion** (reviewer pt 2) — reuse A-series
   ablations; frame: no review text → text pathway structurally absent;
   structural scores ≈0 on strong features → reinforces dose–response. TODO: one
   table + 1 paragraph.
3. **De-confounded R2** (no LLM priors) — IN FLIGHT (`r2_no_priors`). Report
   both R2 variants; state priors explicitly per row.
4. **n=30 semantic deltas** (G2) — TODO, cheap. Fixes rigor asymmetry.
5. **z_v-in-state isolation** (L3) — TODO, tests the novel routing claim.
6. **Framework-on-BWGNN 4-row table** (G1) — TODO. The modularity proof and the
   competitive headline (~0.89).
7. **Cost/efficiency comparison** — template $0 vs Haiku API cost vs
   sentence-embedding local; training time/params per pillar. One small table;
   strengthens the practitioner story either way L1 goes.
8. **Global-prune-ratio control for RL** (R2's control arm) — doubles as a
   GHRN-style baseline row reviewers will expect.
9. **Robustness metrics** (R3): worst-decile AUC + seed-std per config, n=50.
10. **Published-numbers context table** (BWGNN 0.905, GHRN 0.907, GAGA 0.944 at
    40%; XGB-Graph 0.96–0.97 at 70%) clearly marked *different protocol, not
    comparable* — pre-empts the "you're below SOTA" reflex with transparency.

## Execution order (dependency-aware)

```
Week 1: R1 (reward) ─┬─ R3a (n=50 seeds, CPU, background)
                     ├─ G2 (n=30 semantic, GPU nights)
                     ├─ L3 + item 3 report (cheap GPU runs)
                     └─ items 2, 7, 10 (paper-only, no compute)
Week 2: L1 (embeddings) ── R2 (edge-action BWGNN; merges into G1)
Week 3: G1 framework-on-BWGNN full table (uses R2 if it worked, else 3-row)
        → confirm runs (n=20+) for every claim that will say "significant"
        → paper rewrite w/ final framing decision
```

Falsifiable end-state: by end of week 3 each pillar has either (a) a significant,
held-out-confirmed contribution, or (b) a documented, pre-registered null. The
paper claims exactly what survives. Both outcomes are publishable; only (a) for
all three pillars fulfils the "three equal pillars" goal, and R2/L1 are the bets
that make it possible.
