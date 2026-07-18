# RL-pillar significance quest

**Goal (user `/goal`, 2026-07-05):** make the adaptive (RL/CAPN) pillar's
contribution *statistically significant* (positive delta, p<0.05), like the GNN
and LLM pillars. Iterate reward + policy improvements until met, or until an
honest wall is reached.

**Success criterion:** RL delta (CAPN vs standalone, same seeds, paired t-test)
positive with p<0.05, **confirmed on held-out seeds** not used to pick the
config (see Anti-p-hacking below).

## Anti-p-hacking protocol (non-negotiable)

We are deliberately searching config space for significance, which invites
selection bias. Guards:
1. **Exploration** on seeds 72–76 (n=5). Pick the config by *effect size and
   variance*, not by p alone.
2. **Confirmation** of the chosen config on **fresh held-out seeds 100–119**
   (n=20), never touched during exploration. The confirmation p-value is the
   one that counts.
3. **Every** config tried is logged below (the log IS the multiple-comparisons
   record). If k configs were explored, the reader can Bonferroni-correct.
4. Results saved to `results/experiments/rl_quest/<config>_n<k>.json`.

## Platform

`scripts/graphconsis_experiment.py --rl-quest <config> --seeds ...`. GraphConsis
is the host: RL's largest observed effect (+1.20pp) and richest selection of the
CARE-family backbones. RL pillar = CAPN vs standalone, both `use_llm=False`, to
isolate the adaptive pillar. Standalone for seeds 72–76 is reused from
`graphconsis_table.json` (raw32); other seeds computed fresh.

## Baseline (pre-quest, from stage-3 wrapper run)

| | AUC (n=5, seeds 72–76) |
|---|---|
| GraphConsis standalone | 0.7840 ± 0.0357 |
| + CAPN (RL) | 0.7960 ± 0.0207 |
| **RL delta** | **+1.20pp, t=0.53, p=0.63 — n.s.** |

Diagnosis: not a null (mean is +1.20pp, variance already halved), but
underpowered + high-variance. Two fixable causes identified:
- **Reward:** the w2 accuracy term (`model.py:221`, and the GraphConsis runner)
  used *raw* accuracy — majority-dominated at 14.5% positives, ~useless signal.
- **Policy:** vanilla advantage-weighted REINFORCE, aggressive single step
  (policy_lr 3e-3, warmup/ramp 5/5, no entropy anneal) → high variance, and it
  destabilises the LLM-stacked config.

## Config ledger

Each config changes the baseline one lever at a time (definitions in
`RL_QUEST_CONFIGS`, `scripts/graphconsis_experiment.py`).

| Config | Change vs baseline |
|---|---|
| `baseline` | none (raw-acc reward, lr 3e-3, warmup/ramp 5/5) |
| `v1_reward` | reward → balanced accuracy (macro-recall) |
| `v2_stab` | v1 + policy_lr 1e-3, warmup/ramp 10/10, entropy 0.02 annealed |
| `v3_stab_soft` | v1 + policy_lr 5e-4, warmup/ramp 10/10, entropy 0.03 annealed, λ_policy 0.10 |

## FINAL VERDICT (2026-07-15): GOAL MET — clean RL is significant on CARE-GNN

The reviewer-prompted de-confounding of R2 (removing `--llm-priors-file`)
revealed the priors were SUPPRESSING the policy (−0.21pp). Clean CAPN vs R1 on
CARE-GNN: exploration (seeds 72–76) +0.39pp p=0.039; **pre-registered n=20
held-out confirmation (seeds 100–119): +0.579pp, positive on 20/20 seeds,
t=13.77, p<1e-10.** `results/experiments/rl_clean_confirmation_n20.json`.

Reconciliation with the GraphConsis null below: the adaptive effect is ~+0.4–0.6pp
everywhere we measured it; CARE-GNN's seed-std (±0.0016) makes it detectable,
GraphConsis's (±0.04) does not at practical n. Both results are true; the pillar
is small-but-real, and the earlier "honest null on mean AUC" applies to the
high-variance host, not to the pillar per se.

## VERDICT (2026-07-09): honest null on mean AUC (GraphConsis host — superseded in scope by the above)

**v3_stab_soft, n=30 held-out seeds 100–129:** standalone 0.7617±0.0420 →
CAPN 0.7658±0.0312 · **RL delta +0.41pp, t=0.58, p=0.56 — n.s.** (14/30 seeds
positive, delta range [−8.3, +9.8]pp). The +1.39pp seen on exploration seeds
72–76 was favourable-seed noise; the true mean effect on fresh seeds is ~+0.4pp.

Pre-registered rule (p>0.10 → honest null) fires: **the GOAL as stated — a
significant positive mean-AUC contribution — is NOT MET, and I stop chasing it
(continuing to try configs until one crosses p<0.05 would be p-hacking).**

**What CAPN actually does (secondary, partly confounded):** CAPN's own variance
is lower than standalone's (0.031 vs 0.042, ratio 0.55; Pitman-Morgan p=0.079 —
a TREND, not significant). A delta-vs-standalone-level regression is steeply
negative (slope −0.64, p<1e-4: worst-10 seeds +3.75pp, best-10 −1.74pp) BUT this
is substantially inflated by **regression to the mean** (delta = CAPN−standalone
is mechanically anti-correlated with standalone under measurement noise), so it
cannot be claimed cleanly. Honest read: CAPN trades a negligible mean change for
a modest, not-quite-significant variance reduction.

**Why more of the same won't work:** to detect +0.41pp at delta-std 3.84pp needs
n≈340 seeds (absurd); tuning was exhausted (iter1/iter2: reward fix is the only
lever, stabilisation/bounding neutral-to-harmful). Only a *genuinely different*
mechanism (reward realigned to val-AUC; richer/edge action space; PC-GNN host)
could enlarge the true effect — real research, uncertain, days, may still fail.

## Run log

(Appended as runs complete. Format: date · config · n · standalone → CAPN ·
RL delta · p · verdict.)

- 2026-07-05 · quest opened · baseline recap: +1.20pp p=0.63 n.s. · reward+policy
  fixes implemented, exploration launching (v1_reward, v2_stab at n=5).
- Perf: cached the static consistency scores in `ConsisSampler` (they depend only
  on raw features) → 4x faster training, bit-identical values. ~12 min/CAPN seed.
- 2026-07-05 · **v1_reward** (balanced-acc reward) · n=5 (72–76) · standalone
  0.7840±0.0357 → CAPN **0.7983±0.0153** · RL delta **+1.43pp** · t=0.91 **p=0.41**
  · n.s. but improves baseline on all axes (mean ↑, variance ↓26%, p 0.63→0.41).
  Per-seed: policy RESCUES worst seed (74: 0.7365→0.8070, +7pp) but trails on best
  seed (72: 0.8312→0.8069, −2.4pp). **Key insight: significance is blocked by
  delta sign-inconsistency across seeds, not by a weak mean** → stabilisation
  (stop hurting good seeds) + n=20 (average out sign noise) are the right levers.
  `results/experiments/rl_quest/v1_reward_n5.json`.
- 2026-07-05 · **v2_stab** (v1 + lr 1e-3, warmup/ramp 10/10, entropy 0.02 anneal)
  · n=5 · CAPN **0.7930±0.0229** · RL delta **+0.91pp** · p=0.55 · n.s. and
  **WORSE than v1 on every axis**. Verdict: stabilisation is the wrong lever —
  CAPN here is a near-bandit (one action, immediate reward), so conservative
  updates just mean less learning. Dead-end; keep v1's aggressive lr 3e-3 /
  warmup 5. `results/experiments/rl_quest/v2_stab_n5.json`.
- 2026-07-07 · **v3_stab_soft** (v1 + lr 5e-4, warmup/ramp 10/10, entropy 0.03
  anneal, λ 0.10) · n=5 · CAPN **0.7979±0.0127** · RL delta **+1.39pp** · p=0.41 ·
  n.s. but **tied with v1 on effect and LOWEST variance yet** (±0.0127). Revises
  the v2 read: v2 was a bad middle ground, not proof that stabilisation fails —
  going *softer still* recovers v1's effect. Still carries the seed-72 dent
  (−2.6pp). `results/experiments/rl_quest/v3_stab_soft_n5.json`.
- **Decision after iter1:** winners tied = **v1_reward** and **v3_stab_soft**
  (~+1.4pp, p=0.41; v3 tighter). All iter1 configs still dent the best seed
  (72: standalone 0.831 → CAPN ~0.805). Significance math: v3 delta-std ~3.36pp
  → **p≈0.03 at n=30** if the effect holds on fresh seeds. iter2 (eps_scale
  bounding + stronger λ) targets the seed-72 dent to raise both mean and
  consistency before the n=20–30 held-out confirmation on seeds 100–119.

- 2026-07-07 · iter2 launched: v4_bounded (eps_scale 0.15), v5_strong (λ 0.30),
  v6_bounded_strong (both), n=5. Testing whether bounding the action removes the
  good-seed dent.
- 2026-07-07 · **iter2 DEAD-END.** v4_bounded −0.67pp p=0.61 (bounding the action
  turned the delta NEGATIVE — it kills the bad-seed rescues too); v5_strong
  +1.28pp p=0.42 (neutral); v6 == v4 (eps_scale dominates). **Key finding: the
  seed-72 dent is −2.4 to −2.6pp in EVERY config** — it is structural (the policy
  can't improve an already-good backbone solution), not tunable by lr/λ/entropy/
  eps_scale. The "over-filtering" hypothesis was wrong; global action-bounding
  can't tell good seeds from bad. `rl_quest/v{4,5,6}*_n5.json`.
- **Decision:** stop speculative mechanism tuning. Best config = **v3_stab_soft**
  (+1.39pp, tightest variance ±0.0127). Pre-register the honest confirmation:
  **v3_stab_soft at n=30 on held-out seeds 100–129** (never used in exploration).
  n=30 chosen because the n=5 delta-std (~3.37pp) puts p≈0.03 there IF the effect
  holds. Outcome rule: p<0.05 → GOAL MET; 0.05<p<0.10 → iter3 (per-node
  confidence-gated abstention, the principled fix for the seed-72 dent) rather
  than adding seeds; p>0.10 → honest null, effect doesn't survive at scale.
- 2026-07-07 · **PAUSED (user request)** mid-confirmation. Standalone seeds
  100–123 (24/30) were computed and **harvested into
  `results/experiments/rl_quest/_standalone_cache.json`**; rl_quest now reads +
  writes this cache incrementally, so nothing recomputes. CAPN phase had not
  started. **RESUME:** `python scripts/graphconsis_experiment.py --rl-quest
  v3_stab_soft --seeds 100..129 --device cpu` (OMP_NUM_THREADS=6). Standalone
  100–123 load instantly; only 124–129 standalone + all 30 CAPN run (~6.5h).
- 2026-07-08 · **OVERNIGHT-STALL incident.** The resumed run ran normally ~30 min
  (std 124–126) then crawled: ~1 seed in 8 h at ~0.5 core with 15 GB RAM free =
  Windows idle power-throttling (process kept accruing CPU, so not a hang/OOM).
  Cache saved everything (std through 126). **Two robustness fixes:** (1) CAPN
  results now cached per (config,seed) in `_capn_<name>_cache.json`, written as
  each lands → the 6 h CAPN phase is now seed-by-seed resumable too; (2)
  `_prevent_sleep()` (SetThreadExecutionState) keeps Windows from idle-throttling
  during the run (self-reverts on exit). Restarted clean (task b9e1gxgq9): std
  100–126 from cache, 127–129 + 30 CAPN fresh, ~6.5 h. Resume after any future
  kill re-uses both caches → only uncomputed seeds run.
- 2026-07-08 · **PAUSED (user request)**, clean. Standalone 100–127 cached (28/30),
  CAPN 0/30 (phase not yet started). **RESUME:** `OMP_NUM_THREADS=6 python
  scripts/graphconsis_experiment.py --rl-quest v3_stab_soft --seeds 100..129
  --device cpu` → std 100–127 instant, std 128–129 + 30 CAPN fresh (~6.3 h).
