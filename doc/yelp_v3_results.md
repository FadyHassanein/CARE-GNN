# YelpChi backbone-capacity improvement (v3)

**Branch:** `yelp-significant-improvements` · **Commits:** `8f4a32d` (levers), `29aed52` (cache fix)
**Figures:** `figures/v3/` · **Numbers:** `results/experiments/v3_final_summary.json`

## TL;DR

Adding a nonlinear feature encoder, ego/neighbour separation, and a cosine LR
schedule (trained for 80 epochs) lifts YelpChi test AUC to **0.7968 ± 0.0016**
across 3 seeds — **+0.0112 over the documented Row 4 (0.7856)** and **+0.0094
over the real-Haiku baseline (0.7874)**, with all three seeds improving and the
seed variance roughly halved (0.0036 → 0.0016). AP rises from ~0.41 to **0.4287**.

| Config (3 seeds: 72/73/74) | AUC | AP | F1 |
|---|---|---|---|
| Documented Row 4 (template text) | 0.7856 ± 0.0022 | 0.4108 | 0.6099 |
| Baseline (real-Haiku, per-element AC, lp=0.15, 31ep) | 0.7874 ± 0.0036 | 0.4086 | 0.6221 |
| **v3 (+ encoder + ego-sep + cosine, 80ep)** | **0.7968 ± 0.0016** | **0.4287** | **0.6358** |

Per-seed v3 AUC: 72 → 0.7991, 73 → 0.7959, 74 → 0.7954 (3/3 above baseline).

## Why: the backbone was underfitting

A linear/nonlinear probe on the **frozen** YelpChi split (same two-stage
stratified `train_test_split`, `random_state=2`, 25/15/60 — `scripts/probe_yelp_headroom.py`)
showed that classifiers on the *same* node features the GNN already consumes
beat the single-layer CARE-GNN (~0.787 AUC) by a wide margin:

| Model on `raw32 + text6` | Test AUC |
|---|---|
| GNN + CAPN + LLM (pipeline) | ~0.787 |
| Logistic regression | 0.792 |
| MLP (64,64) | 0.829 |
| Gradient boosting | 0.873 |

The original backbone applies **one** linear transform + ReLU to frozen input
features, so it cannot represent the nonlinear feature interactions a plain MLP
or GBM exploits. The headroom was in model capacity, not in the RL or LLM
pillars. (The probe also **falsified** routing the structural `llm_risk_scores`
through the gate: structural-6 *hurts* on Yelp, −0.041 vs raw, so it was rejected.)

## What: three composable, flag-gated levers (defaults off → baselines unchanged)

1. **`--feat-encoder`** — a learnable nonlinear MLP (`FeatureEncoder`, 32→64→64,
   ReLU, dropout 0.5) replaces the single linear transform in `InterAgg`, giving
   the backbone genuine nonlinear capacity.
2. **`--ego-separation`** — concat-project `[self, neighbours]` instead of
   summing them, so the dominant self-signal is not diluted by camouflaged
   (heterophilous) neighbours. On YelpChi, fraud reviewers hide among legitimate
   neighbours, so summing neighbour features *into* the ego representation is the
   wrong inductive bias.
3. **`--cosine-lr`** — linear warmup (8ep) + cosine annealing to 0.05·lr, stepped
   every epoch on both optimisers. The original `ReduceLROnPlateau` never fired
   (LR was pinned at 0.01), so the higher-capacity model overfit a noisy plateau.

## Attribution (seed 72)

| Step | AUC | reading |
|---|---|---|
| Baseline (31ep) | 0.7864 | — |
| + encoder only (31ep) | 0.7743 | **capacity alone HURTS** — neighbour-sum dilutes the self-signal, and the flat LR overfits |
| + ego-sep + cosine (31ep) | 0.7819 | separation + annealing recover it, but the bigger model is still undertrained |
| + 80-epoch budget (**v3**) | 0.7991 | budget lets the higher-capacity model converge → **+1.27pp over baseline** |

The components are interdependent: the encoder is only beneficial *with*
ego-separation (to protect the self-signal) and cosine + budget (to converge
without overfitting). The full one-lever-out table at 80 epochs was blocked by a
hard RAM limit on the training machine (15.7 GB; a Yelp CAPN run is near the
ceiling) and is left as future work.

## Honest framing & limitations

- The improvement lives in the **GNN backbone** (capacity + heterophily-aware
  aggregation + training budget). CAPN (actor-critic, per-element advantage) and
  the real-Haiku LLM text gate are retained on top; this is a stronger, more
  defensible contribution than RL-hyperparameter tuning.
- A gap to the tabular ceiling remains (GBM 0.873). Two attempts to close it —
  a wider/deeper encoder and milder (1:3) undersampling — failed structurally
  (training divergence; a CAPN memory bug, since fixed), suggesting ~0.80 is a
  robust ceiling for this GNN+CAPN framework without deeper redesign.
- All numbers are test AUC/AP/F1 at the best-validation checkpoint, 3 seeds
  (72/73/74), reported as mean ± population std.

## Reproduce

```bash
python scripts/run_yelp_experiment.py --tag v3 -- \
  --use-capn --use-actor-critic --gnn-warmup-epochs 5 --lambda-policy-ramp-epochs 5 \
  --llm-priors-file data/llm_priors/yelp_priors.json \
  --text-enrichment gate --text-state-enrichment --lambda-policy 0.15 \
  --feat-encoder --ego-separation --cosine-lr \
  --num-epochs 80 --lr-warmup-epochs 8 --test-epochs 5 --patience 12
```
