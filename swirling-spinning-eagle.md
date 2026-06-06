# Plan: GNN + RL + LLM Unified Framework for Yelp Fraud Detection

## Context

The thesis requires a framework where **all three components (GNN, RL, LLM) clearly contribute** to fraud detection performance. Previous experiments showed:
- Generic sentence embeddings into the policy state: **0pp improvement** (3 phases tried, all failed)
- Root cause: generic embeddings don't capture fraud-specific signals; the 32 handcrafted features already cover behavioral spam patterns

The key insight: the LLM must analyze **actual review text content** — something the handcrafted features and graph statistics cannot capture. On Amazon, template risk scores (from graph stats) gave +0.05pp. On Yelp, we have 45,954 raw review texts that no existing feature covers.

---

## Architecture

```
Stage 1 (LLM — offline):  Review Text → Claude Haiku → Text Fraud Scores [N, 6]
                           Relation Analysis → LLM Priors (JSON)

Stage 2 (GNN — backbone):  Node Features (32) + Gated Text Scores → CARE-GNN → Embeddings

Stage 3 (RL — adaptive):   Actor-Critic Policy (state includes text scores) → Thresholds
```

LLM feeds into both GNN (via gated feature enrichment) and RL (via state vector). RL controls GNN's neighbor filtering. All three interact.

---

## Phase 1: Text-Based LLM Risk Scoring

**New file:** `llm/generate_text_risk_scores.py`

Use Claude Haiku to analyze each review's TEXT and produce 6 fraud-specific scores:

| Score | What it captures |
|-------|-----------------|
| `generic_language` | How template-like/vague (0=specific, 1=generic) |
| `sentiment_mismatch` | Inconsistency between expressed opinion and overall tone |
| `promotional_tone` | How advertorial the language sounds |
| `copy_paste_signal` | Signs of formulaic/automated text |
| `detail_authenticity` | Authenticity of specific details (inverted: 0=fake, 1=real) |
| `behavioral_anomaly` | Unusual writing patterns (caps, emoji, repetition) |

**Implementation:** Model on existing `llm/generate_risk_scores.py` (lines 264-369):
- Batch 10 reviews per API call
- Checkpoint every 500 nodes
- Template fallback (text length, exclamation count, unique word ratio, etc.)
- Output: `llm_embeddings/yelp/text_risk_scores.pt` shape `[45954, 6]`

**Also create:** `data/llm_priors/yelp_priors.json` — relation-specific priors:
- R-U-R: importance=0.85, camouflage_risk=0.55 (same-user reviews, high label similarity 0.90)
- R-T-R: importance=0.60, camouflage_risk=0.40 (same-product-month, low label similarity 0.05)
- R-S-R: importance=0.70, camouflage_risk=0.45 (same-product-rating, low label similarity 0.05)

**Cost:** ~$10-18 for Claude Haiku on 45,954 reviews.

---

## Phase 2: Actor-Critic RL (Replace REINFORCE)

**New class in `capn.py`:** `ValueNetwork`
```
state [batch, state_dim] → Linear(state_dim, 64) → ReLU → Linear(64, 32) → ReLU → Linear(32, 1)
```

**Changes to `capn.py`:**
- `PolicyNetwork.get_policy_loss(advantage)` — takes advantage instead of raw reward
- `ShapedRewardComputer` — returns raw reward (no internal advantage normalization)

**Changes to `model.py` — `CAPNOneLayerCARE.loss()`:**
```
raw_reward = reward_computer.compute_reward(...)
V = value_network(state).mean()
advantage = raw_reward - V.detach()
policy_loss = policy_network.get_policy_loss(advantage)
critic_loss = (V - raw_reward.detach())²
total = supervised + λ_policy * policy_loss + λ_critic * critic_loss
```

**GNN warmup** in `train.py` training loop:
- Epochs 0 to `gnn_warmup_epochs-1`: train GNN only (policy disabled)
- Epoch `gnn_warmup_epochs` onward: enable policy with linear lambda ramp

**New args:** `--use-actor-critic`, `--critic-lr 1e-3`, `--lambda-critic 0.1`, `--gnn-warmup-epochs 5`

---

## Phase 3: Text Feature Integration into GNN

**New class in `capn.py`:** `TextFeatureGate`
```python
class TextFeatureGate(nn.Module):
    # output = original_features + sigmoid(gate) * project(text_scores)
    # gate initialized at -2.0 (sigmoid ≈ 0.12) — text influence starts small
    def __init__(self, feat_dim, text_dim):
        self.project = nn.Linear(text_dim, feat_dim)
        self.gate = nn.Parameter(torch.zeros(feat_dim) - 2.0)
```

**Why gate instead of concat:** v3 concat failed on Amazon (-0.59pp AUC) because it changed the weight matrix dimensions. The gate keeps the original feature dimension and learns which text dimensions are useful.

**Changes to `train.py`:**
- Load `text_risk_scores.pt` 
- Apply `TextFeatureGate` after feature normalization, before creating embedding layer
- New arg: `--text-enrichment` choices `['none', 'concat', 'gate']`

---

## Phase 4: Text Scores in RL State

**Changes to `capn.py` — `StateConstructor`:**
- Accept `text_risk_scores` tensor `[N, 6]`
- In `compute_state()`, append the 6 text scores directly to state vector (no projection needed — already compact)
- State: `[x_v(32), conf(1), deg(1), dist(1), overlap(1), var(1), text_risk(6)] = 43 dims`

**New arg:** `--text-state-enrichment` (bool)

---

## Ablation Study Design

**Main table (thesis):**

| Row | Config | GNN | RL | LLM | Expected AUC |
|-----|--------|-----|-----|-----|-------------|
| 1 | CARE-GNN | yes | no | no | ~0.766 |
| 2 | CAPN-AC | yes | yes | no | ~0.775 |
| 3 | CARE-GNN + text | yes | no | yes | ~0.772 |
| 4 | Full framework | yes | yes | yes | ~0.780+ |

Each row should beat the one above. The full framework shows all three contributing.

**Component isolation (secondary):**
- REINFORCE vs Actor-Critic
- Warmup=0 vs 5 vs 10 epochs
- Text concat vs gate
- Text in state vs not

---

## Implementation Order

| Step | What | Files | Depends on |
|------|------|-------|-----------|
| 1 | Create text risk score script + template mode | `llm/generate_text_risk_scores.py` | nothing |
| 2 | Run template scores (immediate) | run script | Step 1 |
| 3 | Run Claude Haiku scores (background, ~2hr) | run script | Step 1 |
| 4 | Create Yelp priors JSON | `data/llm_priors/yelp_priors.json` | nothing |
| 5 | Add ValueNetwork + Actor-Critic | `capn.py`, `model.py`, `train.py`, `config.py` | nothing |
| 6 | Test: CAPN Actor-Critic (no LLM) | run training | Step 5 |
| 7 | Add TextFeatureGate + text enrichment | `capn.py`, `train.py` | Step 2 |
| 8 | Test: CARE-GNN + text gate (no RL) | run training | Steps 2, 7 |
| 9 | Add text state enrichment to StateConstructor | `capn.py`, `train.py` | Steps 2, 5 |
| 10 | Test: Full framework (GNN+RL+LLM) | run training | Steps 2-9 |
| 11 | Add ablation experiments | `ablation.py` | Steps 5-9 |
| 12 | Run full ablation (3 seeds) | run ablation | Step 11 |

Steps 1-4 and 5 can run **in parallel**.

---

## Verification

```bash
# Phase 1: Generate text scores
python -m llm.generate_text_risk_scores --data yelp --mode template

# Phase 2: Actor-Critic only
python train.py --data yelp --use-capn --use-actor-critic --gnn-warmup-epochs 5

# Phase 3: Text features only (no RL)
python train.py --data yelp --text-enrichment gate

# Full framework
python train.py --data yelp --use-capn --use-actor-critic --gnn-warmup-epochs 5 \
    --text-enrichment gate --text-state-enrichment \
    --llm-priors-file data/llm_priors/yelp_priors.json

# Ablation
python ablation.py --study thesis --data yelp --num-seeds 3
```

Success: Row 4 (full) > Row 2 (GNN+RL) > Row 1 (GNN only), AND Row 4 > Row 3 (GNN+LLM).

---

## Critical Files

| File | Changes |
|------|---------|
| `llm/generate_text_risk_scores.py` | **NEW** — text-based Claude fraud scoring |
| `data/llm_priors/yelp_priors.json` | **NEW** — relation priors for Yelp |
| `capn.py` | Add `ValueNetwork`, `TextFeatureGate`, modify `StateConstructor`, `PolicyNetwork` |
| `model.py` | Modify `CAPNOneLayerCARE.loss()` for actor-critic advantage + critic loss |
| `train.py` | New CLI args, warmup logic, text score loading, gate integration, critic optimizer |
| `config.py` | New fields for actor-critic, text enrichment, warmup |
| `ablation.py` | Add thesis ablation + component isolation experiments |






● The current architecture (after all our changes in this session):

                           CARE-GNN / CAPN Architecture
                           ============================

  INPUT LAYER
  ├── Node Features [N=45954, 32] (handcrafted: rating deviation, burstiness, sentiment, etc.)
  ├── 3 Adjacency Lists:
  │   ├── R-U-R (same User, 49K edges)
  │   ├── R-T-R (same producT+month, 574K edges)
  │   └── R-S-R (same product+Star rating, 3.4M edges)
  └── Labels: binary (6,677 fraud / 39,277 legit)

  GNN BACKBONE (CARE-GNN)
  ├── Label-Aware Similarity (MLP label predictor → confidence scores)
  ├── Per-Relation Neighbor Filtering:
  │   ├── [CARE-GNN mode]: global threshold per relation (heuristic RL)
  │   └── [CAPN mode]: per-node threshold via PolicyNetwork (Beta distribution)
  │       ├── Hard top-K filtering (default)
  │       └── Soft attention weighting (--soft-attn flag, Phase 2)
  ├── IntraAgg: mean aggregation of filtered neighbors per relation
  ├── InterAgg: weighted sum across relations (threshold as weight)
  └── Output: node embeddings [batch, 64] → classification [batch, 2]

  RL MODULE (CAPN — when --use-capn)
  ├── StateConstructor builds state vector per node:
  │   ├── node features [32]
  │   ├── confidence [1]
  │   ├── degree [1]
  │   ├── mean neighbor distance [1]
  │   ├── Jaccard overlap [1]
  │   ├── feature variance [1]
  │   ├── [optional] LLM embedding projection [64] (--use-llm-state)
  │   └── [optional] enrichment projection [16] (--enrichment-mode)
  │   = state_dim: 37 base, 101 with LLM, 53 with enrichment
  │
  ├── PolicyNetwork (per relation):
  │   ├── Shared: Linear(state_dim→64) → ReLU → Linear(64→32) → ReLU
  │   └── 3 relation heads: Linear(32→2) → softplus → Beta(α,β)
  │   └── Output: per-node threshold ∈ (0,1), log_prob
  │
  ├── ShapedRewardComputer:
  │   ├── R = w1*dist_improvement + w2*acc_improvement - w3*threshold_reg
  │   └── Advantage normalization (running EMA mean/std)
  │
  └── REINFORCE: loss = -advantage * Σ log π(t|s) - λ_H * entropy

  LLM COMPONENTS (current state)
  ├── Sentence embeddings: all-MiniLM-L6-v2 [45954, 384]
  │   ├── llm_semantic_embeddings.pt (from actual review text)
  │   └── Projected via deep_mlp: 384→256→128→64 with LayerNorm
  ├── Template risk scores: [N, 6] (from graph statistics, not text)
  │   └── structural_anomaly, relation_consistency, neighborhood_risk,
  │       feature_anomaly, coordination_signal, isolation_score
  ├── Node statistics: [N, 26] (degrees, percentiles, disagreement, etc.)
  └── LLM priors: only exist for Amazon, NOT yet for Yelp

  LOSS FUNCTION
  ├── L_gnn: CrossEntropy on GNN predictions
  ├── L_label: CrossEntropy on label predictor (weighted by λ₁=2)
  ├── L_policy: REINFORCE with entropy reg (weighted by λ_policy=0.3)
  └── Total = L_gnn + λ₁*L_label + λ_policy*L_policy

  TRAINING
  ├── Dual optimizer: Adam(GNN, lr=0.01) + Adam(Policy, lr=3e-3)
  ├── 31 epochs, batch_size=1024, 1:1 undersampling
  ├── Gradient clipping: GNN=1.0, Policy=5.0
  └── LR scheduler: ReduceLROnPlateau on val AUC

  Current Yelp results with this architecture:

  ┌─────────────────────────────────────────┬─────────┐
  │                 Config                  │ Val AUC │
  ├─────────────────────────────────────────┼─────────┤
  │ CARE-GNN (no RL, no LLM)                │ 0.7705  │
  ├─────────────────────────────────────────┼─────────┤
  │ CAPN (RL only)                          │ 0.7759  │
  ├─────────────────────────────────────────┼─────────┤
  │ CAPN + text embeddings (all variations) │ ~0.773  │
  └─────────────────────────────────────────┴─────────┘

  What's missing (from the plan):
  - No text-based fraud scores from actual review content (LLM analyzes graph stats, not text)
  - No Actor-Critic (still using REINFORCE)
  - No GNN warmup
  - No gated text feature integration into GNN
  - No Yelp-specific LLM priors



