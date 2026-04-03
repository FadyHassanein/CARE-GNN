# CARE-GNN / CAPN Ablation Study Guide

This document explains every ablation study in the framework: what it tests, why it matters, what configurations are compared, and how to interpret the results.

---

## Table of Contents

1. [Overview](#overview)
2. [Datasets](#datasets)
3. [Metrics](#metrics)
4. [Baseline Studies](#baseline-studies)
   - [inter — Inter-Aggregator Comparison](#inter--inter-aggregator-comparison)
   - [loss — Loss Function Comparison](#loss--loss-function-comparison)
   - [layers — GNN Depth Comparison](#layers--gnn-depth-comparison)
   - [baseline — CARE-GNN vs GraphSAGE](#baseline--care-gnn-vs-graphsage)
5. [CAPN Studies](#capn-studies)
   - [capn — Main CAPN vs Original RL](#capn--main-capn-vs-original-rl)
   - [capn_label — Label Predictor Ablation](#capn_label--label-predictor-ablation)
   - [capn_reward — Reward Shaping Ablation](#capn_reward--reward-shaping-ablation)
   - [capn_llm — LLM Prior Initialization](#capn_llm--llm-prior-initialization)
   - [capn_lambda — Policy Loss Weight Sensitivity](#capn_lambda--policy-loss-weight-sensitivity)
   - [capn_llm_state — LLM Semantic State Enrichment](#capn_llm_state--llm-semantic-state-enrichment)
6. [Running the Studies](#running-the-studies)
7. [Results Format](#results-format)
8. [Generating Figures](#generating-figures)

---

## Overview

The ablation framework (`ablation.py`) systematically evaluates different components and configurations of CARE-GNN and its CAPN extension. Each study isolates a single design decision by holding all other hyperparameters constant and varying only the target variable.

**Architecture summary:**

```
CARE-GNN (base)
  |-- Inter-relation aggregation (GNN / Att / Weight / Mean)
  |-- Intra-relation aggregation (neighbor sampling + averaging)
  |-- Heuristic RL module (Bernoulli bandit, per-relation thresholds)
  |-- Linear label classifier
  |-- L1 neighbor distance

CAPN (extension)
  |-- Learned PolicyNetwork (Beta distribution thresholds, per-node adaptive)
  |-- EnhancedLabelPredictor (2-layer MLP)
  |-- MultiViewDistance (L1 label + structural Jaccard)
  |-- ShapedRewardComputer (distance + accuracy + regularization)
  |-- Optional: LLM prior initialization
  |-- Optional: LLM semantic state enrichment
```

---

## Datasets

| Property | Yelp (YelpChi) | Amazon |
|----------|---------------|--------|
| Domain | Restaurant review fraud | Marketplace review fraud |
| Nodes | 45,954 users | 11,944 users |
| Features | 32-dim | 25-dim |
| Fraud rate | ~14.5% | ~9.5% |
| Relations | RUR, RTR, RSR | UPU, USU, UVU |
| Labeled nodes | All | 3,305+ (0-3,304 unlabeled) |
| Batch size | 1,024 (default) | 256 (recommended) |

**Yelp relations:**
- **RUR** (Reviews-Under-same-Restaurant): Users who reviewed the same restaurants
- **RTR** (Rating-Time-Rating): Users with same star level under the same timestamp
- **RSR** (Rating-Star-Rating): Users who gave same star rating to the same restaurant

**Amazon relations:**
- **UPU** (User-Product-User): Users who reviewed the same products
- **USU** (User-Seller-User): Users who gave the same rating to the same seller within one week
- **UVU** (User-View-User): Users whose review texts have high TF-IDF similarity

**Train/val/test split:** Stratified split with `random_state=2`. Default: 25% train, 15% val, 60% test. Amazon skips unlabeled nodes (indices 0-3,304).

---

## Metrics

All studies track these metrics for both the GNN module and the label classifier:

| Metric | Key | Description |
|--------|-----|-------------|
| AUC-ROC | `gnn_auc` | Area under the ROC curve. Primary ranking metric. |
| Average Precision | `gnn_ap` | Area under the precision-recall curve. Important for imbalanced data. |
| F1 (macro) | `gnn_f1` | Harmonic mean of precision and recall, macro-averaged over both classes. |
| Accuracy | `gnn_accuracy` | Fraction of correctly classified nodes. |
| Recall (macro) | `gnn_recall` | Macro-averaged recall (sensitivity to both fraud and benign). |
| Precision (macro) | `gnn_precision` | Macro-averaged precision. |

**Best model selection:** The epoch with the highest `gnn_auc` is selected as the best result for each experiment.

---

## Baseline Studies

These studies evaluate the original CARE-GNN architecture components without the CAPN extension.

### `inter` -- Inter-Aggregator Comparison

**Research question:** Which method for combining information across relations works best?

**Background:** CARE-GNN aggregates neighbor features within each relation (intra-aggregation), then combines the three relation-level embeddings into a single node embedding (inter-aggregation). The inter-aggregation strategy directly affects how the model weighs different types of social connections.

**Experiments:**

| Experiment | Config | Description |
|------------|--------|-------------|
| CARE-GNN (Threshold) | `inter='GNN'` | Uses the RL-learned thresholds as inter-relation weights. Each relation's contribution is proportional to its filtering threshold. |
| CARE-Weight | `inter='Weight'` | Learnable softmax weights per relation. The model learns a scalar importance for each relation via backpropagation. |
| CARE-Mean | `inter='Mean'` | Simple average across relations. Treats all relations equally. |
| CARE-Att | `inter='Att'` | Attention mechanism over relation embeddings. Each node attends to its relation-level embeddings to compute dynamic weights. |

**What to look for:**
- GNN (Threshold) should perform well because it ties aggregation weights to the RL-learned filtering quality.
- Att may perform comparably by learning per-node dynamic weights.
- Mean serves as the baseline — any method that underperforms Mean is likely overfitting the inter-relation weights.

---

### `loss` -- Loss Function Comparison

**Research question:** Which loss function handles the class imbalance in fraud detection best?

**Background:** Both Yelp (~14.5% fraud) and Amazon (~9.5% fraud) are heavily imbalanced. The loss function determines how the model treats misclassifications of the minority (fraud) class.

**Experiments:**

| Experiment | Config | Description |
|------------|--------|-------------|
| CrossEntropy | `loss='ce'` | Standard cross-entropy. When under-sampling is active, classes are already balanced, so no additional weighting is applied. When under-sampling is off, class weights are computed as `total / (2 * class_count)`. |
| FocalLoss | `loss='focal'` | Focal loss with gamma=2.0. Down-weights easy examples, focuses on hard misclassifications. Particularly useful when the model quickly learns to classify benign users but struggles with fraud. |
| WeightedCE | `loss='weighted_ce'` | Cross-entropy with explicit class weights. Always applies weighting regardless of under-sampling. |

**Important note:** With `under_sample=1` (default), the training batches are already balanced 50/50. Adding class weights on top creates a double-correction that can cause the model to over-predict fraud. The `ce` loss correctly detects this and skips weighting when under-sampling is active.

---

### `layers` -- GNN Depth Comparison

**Research question:** Does stacking multiple CARE-GNN layers improve performance?

**Background:** Each CARE-GNN layer aggregates information from 1-hop neighbors. Stacking L layers captures L-hop neighborhood information. However, deeper GNNs risk over-smoothing (all node embeddings converge to similar values).

**Experiments:**

| Experiment | Config | Description |
|------------|--------|-------------|
| 1-Layer CARE | `model='CARE'` | Standard single-layer. Captures 1-hop neighborhood. |
| 2-Layer CARE | `model='MULTI_CARE', num_layers=2` | Two stacked InterAgg layers with residual connections. Captures 2-hop patterns. |
| 3-Layer CARE | `model='MULTI_CARE', num_layers=3` | Three layers. Risk of over-smoothing increases. |

**What to look for:**
- Fraud detection often benefits from 2-hop information (friend-of-friend patterns).
- 3 layers may degrade due to over-smoothing, especially on small datasets like Amazon.

---

### `baseline` -- CARE-GNN vs GraphSAGE

**Research question:** How much does CARE-GNN's relation-aware filtering improve over a standard GNN baseline?

**Experiments:**

| Experiment | Config | Description |
|------------|--------|-------------|
| GraphSAGE | `model='SAGE'` | Standard GraphSAGE with mean aggregation. Uses the homogeneous graph (union of all relations). No relation-aware filtering or RL. |
| CARE-GNN | `model='CARE', inter='GNN'` | Full CARE-GNN with RL-based threshold filtering and relation-aware aggregation. |

**What to look for:**
- The gap between GraphSAGE and CARE-GNN shows the value of relation-aware neighbor filtering.
- GraphSAGE treats all edges equally, so it cannot suppress camouflaged connections.

---

## CAPN Studies

These studies evaluate the CAPN (Camouflage-Aware Policy Network) extension, which replaces CARE-GNN's heuristic RL with a learned policy network.

### `capn` -- Main CAPN vs Original RL

**Research question:** Does replacing the heuristic Bernoulli bandit RL with a learned policy network improve fraud detection?

**This is the central experiment of the thesis.**

**Experiments:**

| Experiment | Config | Description |
|------------|--------|-------------|
| CARE-GNN (original RL) | `use_capn=False` | Heuristic RL: one threshold per relation, updated via bandit rewards based on average neighbor distance of positive nodes. Thresholds are scalars shared across all nodes. |
| CAPN (policy network) | `use_capn=True` | Learned policy: PolicyNetwork maps per-node state vectors to Beta distribution parameters. Thresholds are sampled per-node, per-relation, trained via REINFORCE. |

**Key differences between the two approaches:**

| Aspect | Original RL | CAPN |
|--------|------------|------|
| Threshold granularity | 1 per relation (3 total) | 1 per node per relation (N x 3) |
| Update mechanism | Bandit reward (+/- step_size) | REINFORCE policy gradient |
| State representation | None (stateless) | Node features + confidence + structural metrics |
| Reward signal | Binary (-1, 0, +1) from avg distance | Shaped (distance improvement + accuracy + regularization) |
| Label predictor | Linear (1 layer) | MLP (2 layers) |
| Distance metric | L1 label distance | MultiView (L1 + structural Jaccard) |

**What to look for:**
- CAPN should show higher AUC/AP if per-node adaptive thresholds help detect camouflaged fraudsters.
- The label predictor upgrade (MLP vs linear) also contributes. Study `capn_label` isolates this effect.

---

### `capn_label` -- Label Predictor Ablation

**Research question:** How much of CAPN's improvement comes from the better label predictor vs the policy network?

**Background:** CAPN makes two changes simultaneously: (1) replaces the linear label classifier with a 2-layer MLP, and (2) adds the policy network. This study disentangles their contributions.

**Experiments:**

| Experiment | Config | Description |
|------------|--------|-------------|
| Linear predictor (CARE-GNN) | `use_capn=False, use_mlp_label=False` | Original CARE-GNN with `nn.Linear(feat_dim, 2)` label classifier. Baseline. |
| MLP predictor (no policy) | `use_capn=False, use_mlp_label=True` | Same CARE-GNN RL, but with `EnhancedLabelPredictor` (2-layer MLP with dropout). Isolates the MLP effect. |
| MLP predictor + policy (CAPN) | `use_capn=True` | Full CAPN with MLP predictor and policy network. |

**What to look for:**
- If "MLP (no policy)" matches CAPN performance, the gain is entirely from the better classifier.
- If CAPN outperforms "MLP (no policy)", the policy network provides genuine additional value.
- The gap between "Linear" and "MLP (no policy)" shows the MLP's standalone contribution.

---

### `capn_reward` -- Reward Shaping Ablation

**Research question:** Does the shaped reward improve over binary reward? Which component matters most?

**Background:** The original RL uses binary rewards: +1 if average neighbor distance increases (good filtering), -1 if it decreases, 0 otherwise. CAPN uses a shaped reward with three components:

```
R = w1 * clamp(delta_dist / (prev_dist + eps), -1, 1)   -- distance improvement
  + w2 * (batch_acc - ema_baseline_acc)                   -- accuracy signal
  - w3 * mean(|threshold - 0.5|)                          -- regularization
```

**Experiments:**

| Experiment | Config | Description |
|------------|--------|-------------|
| Binary reward (original) | `use_capn=False` | Original RL with -1/0/+1 rewards. |
| Shaped reward (CAPN) | `use_capn=True` | Full shaped reward (w1=0.5, w2=0.3, w3=0.2). |
| Shaped (dist only) | `use_capn=True, reward_w1=1.0, w2=0.0, w3=0.0` | Only the distance improvement component. Tests if filtering quality alone is sufficient. |
| Shaped (acc only) | `use_capn=True, reward_w1=0.0, w2=1.0, w3=0.0` | Only the accuracy signal. Tests if downstream task performance is a better reward. |

**Reward components explained:**

1. **Distance improvement (w1):** Measures how much better the current epoch's average neighbor distance is compared to the previous epoch. Positive = neighbors are more distinguishable from the center node (better filtering). Uses relative improvement to normalize across different distance scales.

2. **Accuracy signal (w2):** Compares current batch accuracy against an exponential moving average baseline (decay=0.95). Positive = the model is classifying better than its recent average. Ties the reward to the actual fraud detection task.

3. **Regularization (w3):** Penalizes thresholds that deviate far from 0.5. Prevents the policy from learning extreme thresholds (0.0 = keep no neighbors, 1.0 = keep all) that would collapse the graph structure.

**What to look for:**
- "Shaped reward" should outperform "Binary reward" if the richer signal helps policy learning.
- "dist only" shows whether filtering quality is the primary driver.
- "acc only" shows whether the task-level signal alone is sufficient.
- If "dist only" and "acc only" both underperform the combined reward, the components are complementary.

---

### `capn_llm` -- LLM Prior Initialization

**Research question:** Does initializing CAPN with LLM-derived prior knowledge improve performance?

**Background:** An LLM (Claude) was prompted to analyze each dataset's relation types and assign:
- **Importance scores** per relation: How informative each relation type is for detecting fraud. Used to initialize the gamma mixing parameter in MultiViewDistance.
- **Camouflage risk** per relation: How easy it is for fraudsters to disguise their connections in each relation. Used as initial biases in the PolicyNetwork's per-relation heads.

These priors are stored in `data/llm_priors/{dataset}_priors.json`.

**Experiments:**

| Experiment | Config | Description |
|------------|--------|-------------|
| CAPN (no priors) | `use_capn=True, llm_priors_file=''` | Standard CAPN with default initialization (gamma=0.7, no relation biases). |
| CAPN (LLM priors) | `use_capn=True, llm_priors_file='data/llm_priors/{dataset}_priors.json'` | CAPN with LLM-derived gamma and relation bias initialization. |

**Prior format example:**
```json
{
  "relations": [
    {"name": "R-U-R", "importance": 0.8, "camouflage_risk": 0.6},
    {"name": "R-T-R", "importance": 0.5, "camouflage_risk": 0.4},
    {"name": "R-S-R", "importance": 0.7, "camouflage_risk": 0.5}
  ]
}
```

**What to look for:**
- If LLM priors help, it suggests the LLM correctly identified which relations are more vulnerable to camouflage.
- If no difference, the policy network learns the correct relation priorities on its own.
- LLM priors may help more on smaller datasets (Amazon) where the model has less data to learn from.

---

### `capn_lambda` -- Policy Loss Weight Sensitivity

**Research question:** How sensitive is CAPN to the balance between GNN loss and policy gradient loss?

**Background:** The total CAPN loss is:

```
L_total = (L_gnn + lambda_1 * L_label) + lambda_policy * L_policy
```

If `lambda_policy` is too small, the policy receives weak gradients and barely learns. If too large, the policy gradient dominates and destabilizes GNN training.

**Experiments:**

| Experiment | Config | Description |
|------------|--------|-------------|
| lambda=0.01 | `lambda_policy=0.01` | Very weak policy signal. Policy barely influences training. |
| lambda=0.05 | `lambda_policy=0.05` | Moderate-low. Policy learns slowly but stably. |
| lambda=0.1 | `lambda_policy=0.1` | Default value. Balanced contribution. |
| lambda=0.5 | `lambda_policy=0.5` | Strong policy signal. May destabilize GNN training. |

**What to look for:**
- A robust system should perform similarly across a range of lambda values (0.01-0.1).
- Sharp performance drops at either extreme indicate sensitivity.
- The optimal lambda may differ between Yelp (larger, more data) and Amazon (smaller, less data).

---

### `capn_llm_state` -- LLM Semantic State Enrichment

**Research question:** Does adding LLM-derived semantic embeddings to the policy's state vector improve threshold decisions?

**Background:** The PolicyNetwork decides thresholds based on a state vector per node. The base state is:

```
s = [x_v, conf, deg, mean_dist, overlap, feat_var]   (dim = feat_dim + 5)
```

LLM state enrichment adds a 16-dimensional projected embedding derived from each node's relational profile:

```
s_enriched = [x_v, conf, deg, mean_dist, overlap, feat_var, e_LLM]   (dim = feat_dim + 5 + 16)
```

**LLM embedding pipeline (preprocessing):**

1. **Node statistics** (`llm/compute_node_statistics.py`): For each node, computes degree percentiles, label disagreement with neighbors, feature variance, cross-relation Jaccard overlap, degree ratios, and z-score extremeness -- all per relation. Training labels only used for disagreement (no data leakage).

2. **Text descriptions** (`llm/generate_descriptions.py`): Converts statistics into natural language summaries. Template mode produces deterministic text like: *"Online review platform user with 12 shared-product connections (percentile 73), 3 same-seller-rating connections (percentile 45)..."*

3. **Sentence encoding** (`llm/encode_embeddings.py`): Encodes descriptions with `all-MiniLM-L6-v2` sentence-transformer into 384-dim vectors, L2 normalized.

4. **Projection** (`llm/projector.py`): During training, `LLMProjector` (384 -> 64 -> 16 MLP) compresses the embedding and is trained end-to-end with the policy network.

**Experiments:**

| Experiment | Config | Description |
|------------|--------|-------------|
| CAPN (base state) | `use_capn=True, use_llm_state=False` | Standard CAPN. State dim = feat_dim + 5 (30 for Amazon, 37 for Yelp). |
| CAPN (LLM state) | `use_capn=True, use_llm_state=True` | LLM-enriched state. State dim = feat_dim + 5 + 16 (46 for Amazon, 53 for Yelp). |

**What to look for:**
- If LLM state helps, the semantic profile captures information not already present in raw features.
- The projector is trained end-to-end, so the model can learn which aspects of the semantic embedding are useful.
- This tests whether LLM-derived representations provide complementary signals to the graph structure.

**Prerequisites:** LLM embeddings must be generated before running this study:
```bash
python -m llm.compute_node_statistics --data {dataset}
python -m llm.generate_descriptions --mode template --data {dataset}
python -m llm.encode_embeddings --data {dataset}
```

---

## Running the Studies

### Single study, single dataset

```bash
python ablation.py --study capn --data yelp --num-epochs 31
```

### Single study, both datasets

```bash
python ablation.py --study capn --data both
```

### All CAPN studies, both datasets

```bash
python ablation.py --study all_capn --data both
```

This runs all 6 CAPN studies: `capn`, `capn_label`, `capn_reward`, `capn_llm`, `capn_lambda`, `capn_llm_state`.

### All studies (including baselines)

```bash
python ablation.py --study all --data both
```

### CLI reference

| Argument | Values | Default | Description |
|----------|--------|---------|-------------|
| `--study` | `inter`, `loss`, `layers`, `baseline`, `capn`, `capn_label`, `capn_reward`, `capn_llm`, `capn_lambda`, `capn_llm_state`, `all`, `all_capn` | required | Which study to run |
| `--data` | `yelp`, `amazon`, `both` | `yelp` | Dataset(s) to evaluate on |
| `--num-epochs` | integer | `31` | Training epochs per experiment |
| `--output-dir` | path | `results/ablation` | Root output directory |

### Output directory structure

When using `--data both`:
```
results/ablation/
  amazon/
    capn/ablation_results_YYYYMMDD_HHMMSS.json
    capn_label/...
    capn_reward/...
    capn_llm/...
    capn_lambda/...
    capn_llm_state/...
  yelp/
    capn/...
    ...
```

When using a single dataset:
```
results/ablation/
  capn/ablation_results_YYYYMMDD_HHMMSS.json
  ...
```

---

## Results Format

Each study produces a JSON file with this structure:

```json
{
  "Experiment Name": {
    "config": {
      "model": "CARE",
      "use_capn": true,
      "...": "overrides from base args"
    },
    "best_metrics": {
      "gnn_f1": 0.612,
      "gnn_accuracy": 0.643,
      "gnn_recall": 0.621,
      "gnn_precision": 0.638,
      "gnn_auc": 0.876,
      "gnn_ap": 0.512,
      "label_f1": 0.589,
      "label_accuracy": 0.601,
      "label_recall": 0.575,
      "label_precision": 0.604,
      "label_auc": 0.831,
      "label_ap": 0.467
    },
    "elapsed_seconds": 1420.5
  }
}
```

The `best_metrics` correspond to the epoch with the highest `gnn_auc`.

---

## Generating Figures

After running studies, generate thesis-quality figures:

```bash
python generate_figures.py
```

This reads all JSON results from `results/ablation/` and generates:

| Output | Description |
|--------|-------------|
| `figures/thesis/{study}_bar.pdf` | Grouped bar chart comparing experiments across metrics |
| `figures/thesis/{study}_radar.pdf` | Radar/spider plot for multi-metric comparison |
| `figures/thesis/{study}_delta.pdf` | Improvement bars vs first experiment (baseline) |
| `figures/thesis/tables/{study}.tex` | LaTeX table with bold best values |
| `figures/thesis/capn_lambda_sensitivity.pdf` | Line plot for lambda sensitivity |
| `figures/thesis/summary_dashboard.pdf` | Single-page overview of all studies |

All figures use publication styling: serif font, 300 DPI, PDF format.
