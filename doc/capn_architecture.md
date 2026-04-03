# CAPN Architecture Reference

Technical reference for the Camouflage-Aware Policy Network (CAPN) extension to CARE-GNN.

---

## Table of Contents

1. [Motivation](#motivation)
2. [System Overview](#system-overview)
3. [Components](#components)
   - [PolicyNetwork](#policynetwork)
   - [StateConstructor](#stateconstructor)
   - [EnhancedLabelPredictor](#enhancedlabelpredictor)
   - [MultiViewDistance](#multiviewdistance)
   - [ShapedRewardComputer](#shapedrewardcomputer)
   - [LLMPriorLoader](#llmpriorloader)
   - [LLMProjector](#llmprojector)
4. [Training Flow](#training-flow)
5. [Loss Function](#loss-function)
6. [Gradient Management](#gradient-management)
7. [Configuration Reference](#configuration-reference)

---

## Motivation

CARE-GNN uses a heuristic Bernoulli bandit to set one threshold per relation. This has three limitations:

1. **Coarse granularity:** A single threshold per relation cannot adapt to individual nodes. A high-degree hub and a low-degree leaf get the same filtering ratio.
2. **Stateless decisions:** The RL module sees only the average neighbor distance, not per-node structural signals.
3. **Binary reward:** The -1/0/+1 reward provides weak learning signal, making convergence slow.

CAPN addresses all three by replacing the bandit with a learned policy network that maps per-node state vectors to per-node thresholds via a Beta distribution parameterization.

---

## System Overview

```
                        +-----------------------+
                        |   StateConstructor    |
                        |                       |
  Node features x_v --> | [x_v, conf, deg,      |
  Label scores -------> |  mean_dist, overlap,  | --> state vector s
  Neighbor info ------> |  feat_var, {e_LLM}]   |
                        +-----------------------+
                                  |
                                  v
                        +-----------------------+
                        |   PolicyNetwork       |
                        |                       |
  state s ------------> | Shared MLP -> Per-rel | --> threshold t ~ Beta(alpha, beta)
  relation_idx -------> | head -> (alpha, beta) | --> log_prob for REINFORCE
                        +-----------------------+
                                  |
                                  v
                        +-----------------------+
                        |  Neighbor Filtering   |
                        |                       |
  sample_count = -----> | ceil(num_neighs * t)  | --> filtered neighbors
  MultiViewDistance ---> | sort by distance      |
                        +-----------------------+
                                  |
                                  v
                        +-----------------------+
                        | ShapedRewardComputer  |
                        |                       |
  avg_dist -----------> | R = w1*dist_improve   | --> reward R
  batch_acc ----------> |   + w2*acc_improve    |
  thresholds ---------> |   - w3*regularization |
                        +-----------------------+
                                  |
                                  v
                  L_policy = -R * sum(log_probs)
```

---

## Components

### PolicyNetwork

**File:** `capn.py`

**Purpose:** Maps state vectors to adaptive thresholds for each node-relation pair.

```python
PolicyNetwork(state_dim, hidden_dim=64, num_relations=3, relation_biases=None)
```

**Architecture:**
```
state [batch, state_dim]
  -> Linear(state_dim, hidden_dim) + ReLU
  -> Linear(hidden_dim, hidden_dim // 2) + ReLU
  -> per-relation head: Linear(hidden_dim // 2, 2)
  -> softplus + 0.1 -> (alpha, beta)  # Beta distribution params, floor at 0.1
```

**Forward pass:**
1. Extract shared features via 2-layer MLP
2. Select the relation-specific head based on `relation_idx`
3. Compute `alpha, beta` (both > 0.1 via softplus)
4. **Training (stochastic):** Sample `t ~ Beta(alpha, beta)` using REINFORCE
5. **Inference (deterministic):** Use mean `t = alpha / (alpha + beta)`
6. Return `(thresholds, log_probs)`

**Log probability computation:**
```
log p(t | alpha, beta) = (alpha - 1) * log(t + eps)
                       + (beta - 1) * log(1 - t + eps)
                       - lgamma(alpha) - lgamma(beta)
                       + lgamma(alpha + beta)
```

Uses `SafeLgamma`, a custom autograd function that computes `lgamma` and `digamma` on CPU to avoid CUDA NVRTC JIT compilation errors.

**Policy loss (REINFORCE):**
```
L_policy = -reward * mean(sum_of_log_probs_across_relations)
```

---

### StateConstructor

**File:** `capn.py`

**Purpose:** Constructs the per-node state vector that the PolicyNetwork uses to make threshold decisions.

```python
StateConstructor(adj_lists, homo_adj, features, device,
                 llm_embeddings=None, llm_projector=None)
```

**State vector components (dim = feat_dim + 5 [+ llm_projection_dim]):**

| Component | Dim | Computation |
|-----------|-----|-------------|
| `x_v` | feat_dim | Raw node features (detached from GNN gradient) |
| `conf_v` | 1 | Label predictor confidence: `1 - H(p) / log(C)` where H is entropy, C is num_classes |
| `deg_r(v)` | 1 | Degree in relation r, normalized by `max(1, max_degree)` |
| `mean_dist_r(v)` | 1 | Mean L1 distance to neighbors' label scores in relation r |
| `overlap_r(v)` | 1 | Jaccard overlap: `|N_r(v) & N_homo(v)| / |N_r(v) | N_homo(v)|` |
| `feat_var_r(v)` | 1 | Mean feature variance across neighbors in relation r |
| `e_LLM(v)` | projection_dim | (Optional) LLM semantic embedding projected from 384 -> 16 dim |

**Design decisions:**
- Node features are **detached** from the computation graph to prevent policy gradients from flowing into the GNN feature embeddings.
- `overlap` measures consistency between a specific relation and the homogeneous (all-relations) graph. High overlap = the relation is redundant with the overall graph structure.
- `feat_var` captures neighbor diversity. Low variance = neighbors are similar (potentially a fraud cluster).

---

### EnhancedLabelPredictor

**File:** `capn.py`

**Purpose:** Classifies nodes as fraud/benign based on features. Replaces the linear classifier in CARE-GNN.

```python
EnhancedLabelPredictor(feat_dim, num_classes=2, hidden_dim=None, dropout=0.3)
```

**Architecture:**
```
features [batch, feat_dim]
  -> Linear(feat_dim, hidden_dim) + ReLU + Dropout(0.3)
  -> Linear(hidden_dim, num_classes)
  -> scores [batch, num_classes]
```

Default `hidden_dim = feat_dim * 2`.

Used in two places:
1. **Label loss:** Predictions contribute to `L_label` in the total loss.
2. **State construction:** Prediction confidence feeds into the PolicyNetwork's state.

---

### MultiViewDistance

**File:** `capn.py`

**Purpose:** Computes a composite distance between a center node and its neighbors, combining label-aware and structural signals.

```python
MultiViewDistance(num_relations, gamma_init=0.7, adj_lists=None)
```

**Distance formula:**
```
d(v, u, r) = gamma_r * |s_v - s_u|_1  +  (1 - gamma_r) * (1 - Jaccard(v, u, r))
```

where:
- `|s_v - s_u|_1`: L1 distance between label prediction scores
- `Jaccard(v, u, r) = |N_r(v) & N_r(u)| / |N_r(v) | N_r(u)|`: Structural similarity
- `gamma_r`: Learned mixing parameter (can be initialized from LLM priors)

**Caching:** Jaccard distances are cached in `_jaccard_cache` keyed by `(min(v,u), max(v,u), r)` because Jaccard is symmetric and expensive to compute.

**Comparison with original CARE-GNN:** The original uses only L1 label distance. MultiViewDistance adds the structural Jaccard component, which captures whether two nodes share similar neighborhood structures.

---

### ShapedRewardComputer

**File:** `capn.py`

**Purpose:** Computes a scalar reward for the policy network based on multiple signals.

```python
ShapedRewardComputer(w1=0.5, w2=0.3, w3=0.2, ema_decay=0.95)
```

**Reward formula:**
```
R = w1 * clamp(delta_dist / (prev_dist + eps), -1, 1)
  + w2 * (batch_acc - baseline_acc)
  - w3 * mean(|threshold - 0.5|)
```

**Component breakdown:**

| Component | Weight | Range | Signal |
|-----------|--------|-------|--------|
| Distance improvement | w1=0.5 | [-1, 1] | Did the thresholds improve neighbor separability? Relative improvement over previous epoch. |
| Accuracy signal | w2=0.3 | ~[-1, 1] | Is the model classifying better than its recent average? EMA baseline (decay=0.95) prevents reward from always being positive. |
| Threshold regularization | w3=0.2 | [0, 0.5] | Penalty for extreme thresholds. Prevents collapse to 0.0 (discard all neighbors) or 1.0 (keep all). |

**Epoch reset:** `reset_epoch()` clears `prev_avg_dist` (distance tracking) but **preserves** the EMA baseline accuracy, so the accuracy component remains calibrated across epochs.

---

### LLMPriorLoader

**File:** `capn.py`

**Purpose:** Loads LLM-generated prior knowledge to initialize CAPN components.

```python
LLMPriorLoader(priors_file=None)
```

**Methods:**
- `get_relation_biases()`: Returns `[camouflage_risk_r1, camouflage_risk_r2, camouflage_risk_r3]`. Used as initial biases in PolicyNetwork's per-relation heads.
- `get_gamma_init()`: Returns `[importance_r1, importance_r2, importance_r3]`. Used to initialize MultiViewDistance's gamma mixing parameters.

If no file is provided, returns `None` (default initialization used).

---

### LLMProjector

**File:** `llm/projector.py`

**Purpose:** Projects high-dimensional sentence-transformer embeddings into a compact representation for the policy state.

```python
LLMProjector(input_dim=384, projection_dim=16, dropout=0.1)
```

**Architecture:**
```
embedding [batch, 384]
  -> Linear(384, 64) + ReLU + Dropout(0.1)
  -> Linear(64, projection_dim)
  -> projected [batch, 16]
```

Trained end-to-end via the policy gradient. The projector's parameters are included in the policy optimizer.

---

## Training Flow

```
For each epoch:
  reward_computer.reset_epoch()     # clear distance tracking, keep EMA
  undersample training set

  For each batch:
    1. model.forward(batch_nodes, labels, train_flag=True)
       |-- InterAgg.forward():
       |   |-- Compute label scores for all unique nodes
       |   |-- For each relation r:
       |   |   |-- StateConstructor.compute_state(nodes, r, scores, neigh_scores)
       |   |   |-- PolicyNetwork.forward(state, r) -> (thresholds, log_probs)
       |   |   |-- sample_count = ceil(num_neighbors * threshold)
       |   |   |-- IntraAgg.forward(nodes, neighs, scores, sample_count)
       |   |   |   |-- filter_neighs_ada_threshold() with MultiViewDistance
       |   |   |   |-- sparse aggregation
       |   |-- Inter-aggregation (combine relation embeddings)
       |   |-- Store log_probs, avg_dist, thresholds
       |-- Return (gnn_scores, label_scores)

    2. Compute reward:
       batch_acc = accuracy(gnn_scores, labels)
       reward = reward_computer.compute_reward(avg_dist, batch_acc, thresholds)

    3. Compute loss:
       L_gnn = loss_fn(gnn_scores, labels)
       L_label = loss_fn(label_scores, labels)
       L_policy = policy_network.get_policy_loss(reward)
       L_total = L_gnn + lambda_1 * L_label + lambda_policy * L_policy

    4. Backward + gradient clipping:
       L_total.backward()
       clip_grad_norm(gnn_params, max_norm=1.0)     # GNN: tight clipping
       clip_grad_norm(policy_params, max_norm=5.0)   # Policy: looser clipping
       optimizer.step()

  Evaluate on val/test set every test_epochs
  Early stopping based on val metric
  LR scheduling (ReduceLROnPlateau)
```

---

## Loss Function

**Standard CARE-GNN:**
```
L = L_gnn + lambda_1 * L_label + lambda_2 * ||W||_2
```

**CAPN extension:**
```
L = L_gnn + lambda_1 * L_label + lambda_2 * ||W||_2 + lambda_policy * L_policy
```

where `L_policy = -reward * mean(log_probs)` (REINFORCE gradient estimator).

| Term | Default weight | Purpose |
|------|---------------|---------|
| `L_gnn` | 1.0 | Cross-entropy on GNN final predictions |
| `L_label` | lambda_1 = 2.0 | Cross-entropy on label predictor. Trains the classifier used for neighbor scoring. |
| `||W||_2` | lambda_2 = 1e-3 | L2 weight decay |
| `L_policy` | lambda_policy = 0.1 | Policy gradient loss. Trains the PolicyNetwork to set better thresholds. |

---

## Gradient Management

CAPN uses **separate gradient clipping** for GNN and policy parameters:

| Parameter group | Max norm | Rationale |
|----------------|----------|-----------|
| GNN parameters | 1.0 | Standard tight clipping for supervised learning |
| Policy parameters | 5.0 | REINFORCE gradients are inherently high-variance. Tight clipping (1.0) crushes them, preventing the policy from learning. |

**Dual optimizer setup:**
```python
gnn_optimizer = Adam(gnn_params, lr=0.01, weight_decay=1e-3)
policy_optimizer = Adam(policy_params, lr=1e-3)  # separate, lower LR
```

The policy uses a lower learning rate (1e-3 vs 0.01) because REINFORCE updates are noisier than supervised gradients.

---

## Configuration Reference

All CAPN-specific config fields in `config.py`:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `use_capn` | bool | False | Enable CAPN policy network |
| `policy_lr` | float | 1e-3 | Policy network learning rate |
| `policy_hidden` | int | 64 | Policy MLP hidden dimension |
| `lambda_policy` | float | 0.1 | Policy loss weight in total loss |
| `policy_grad_clip` | float | 5.0 | Gradient clip norm for policy params |
| `reward_w1` | float | 0.5 | Distance improvement reward weight |
| `reward_w2` | float | 0.3 | Accuracy signal reward weight |
| `reward_w3` | float | 0.2 | Regularization reward weight |
| `gamma_init` | float | 0.7 | MultiViewDistance L1/Jaccard mixing |
| `llm_priors_file` | str | '' | Path to LLM priors JSON |
| `use_llm_state` | bool | False | Enable LLM semantic state enrichment |
| `llm_embedding_path` | str | '' | Path to LLM embeddings (auto-resolved to `llm_embeddings/{data}/llm_semantic_embeddings.pt`) |
| `llm_projection_dim` | int | 16 | LLM embedding projection dimension |
| `use_mlp_label` | bool | False | Use MLP label predictor without full CAPN |
