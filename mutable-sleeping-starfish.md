# Improved LLM State Enrichment for CAPN (v2)

## Context

### Problem
The v1 LLM state enrichment pipeline (sentence-transformer embeddings) provides no meaningful gain on either dataset:
- Amazon: -0.07pp AUC (negligible)
- Yelp: -0.40pp AUC (actively hurts)

**Root cause**: The pipeline converts 20 precise numerical graph features into text descriptions, encodes them with a generic sentence-transformer (`all-MiniLM-L6-v2`), then projects the 384-dim embedding down to 16 dims. This is a lossy round-trip that: (a) destroys numerical precision, (b) re-encodes information already in the state, and (c) uses an encoder that has no notion of fraud relevance.

### Solution: Hybrid Direct Features + Claude Reasoning
Replace the sentence-transformer bottleneck with two complementary enrichment channels:

1. **Direct Graph Structural Features** — The existing `node_statistics.npz` features (plus new 2-hop and ego-density features) injected numerically into the state via an MLP projector. No text round-trip.

2. **Claude API Reasoning Scores** — Instead of using an LLM to *embed text*, use Claude to *reason about fraud patterns*. For each node, send its structural profile to Claude and receive structured JSON risk assessments (6 scores). Claude's value is in understanding pattern interactions, not in embedding.

This produces a 2×2 ablation table for the thesis:
| | No structural features | + Structural features |
|---|---|---|
| **No Claude reasoning** | CAPN baseline | CAPN + structural |
| **+ Claude reasoning** | CAPN + reasoning | CAPN + both |

### Thesis narrative
*"We first showed that generic text embeddings from sentence-transformers don't improve CAPN (Table X). Diagnosing the failure as a lossy numerical-to-text-to-embedding round-trip, we replaced it with two targeted approaches: (1) direct structural features capturing cross-relation graph topology not visible in the per-relation state vector, and (2) LLM reasoning scores where Claude analyzes each node's structural profile and produces fraud-relevant risk assessments. Both independently improve performance, and combine for the best result."*

## Key Integration Points
- **State construction**: `capn.py:75-183` — `StateConstructor.__init__()` and `compute_state()`
- **Existing LLM loading**: `train.py:233-251` — loads `.pt`, creates projector
- **State dim**: `train.py:259` — `state_dim = feat_data.shape[1] + 5`
- **Optimizer**: `train.py:313-332` — dual optimizer with policy_param_groups
- **Config**: `config.py:80-85` — existing LLM state fields
- **Ablation**: `ablation.py:172-177` — `get_capn_llm_state_experiments()`
- **Node stats**: `llm/compute_node_statistics.py` — already computes 20 features per node
- **LLM API infra**: `llm/generate_descriptions.py:198-234` — Anthropic client, checkpointing, rate limiting
- **Projector**: `llm/projector.py` — `LLMProjector(input_dim, projection_dim)` 2-layer MLP

---

## Deliverable 1: Extend `llm/compute_node_statistics.py` — Add 2-hop + Ego-density

Add two new feature groups to the existing preprocessing script:

### 1a. 2-hop neighborhood size per relation (3 features)
For each node `v` and relation `R`: count unique nodes reachable in exactly 2 hops (excluding `v` and its direct neighbors). This captures how deeply embedded the node is in the graph. Fraud rings often form dense local clusters visible at 2 hops.

```python
# Pseudocode
for r_idx, adj_list in enumerate(adj_lists):
    for node in range(num_nodes):
        direct_neighs = adj_list.get(node, set())
        two_hop = set()
        for n in direct_neighs:
            two_hop |= adj_list.get(n, set())
        two_hop -= direct_neighs
        two_hop.discard(node)
        two_hop_size[node, r_idx] = len(two_hop)
```

### 1b. Ego-network density per relation (3 features)
For each node `v` and relation `R`: what fraction of possible edges among `v`'s neighbors actually exist? High density → clique-like structure → possible coordinated behavior.

```python
# Pseudocode (cap neighbor count for performance)
MAX_NEIGHS = 200  # sample if more
for r_idx, adj_list in enumerate(adj_lists):
    for node in range(num_nodes):
        neighs = list(adj_list.get(node, set()))
        if len(neighs) > MAX_NEIGHS:
            neighs = random.sample(neighs, MAX_NEIGHS)
        n = len(neighs)
        if n < 2:
            continue
        edges = sum(1 for i, ni in enumerate(neighs) for nj in neighs[i+1:]
                    if nj in adj_list.get(ni, set()))
        ego_density[node, r_idx] = edges / (n * (n - 1) / 2)
```

### Updated output
`node_statistics.npz` gains two new arrays:
- `two_hop_size`: shape `[N, 3]`
- `ego_density`: shape `[N, 3]`

Total features per node: 20 (existing) + 3 + 3 = **26**

---

## Deliverable 2: New file `llm/graph_features.py` — Load & Stack Features

New file. Loads the `.npz` and stacks all features into a single PyTorch tensor.

```python
def load_graph_features(data='amazon', stats_dir=None) -> torch.Tensor:
    """Load precomputed graph structural features as a tensor.
    
    Returns: tensor of shape [N, 26], float32
    """
    if stats_dir is None:
        stats_dir = f'llm_embeddings/{data}'
    stats = np.load(os.path.join(stats_dir, 'node_statistics.npz'))
    
    features = np.column_stack([
        stats['degrees'],              # [N, 3] per-relation degree
        stats['degree_percentiles'],   # [N, 3] relative rank
        stats['label_disagreement'],   # [N, 3] neighbor homophily (training labels)
        stats['neighbor_feat_var'],    # [N, 3] neighbor feature variance
        stats['cross_overlap'],        # [N, 3] Jaccard between relation pairs
        stats['degree_ratios'],        # [N, 3] cross-relation degree ratios
        stats['num_extreme_features'][:, None],  # [N, 1]
        stats['mean_abs_zscore'][:, None],       # [N, 1]
        stats['two_hop_size'],         # [N, 3] 2-hop reach
        stats['ego_density'],          # [N, 3] local clustering
    ])  # total: 26 features
    
    # normalize each feature to [0, 1] range
    mins = features.min(axis=0, keepdims=True)
    maxs = features.max(axis=0, keepdims=True)
    features = (features - mins) / np.maximum(maxs - mins, 1e-8)
    
    return torch.tensor(features, dtype=torch.float32)
```

**Output**: Returns `[N, 26]` tensor ready for the projector.

---

## Deliverable 3: New file `llm/generate_risk_scores.py` — Claude API Reasoning

New file. For each node, sends its structural statistics to Claude and receives structured JSON risk assessments. Two modes, following the pattern in `generate_descriptions.py`.

### Mode A: Template (heuristic scoring, no API)
Deterministic formulas that compute the same 6 scores using simple rules:

```python
def compute_template_scores(node_stats: dict) -> dict:
    """Heuristic approximation of Claude reasoning scores."""
    # structural_anomaly: based on degree percentile extremes
    deg_percs = node_stats['degree_percentiles']
    structural_anomaly = max(abs(p - 50) / 50 for p in deg_percs)
    
    # relation_consistency: 1 - variance of normalized degrees across relations
    deg_norm = [d / max(d_max, 1) for d, d_max in zip(degrees, max_degrees)]
    relation_consistency = 1.0 - np.std(deg_norm) / max(np.mean(deg_norm), 1e-8)
    
    # neighborhood_risk: mean label disagreement across relations
    neighborhood_risk = np.mean(node_stats['label_disagreement'])
    
    # feature_anomaly: normalized mean |z-score|
    feature_anomaly = min(node_stats['mean_abs_zscore'] / 3.0, 1.0)
    
    # coordination_signal: based on ego-network density
    coordination_signal = np.mean(node_stats['ego_density'])
    
    # isolation_score: inverse of 2-hop reach (normalized)
    isolation_score = 1.0 - min(np.mean(node_stats['two_hop_size']) / max_2hop, 1.0)
    
    return {scores clipped to [0, 1]}
```

### Mode B: Claude API reasoning
Reuses the Anthropic client + checkpointing infrastructure from `generate_descriptions.py`.

**Prompt per node:**
```
You are analyzing a node in a {dataset_name} review network for fraud detection.
Based on these structural statistics, output ONLY a JSON object with 6 scores (0.0 to 1.0).

Node #{node_id} structural profile:
- {relation_1_name}: degree={d1} ({p1}th percentile), label_disagreement={ld1:.3f}, 
  neighbor_variance={v1:.3f}, 2-hop_reach={h1}, ego_density={e1:.3f}
- {relation_2_name}: degree={d2} ({p2}th percentile), label_disagreement={ld2:.3f},
  neighbor_variance={v2:.3f}, 2-hop_reach={h2}, ego_density={e2:.3f}
- {relation_3_name}: degree={d3} ({p3}th percentile), label_disagreement={ld3:.3f},
  neighbor_variance={v3:.3f}, 2-hop_reach={h3}, ego_density={e3:.3f}
- Cross-relation overlap: {pair1}={j1:.3f}, {pair2}={j2:.3f}, {pair3}={j3:.3f}
- Degree ratios: {pair1}={r1:.2f}, {pair2}={r2:.2f}, {pair3}={r3:.2f}
- Feature anomaly: {n_extreme} extreme features, mean |z-score|={mz:.3f}

Score definitions:
- structural_anomaly: How unusual is this node's connectivity pattern (0=typical, 1=extreme outlier)
- relation_consistency: How consistent is behavior across relations (0=very inconsistent, 1=uniform)
- neighborhood_risk: How suspicious are the neighbors (0=clean neighborhood, 1=mostly fraudulent neighbors)
- feature_anomaly: How statistically unusual are the node's features (0=normal, 1=extreme outlier)
- coordination_signal: Likelihood of coordinated/organized behavior (0=independent, 1=strong coordination)
- isolation_score: Structural isolation vs embeddedness (0=deeply embedded, 1=isolated/peripheral)

Output ONLY valid JSON, no explanation:
```

**Claude model**: `claude-haiku-4-5-20251001` (cheapest, sufficient for structured scoring)
**Temperature**: 0 (deterministic)
**Max tokens**: 150 (JSON only)
**Checkpointing**: Every 500 nodes (reuse pattern from `generate_descriptions.py`)
**Rate limiting**: 0.05s delay (Haiku is fast)
**Error handling**: Falls back to template scores on API failure
**Batching**: 5 nodes per request to reduce API calls (5× fewer calls)

**Output**: `llm_embeddings/{data}/llm_risk_scores.pt` — shape `[N, 6]`, float32

### Cost estimate
| Dataset | Nodes | Input tokens | Output tokens | Estimated cost |
|---------|-------|-------------|--------------|----------------|
| Amazon | 11,944 | ~3.6M | ~1.8M | ~$3 |
| Yelp | 45,954 | ~13.8M | ~6.9M | ~$10 |

**Runnable**: `python -m llm.generate_risk_scores --data amazon --mode template|llm`

---

## Deliverable 4: Modify `llm/projector.py` — Flexible Projector

Update `LLMProjector` to handle variable input dimensions. The existing 2-layer MLP works, just needs to accept different `input_dim` values:

```python
class LLMProjector(nn.Module):
    def __init__(self, input_dim=384, projection_dim=16, dropout=0.1):
        super().__init__()
        hidden_dim = max(32, input_dim * 2)  # scale hidden with input
        hidden_dim = min(hidden_dim, 128)     # cap at 128
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, projection_dim),
        )
```

No change needed if input_dim is passed correctly — the current code already accepts variable `input_dim`. The only change is the hidden_dim scaling logic to avoid a 384→64 bottleneck being applied when input is only 26 dims.

---

## Deliverable 5: Modify `config.py` — New Flags

Add to the existing CAPN config section:

```python
# Enrichment mode: 'none', 'structural', 'reasoning', 'both'
enrichment_mode: str = 'none'
graph_features_path: str = ''     # resolved at runtime to llm_embeddings/{data}/node_statistics.npz
risk_scores_path: str = ''        # resolved at runtime to llm_embeddings/{data}/llm_risk_scores.pt
structural_projection_dim: int = 16
```

Keep existing `use_llm_state` and `llm_embedding_path` for backward compatibility with v1 sentence-transformer path (needed for comparison ablation).

CLI args in `train.py`:
```
--enrichment-mode {none,structural,reasoning,both}
--graph-features-path PATH
--risk-scores-path PATH
--structural-projection-dim INT
```

---

## Deliverable 6: Modify `train.py` — Loading & Integration

In the CAPN initialization block (after line 233):

```python
# v2 enrichment: graph structural features and/or Claude reasoning scores
graph_features_tensor = None
risk_scores_tensor = None
enrichment_projector = None

if args.enrichment_mode in ('structural', 'both'):
    from llm.graph_features import load_graph_features
    gf_path = args.graph_features_path or f'llm_embeddings/{args.data}/node_statistics.npz'
    graph_features_tensor = load_graph_features(args.data, os.path.dirname(gf_path)).to(device)
    logger.info(f'Loaded graph structural features: {graph_features_tensor.shape}')

if args.enrichment_mode in ('reasoning', 'both'):
    rs_path = args.risk_scores_path or f'llm_embeddings/{args.data}/llm_risk_scores.pt'
    risk_scores_tensor = torch.load(rs_path, weights_only=True).to(device)
    logger.info(f'Loaded Claude reasoning scores: {risk_scores_tensor.shape}')

# combine enrichment sources
enrichment_dim = 0
if graph_features_tensor is not None and risk_scores_tensor is not None:
    enrichment_tensor = torch.cat([graph_features_tensor, risk_scores_tensor], dim=1)  # [N, 32]
    enrichment_dim = enrichment_tensor.shape[1]
elif graph_features_tensor is not None:
    enrichment_tensor = graph_features_tensor  # [N, 26]
    enrichment_dim = enrichment_tensor.shape[1]
elif risk_scores_tensor is not None:
    enrichment_tensor = risk_scores_tensor  # [N, 6]
    enrichment_dim = enrichment_tensor.shape[1]

if enrichment_dim > 0:
    enrichment_projector = LLMProjector(
        input_dim=enrichment_dim,
        projection_dim=args.structural_projection_dim).to(device)
    logger.info(f'Enrichment projector: {enrichment_dim} -> {args.structural_projection_dim}')
```

Update `state_dim`:
```python
state_dim = feat_data.shape[1] + 5
if args.use_llm_state:                    # v1 path
    state_dim += args.llm_projection_dim
if enrichment_dim > 0:                     # v2 path
    state_dim += args.structural_projection_dim
```

Pass to `StateConstructor` (extend its init to accept enrichment params alongside existing llm params).

Add enrichment projector params to `policy_param_groups`.

---

## Deliverable 7: Modify `capn.py` — StateConstructor

Extend `StateConstructor.__init__()` to accept enrichment tensors:

```python
def __init__(self, adj_lists, homo_adj, features, device=None,
             llm_embeddings=None, llm_projector=None,
             enrichment_tensor=None, enrichment_projector=None):
```

In `compute_state()`, after the existing LLM block (line 176):

```python
# v2 enrichment: graph structural features + Claude reasoning scores
if self.enrichment_tensor is not None and self.enrichment_projector is not None:
    node_indices = torch.LongTensor(nodes).to(self.enrichment_tensor.device)
    enrich_raw = self.enrichment_tensor[node_indices]
    enrich_proj = self.enrichment_projector(enrich_raw.to(self.device))
    components.append(enrich_proj)
```

---

## Deliverable 8: Modify `ablation.py` — New Study

Add a new `capn_enrichment` study with the 4 combinations:

```python
def get_capn_enrichment_experiments():
    """A7: State enrichment — structural features, Claude reasoning, both."""
    return [
        {'name': 'CAPN (baseline)',
         'model': 'CARE', 'use_capn': True, 'enrichment_mode': 'none'},
        {'name': 'CAPN + structural',
         'model': 'CARE', 'use_capn': True, 'enrichment_mode': 'structural'},
        {'name': 'CAPN + reasoning',
         'model': 'CARE', 'use_capn': True, 'enrichment_mode': 'reasoning'},
        {'name': 'CAPN + both',
         'model': 'CARE', 'use_capn': True, 'enrichment_mode': 'both'},
    ]
```

Register in `studies` dict and add to `capn_studies` list.

---

## Files Modified
| File | Change |
|------|--------|
| `llm/compute_node_statistics.py` | Add 2-hop neighborhood + ego-density computation (2 new arrays in .npz) |
| `llm/projector.py` | Scale hidden_dim with input_dim (minor) |
| `config.py` | Add `enrichment_mode`, `graph_features_path`, `risk_scores_path`, `structural_projection_dim` |
| `train.py` | Add CLI args, load graph features + risk scores, create enrichment projector, update state_dim + optimizer |
| `capn.py` | Extend `StateConstructor` to accept/use enrichment_tensor + enrichment_projector |
| `ablation.py` | Add `get_capn_enrichment_experiments()`, register study |

## Files Created
| File | Purpose |
|------|---------|
| `llm/graph_features.py` | Load .npz, stack features into normalized tensor |
| `llm/generate_risk_scores.py` | Claude API reasoning scores (Mode A: template heuristic, Mode B: API) |

## Verification

### Preprocessing
```bash
# 1. Recompute statistics with new 2-hop + ego-density features
python -m llm.compute_node_statistics --data amazon
python -m llm.compute_node_statistics --data yelp

# 2. Generate Claude reasoning scores (template mode for testing, llm for production)
python -m llm.generate_risk_scores --data amazon --mode template
python -m llm.generate_risk_scores --data yelp --mode template

# 3. (Optional) Generate with actual Claude API
python -m llm.generate_risk_scores --data amazon --mode llm
python -m llm.generate_risk_scores --data yelp --mode llm
```

### Training (smoke test)
```bash
# v2 structural only
python train.py --data amazon --model CARE --use-capn --enrichment-mode structural --batch-size 256 --num-epoch 5

# v2 reasoning only
python train.py --data amazon --model CARE --use-capn --enrichment-mode reasoning --batch-size 256 --num-epoch 5

# v2 both
python train.py --data amazon --model CARE --use-capn --enrichment-mode both --batch-size 256 --num-epoch 5
```

### Full ablation
```bash
python ablation.py --study capn_enrichment --data both
```

### Expected outcomes
- `enrichment_mode=structural` should improve over baseline (cross-relation features + label disagreement add genuinely new signal)
- `enrichment_mode=reasoning` should improve over baseline (Claude identifies pattern interactions)
- `enrichment_mode=both` should be best (complementary signals)
- All 3 should outperform v1 sentence-transformer approach
- No data leakage: label_disagreement uses only training labels, reasoning scores derived from training-label-based stats
