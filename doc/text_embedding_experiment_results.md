# Text Embedding Experiment Results — YelpChi Dataset

## Objective

Investigate whether integrating raw review text (via sentence embeddings) into the CAPN policy network improves fraud detection performance on the YelpChi dataset.

## Background

- Recovered 45,954 raw review texts from the original YelpChi dataset and mapped them 1:1 to the existing node indices in `YelpChi.mat`
- Verified mapping correctness via R-U-R relation (100% same-user match) and label alignment
- Generated 384-dim sentence embeddings using `all-MiniLM-L6-v2` (L2-normalized)
- Text embeddings are projected and concatenated into the CAPN policy network's state vector

## Results

| Model | Val AUC | Test AUC | Test F1 | Test AP | Description |
|-------|---------|----------|---------|---------|-------------|
| CARE-GNN (original RL) | 0.7705 | 0.7660 | 0.6098 | 0.3779 | Baseline with heuristic RL thresholds |
| CAPN (no text) | 0.7759 | 0.7727 | 0.5858 | 0.3997 | Policy network, no text features |
| CAPN + text (16-dim proj) | 0.7744 | 0.7714 | 0.5901 | 0.3972 | 384→16 MLP projection into state |
| CAPN + text (64-dim deep proj, Phase 1) | 0.7737 | 0.7703 | 0.5851 | 0.3938 | 384→256→128→64 with LayerNorm |
| CAPN + text + soft attn (Phase 2) | 0.7738 | 0.7698 | 0.5630 | 0.3903 | Soft attention neighbor weighting |
| CAPN + text + soft + RL fix (Phase 3) | 0.7734 | 0.7694 | 0.5829 | 0.3872 | +advantage norm, entropy reg, tuned LR |

## Phase 1: Projection Bottleneck Fix

**Hypothesis:** The 384→16 projection (99.95% reduction) destroys too much semantic information. Increasing to 64-dim with a deeper projector (3-layer MLP + LayerNorm) should preserve more signal.

**Changes:**
- `llm/projector.py`: Added `deep_mlp` mode (384→256→128→64 with LayerNorm + ReLU + Dropout at each layer)
- `config.py`: `llm_projection_dim` default changed from 16 to 64
- State vector increased from 53D to 101D (text embeddings now 63% of state vs 30%)

**Result:** No improvement. Val AUC 0.7737 vs 0.7744 (16-dim). Within noise.

**Interpretation:** The projection bottleneck was not the binding constraint. Even with 4x more embedding dimensions, the policy network cannot extract useful signal from the text.

## Phase 2: Soft Attention Neighbor Weighting

**Hypothesis:** The hard top-K neighbor selection (threshold controls only cardinality) gives the policy network a weak control lever. Replacing with soft attention (threshold as temperature) lets the policy control *how* neighbors are weighted, not just *how many*.

**Changes:**
- `layers.py`: Added `_forward_soft_attn()` to `IntraAgg` — computes attention weights as `softmax(-distance / temperature)` over ALL neighbors
- Policy threshold reinterpreted as temperature: low temp = sharp focus on closest neighbors, high temp = more uniform
- `config.py` / `train.py`: Added `--soft-attn` flag

**Result:** No improvement. Val AUC 0.7738 vs 0.7737 (Phase 1). Within noise.

**Interpretation:** Soft vs hard neighbor selection doesn't matter when the underlying distance metric (label-aware L1 score) already provides a reasonable ranking. The policy network's temperature output has the same effect as cardinality control.

## Phase 3: Stronger Policy Learning

**Hypothesis:** REINFORCE with scalar batch reward is too noisy for the policy to learn meaningful thresholds. Advantage normalization + entropy regularization should stabilize learning.

**Changes:**
- `capn.py` ShapedRewardComputer: Added running EMA mean/std for advantage normalization (`advantage = (reward - mean) / std`)
- `capn.py` PolicyNetwork: Added entropy regularization (`-λ_H * H(π)` with binary entropy proxy) to prevent threshold collapse
- `config.py`: `lambda_policy` increased 0.1→0.3, `policy_lr` increased 1e-3→3e-3

**Result:** No improvement. Val AUC 0.7734 vs 0.7738 (Phase 2). Within noise.

**Interpretation:** Stronger policy gradients don't help because there isn't a gradient signal to amplify — the text embeddings simply don't contain discriminative information that the existing features miss.

## Analysis: Why Text Embeddings Don't Help

### 1. The 32 handcrafted features already capture the spam signal

YelpChi's features (Table 2, "Collective Opinion Spam Detection") include:
- **Review features (0-14):** Rank, review date, extremity, deviation, early time frame, ISR, positive/negative content words, length, first person pronouns, review sentiment
- **User features (15-23):** Max/avg number of reviews, writing/rating deviation, burstiness
- **Product features (24-31):** Same statistics aggregated at product level

These behavioral/statistical features directly capture spam patterns (burstiness, rating deviation, extremity). A sentence embedding of "Great food, amazing service!" doesn't add information beyond what the sentiment score and word count already encode.

### 2. Review text content is weakly discriminative for spam

Fraud review examples from the dataset:
- "one word... AMAZING!!!" (22 chars)
- "Great food & service. I've returned many times..." (404 chars)
- "I went there this evening for dinner and it was perfect!" (1085 chars)

These read like normal reviews. Spam on Yelp is often *semantically plausible* text — the spam signal is in the **pattern** (who posted, when, how many, rating consistency) not the **content**. `all-MiniLM-L6-v2` encodes general semantic meaning, not fraud-specific linguistic patterns.

### 3. The policy network is the wrong place for text features

Text embeddings feed into the policy network's state vector, which outputs neighbor filtering thresholds. This is an indirect path:
```
text → projection → state → threshold → neighbor selection → aggregation → prediction
```
Even if text contained useful signal, it would need to influence *how neighbors are selected*, not *the final prediction*. There's no mechanism for text to directly improve classification — it can only change which neighbors get aggregated.

### 4. Comparison with FLAG paper approach

The FLAG paper (KDD 2025) uses text differently:
- LLM extracts **discriminative text** (not raw embeddings) — specifically prompted to identify fraud-relevant patterns
- Text is used for **semantic similarity neighbor sampling** — directly selecting neighbors by text similarity
- Text features are **fine-tuned** end-to-end with the GNN, not frozen and projected

Our approach (frozen sentence embeddings → projected → policy state) is fundamentally more limited.

## Recommendations for Thesis

1. **Report these results honestly** — negative results are valuable. The finding that text doesn't help through the policy network is a valid contribution showing the limits of the CAPN architecture.

2. **Consider feature-level enrichment** — instead of routing text through the policy, concatenate text embeddings directly to node features before the GNN. This gives the GNN direct access to semantic signal for classification.

3. **Consider LLM-based discriminative features** — following FLAG, use an LLM to extract fraud-specific textual features (not generic embeddings) such as: "review mentions specific dishes" (legit signal) vs "review uses generic superlatives" (spam signal).

4. **Frame as architectural insight** — CAPN's policy network benefits from structural features (degree, overlap, variance) that inform *neighbor selection*, not from content features that inform *classification*. This is a meaningful finding about the separation of concerns in GNN fraud detection.

## Files Modified

| File | Changes |
|------|---------|
| `data/yelp_review_texts.json` | 45,954 raw review texts mapped to node indices |
| `llm_embeddings/yelp/llm_semantic_embeddings.pt` | Text-based 384-dim sentence embeddings |
| `llm_embeddings/yelp/llm_semantic_embeddings_text.pt` | Same (explicit backup) |
| `llm_embeddings/yelp/llm_semantic_embeddings_structural.pt` | Original structural embeddings (backup) |
| `llm/projector.py` | Added `deep_mlp` mode with LayerNorm |
| `layers.py` | Added `_forward_soft_attn()` for soft attention aggregation |
| `capn.py` | Added advantage normalization + entropy regularization |
| `config.py` | Updated defaults: `llm_projection_dim=64`, `lambda_policy=0.3`, `policy_lr=3e-3` |
| `train.py` | Added `--soft-attn` flag, updated argparse defaults |

## Reproduction

```bash
# Phase 1: Deep projector (64-dim)
python train.py --data yelp --use-capn --use-llm-state --llm-projection-dim 64

# Phase 2: + Soft attention
python train.py --data yelp --use-capn --use-llm-state --llm-projection-dim 64 --soft-attn

# Phase 3: + RL improvements
python train.py --data yelp --use-capn --use-llm-state --llm-projection-dim 64 --soft-attn --lambda-policy 0.3 --policy-lr 3e-3
```

All runs use `C:/Users/fadyh/miniconda3/envs/torch_cuda/python.exe` environment.
