# CLAUDE.md — CARE-GNN Development Guide

## Project Overview

**CARE-GNN** (Camouflage-REsistant Graph Neural Network) is a PyTorch-based fraud detection system for heterogeneous multi-relation graphs. It identifies fraudulent nodes that deliberately mimic legitimate behavior ("camouflaged fraudsters"). Based on the CIKM 2020 paper by Dou et al.

The project has three evolutionary stages:
1. **CARE-GNN (Original)**: Heuristic RL-based neighbor filtering per relation
2. **CAPN Extension**: Learned policy network producing per-node adaptive thresholds
3. **LLM Enrichment (v2)**: Direct structural features + Claude API reasoning scores

## Repository Structure

```
├── model.py              # CARE-GNN model classes (OneLayerCARE, MultiLayerCARE, CAPNOneLayerCARE)
├── layers.py             # GNN layers: InterAgg (cross-relation), IntraAgg (within-relation)
├── capn.py               # CAPN policy network, state constructor, reward shaping
├── graphsage.py          # GraphSAGE baseline model
├── losses.py             # CrossEntropy, FocalLoss, WeightedCrossEntropy
├── config.py             # CareConfig dataclass — all hyperparameters
├── train.py              # Main training entry point with CLI argument parsing
├── utils.py              # Data loading, evaluation metrics, train/test splitting
├── ablation.py           # Systematic ablation study framework
├── cross_validation.py   # K-fold cross-validation
├── preprocess.py         # General data preprocessing (.mat → pickle/npy)
├── data_process.py       # Data format conversion
├── amazon_preprocess.py  # Amazon-specific preprocessing
├── simi_comp.py          # Similarity metric computation
├── llm_priors.py         # Domain-specific relation priors generation
├── generate_figures.py   # Thesis figure and LaTeX table generation
├── visualization.py      # Training curves, confusion matrices
│
├── llm/                  # LLM state enrichment pipeline
│   ├── compute_node_statistics.py  # Precompute per-node relational stats (20+ features)
│   ├── generate_descriptions.py    # Stats → text descriptions (template or LLM)
│   ├── encode_embeddings.py        # Descriptions → 384-dim sentence embeddings
│   ├── generate_risk_scores.py     # 6-dim risk scores (template or Claude API)
│   ├── graph_features.py           # Load/normalize precomputed structural features
│   └── projector.py                # MLP projecting embeddings to lower dimensions
│
├── data/                 # Datasets: Amazon.mat, YelpChi.mat, preprocessed pickles/npy
├── llm_embeddings/       # Generated LLM features per dataset (runtime)
├── checkpoints/          # Saved model checkpoints (runtime)
├── logs/                 # Training logs (runtime)
├── results/              # Ablation study results (runtime)
├── doc/                  # Additional documentation
│   ├── commands.md       # CLI command reference
│   ├── capn_architecture.md  # CAPN technical deep-dive
│   └── ablation_studies.md   # Ablation results and interpretation
└── mutable-sleeping-starfish.md  # v2 enrichment design specification
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Preprocess data (converts .mat to adjacency lists)
python preprocess.py

# Train standard CARE-GNN
python train.py --data yelp --model CARE --inter GNN

# Train with CAPN policy network
python train.py --data yelp --model CARE --use-capn

# Train with LLM enrichment (requires precomputed embeddings)
python train.py --data amazon --model CARE --use-capn --use-llm-state --batch-size 256

# GraphSAGE baseline
python train.py --data yelp --model SAGE
```

## Common Commands

```bash
# Ablation studies
python ablation.py --study capn --data yelp
python ablation.py --study all --data both

# Cross-validation
python cross_validation.py --num-folds 5 --data yelp --model CARE

# LLM enrichment pipeline (run in order)
python -m llm.compute_node_statistics --data amazon
python -m llm.generate_descriptions --mode template --data amazon
python -m llm.encode_embeddings --data amazon
python -m llm.generate_risk_scores --data amazon --mode template

# v2 enrichment training
python train.py --data amazon --model CARE --use-capn --enrichment-mode structural --batch-size 256
python train.py --data amazon --model CARE --use-capn --enrichment-mode both --batch-size 256
```

## Supported Datasets

| Dataset | Nodes | Features | Fraud % | Relations | Batch Size |
|---------|-------|----------|---------|-----------|------------|
| Yelp    | 45,954 | 32 | 13.0% | RUR, RTR, RSR | 1024 (default) |
| Amazon  | 11,944 | 25 | 5.3%  | UPU, USU, UVU | 256 (recommended) |

Amazon nodes 0–3304 are unlabeled and excluded from evaluation.

## Data Flow

1. `.mat` files → `preprocess.py` → pickle adjacency lists + `.npy` features/labels
2. `utils.py:load_data()` loads preprocessed files into memory
3. LLM pipeline (optional): `compute_node_statistics` → `generate_descriptions` → `encode_embeddings` / `generate_risk_scores`
4. Training reads features, adjacency lists, and optional LLM enrichments

## Architecture Notes

- **InterAgg** combines embeddings across relations (modes: Mean, GNN, Att, Weight)
- **IntraAgg** aggregates neighbor features within a single relation
- **EnhancedLabelPredictor** computes non-linear label-aware similarity scores
- **PolicyNetwork** (CAPN) uses Beta distribution to sample per-node thresholds
- **StateConstructor** builds state vectors from node features + relational statistics + optional LLM embeddings
- All models support `device` parameter for CPU/GPU placement

## Configuration

All hyperparameters are in `config.py` as the `CareConfig` dataclass. Key groups:

- **Model**: `emb_size=64`, `inter='GNN'`
- **Training**: `lr=0.01`, `num_epochs=31`, `dropout=0.6`, `grad_clip=1.0`
- **Loss**: `loss='ce'` (or `'focal'`, `'weighted_ce'`), `lambda_1=2.0`
- **CAPN**: `use_capn=False`, `policy_lr=1e-3`, `policy_hidden=64`
- **LLM v1**: `use_llm_state=False`, `llm_projection_dim=16`
- **LLM v2**: `enrichment_mode='none'` (or `'structural'`, `'reasoning'`, `'both'`)
- **Splits**: `test_size=0.60`, `val_size=0.15`

CLI arguments in `train.py` mirror these fields.

## Code Conventions

- **Style**: `snake_case` for variables/functions, `PascalCase` for classes
- **Imports**: stdlib → third-party (torch, numpy, scipy, sklearn) → local modules
- **Logging**: Use `logging` module (`logger.info()`, `logger.warning()`, etc.)
- **Device handling**: Explicit `device` parameter; use `.to(device)` on tensors and models
- **Docstrings**: Triple-quoted with parameter descriptions and return types
- **Type hints**: Minimal but present in key functions; docstrings carry type info

## Evaluation Metrics

- AUC-ROC, AUC-PR, F1, Precision, Recall, Accuracy
- Computed via `utils.py` using scikit-learn
- Early stopping monitors best AUC on validation set (patience=10)
- Cross-validation reports mean ± std across folds

## Dependencies

```
torch>=1.8.0
numpy>=1.19.0
scipy>=1.5.0
scikit-learn>=0.24.0
matplotlib>=3.3.0
sentence-transformers>=2.2.0  # v1 LLM path
anthropic>=0.18.0             # Claude API for v2 risk scores
```

## Key Design Decisions

1. **Neighbor filtering is the core innovation** — thresholds control which neighbors contribute to aggregation, resisting camouflage
2. **CAPN replaces heuristic RL** — per-node adaptive thresholds via learned policy network outperform global per-relation thresholds
3. **v2 enrichment avoids text bottleneck** — direct numerical features + structured Claude reasoning scores instead of lossy sentence embeddings
4. **Modular composition** — CARE runs without CAPN, CAPN runs without LLM; each layer is independently toggleable
