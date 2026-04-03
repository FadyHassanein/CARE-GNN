# Commands Reference

Quick reference for all runnable commands in the CARE-GNN/CAPN project.

---

## Training

```bash
# Standard CARE-GNN on Yelp
python train.py --data yelp --model CARE --inter GNN

# CARE-GNN on Amazon (smaller batch size)
python train.py --data amazon --model CARE --batch-size 256

# CAPN (policy network) on Yelp
python train.py --data yelp --model CARE --use-capn

# CAPN on Amazon
python train.py --data amazon --model CARE --use-capn --batch-size 256

# CAPN with LLM priors
python train.py --data yelp --model CARE --use-capn --llm-priors-file data/llm_priors/yelp_priors.json

# CAPN with LLM semantic state enrichment
python train.py --data amazon --model CARE --use-capn --use-llm-state --batch-size 256

# GraphSAGE baseline
python train.py --data yelp --model SAGE

# Multi-layer CARE
python train.py --data yelp --model MULTI_CARE --num-layers 2
```

---

## LLM Embedding Pipeline

Must be run before `--use-llm-state`. Run once per dataset.

```bash
# Step 1: Compute node statistics
python -m llm.compute_node_statistics --data amazon
python -m llm.compute_node_statistics --data yelp

# Step 2: Generate text descriptions
python -m llm.generate_descriptions --mode template --data amazon
python -m llm.generate_descriptions --mode template --data yelp

# Step 3: Encode into sentence embeddings
python -m llm.encode_embeddings --data amazon
python -m llm.encode_embeddings --data yelp
```

Output structure:
```
llm_embeddings/
  amazon/
    node_statistics.npz        # [11944, ...] structured arrays
    node_descriptions.json     # {node_id: text} mapping
    llm_semantic_embeddings.pt # [11944, 384] float32 tensor
  yelp/
    node_statistics.npz        # [45954, ...]
    node_descriptions.json
    llm_semantic_embeddings.pt # [45954, 384]
```

---

## Ablation Studies

```bash
# Single study, single dataset
python ablation.py --study capn --data yelp

# Single study, both datasets
python ablation.py --study capn --data both

# All CAPN studies, both datasets
python ablation.py --study all_capn --data both

# All studies (baselines + CAPN), both datasets
python ablation.py --study all --data both

# Custom epochs
python ablation.py --study capn --data both --num-epochs 50
```

Available studies: `inter`, `loss`, `layers`, `baseline`, `capn`, `capn_label`, `capn_reward`, `capn_llm`, `capn_lambda`, `capn_llm_state`, `all`, `all_capn`.

---

## Figure Generation

```bash
# Generate all thesis figures from results
python generate_figures.py
```

Output: `figures/thesis/` (PDFs + LaTeX tables).

---

## Environment

```bash
# Activate conda environment
conda activate torch_cuda

# Install dependencies
pip install -r requirements.txt
```
