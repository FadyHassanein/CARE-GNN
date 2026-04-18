"""Compute all Chapter 3.1 stats with consistent conventions.

Conventions:
  - edges: unique undirected pairs (i,j), i<j, self-loops excluded
  - avg_degree: 2*|E|/N (undirected)
  - fraud_nbr_frac: per-node avg of fraud fraction over non-self neighbours
  - homophily (for fraud): mean over fraud nodes of (# fraud nbrs / # nbrs)
"""
import numpy as np
import pickle
from scipy.stats import pointbiserialr


def edge_set(adj):
    edges = set()
    for i in range(len(adj)):
        for j in adj[i]:
            if i == j:
                continue
            a, b = (i, j) if i < j else (j, i)
            edges.add((a, b))
    return edges


def homophily(adj, labels):
    fr_nbr_of_fraud = []
    fr_nbr_of_benign = []
    for i in range(len(adj)):
        nbrs = [j for j in adj[i] if j != i]
        if not nbrs:
            continue
        frac_fraud = (labels[np.array(nbrs)] == 1).mean()
        if labels[i] == 1:
            fr_nbr_of_fraud.append(frac_fraud)
        elif labels[i] == 0:
            fr_nbr_of_benign.append(frac_fraud)
    return float(np.mean(fr_nbr_of_fraud)) * 100, float(np.mean(fr_nbr_of_benign)) * 100


def run(name, labels_path, feat_path, rels, labelled_mask=None):
    labels = np.load(labels_path).astype(int)
    feats = np.load(feat_path)
    N = len(labels)
    print(f"\n### {name} ###")
    print(f"  nodes: {N}, features: {feats.shape[1]}")
    uniq, cnt = np.unique(labels, return_counts=True)
    print(f"  label dist: {dict(zip(uniq.tolist(), cnt.tolist()))}")
    print(f"  fraud rate (all nodes): {(labels == 1).mean() * 100:.2f}%")
    if labelled_mask is not None:
        ly = labels[labelled_mask]
        print(f"  labelled subset (n={len(ly)}): benign={int((ly==0).sum())}, fraud={int((ly==1).sum())}, fraud_rate={(ly==1).mean()*100:.2f}%")

    # label-masked homophily for fair comparison — restrict neighbours-of analysis to labelled pool
    for rel_name, rel_path in rels:
        with open(rel_path, "rb") as f:
            adj = pickle.load(f)
        E = edge_set(adj)
        avg_deg = 2 * len(E) / N
        h_fr, h_be = homophily(adj, labels)
        print(f"  {rel_name}: |E|={len(E):,}, avg_deg={avg_deg:.2f}, fraud_nbr_of_fraud={h_fr:.2f}%, fraud_nbr_of_benign={h_be:.2f}%")

    # feature correlations
    if labelled_mask is not None:
        ly = labels[labelled_mask]
        lf = feats[labelled_mask]
    else:
        ly = labels
        lf = feats
    print(f"  top |r| features (n={len(ly)}):")
    corrs = []
    for d in range(lf.shape[1]):
        r, _ = pointbiserialr(ly, lf[:, d])
        corrs.append((d, r))
    for d, r in sorted(corrs, key=lambda x: -abs(x[1]))[:8]:
        print(f"    feat {d:2d}: r = {r:+.4f}")

    # return edge sets for Jaccard
    return {rel_name: edge_set(pickle.load(open(rel_path, "rb"))) for rel_name, rel_path in rels}


amz_mask = np.arange(11944) >= 3305
amz = run(
    "AMAZON",
    "data/amz_labels.npy",
    "data/amz_features.npy",
    [
        ("UPU", "data/amz_upu_adjlists.pickle"),
        ("USU", "data/amz_usu_adjlists.pickle"),
        ("UVU", "data/amz_uvu_adjlists.pickle"),
    ],
    labelled_mask=amz_mask,
)
ylp = run(
    "YELP",
    "data/yelp_labels.npy",
    "data/yelp_features.npy",
    [
        ("RUR", "data/yelp_rur_adjlists.pickle"),
        ("RTR", "data/yelp_rtr_adjlists.pickle"),
        ("RSR", "data/yelp_rsr_adjlists.pickle"),
    ],
)

print("\n### Amazon Jaccard (undirected unique, no self-loops) ###")
for a, b in [("UPU", "USU"), ("UPU", "UVU"), ("USU", "UVU")]:
    inter = len(amz[a] & amz[b])
    union = len(amz[a] | amz[b])
    print(f"  Jaccard({a},{b}) = {inter/union:.4f}")
