"""Verify remaining Chapter 3.1 claims: feature correlations and Jaccard overlap."""
import numpy as np
import pickle
from scipy.stats import pointbiserialr


def edge_set(adj):
    """Return set of undirected edges (i<j) for a symmetric adjacency list."""
    edges = set()
    for i, nbrs in adj.items() if isinstance(adj, dict) else enumerate(adj):
        for j in nbrs:
            if i < j:
                edges.add((i, j))
            elif j < i:
                edges.add((j, i))
    return edges


def jaccard(a, b):
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union > 0 else 0.0


# Amazon feature correlations (labelled nodes only)
labels = np.load("data/amz_labels.npy").astype(int)
features = np.load("data/amz_features.npy")
mask = np.arange(len(labels)) >= 3305
y = labels[mask]
X = features[mask]

print("=== AMAZON feature correlations (labelled nodes only, n={}) ===".format(len(y)))
corrs = []
for d in range(X.shape[1]):
    r, _ = pointbiserialr(y, X[:, d])
    corrs.append((d, r))
corrs_sorted = sorted(corrs, key=lambda x: -abs(x[1]))
for d, r in corrs_sorted[:12]:
    print(f"  feature {d:2d}: r = {r:+.4f}")

# Jaccard overlap between relation edge sets
print("\n=== AMAZON relation edge-set Jaccard ===")
adjs = {}
for rel in ["upu", "usu", "uvu"]:
    with open(f"data/amz_{rel}_adjlists.pickle", "rb") as f:
        adjs[rel] = edge_set(pickle.load(f))
    print(f"  |E_{rel}| unique undirected: {len(adjs[rel])}")

pairs = [("upu", "usu"), ("upu", "uvu"), ("usu", "uvu")]
for a, b in pairs:
    print(f"  Jaccard({a}, {b}) = {jaccard(adjs[a], adjs[b]):.4f}")

# Yelp feature correlations
ylabels = np.load("data/yelp_labels.npy").astype(int)
yfeatures = np.load("data/yelp_features.npy")
print("\n=== YELP feature correlations (all nodes, n={}) ===".format(len(ylabels)))
corrs = []
for d in range(yfeatures.shape[1]):
    r, _ = pointbiserialr(ylabels, yfeatures[:, d])
    corrs.append((d, r))
corrs_sorted = sorted(corrs, key=lambda x: -abs(x[1]))
for d, r in corrs_sorted[:10]:
    print(f"  feature {d:2d}: r = {r:+.4f}")
