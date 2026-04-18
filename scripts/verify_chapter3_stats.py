"""Verify dataset statistics claimed in Chapter 3.1."""
import numpy as np
import pickle


def run_dataset(name, label_path, feature_path, rels):
    labels = np.load(label_path)
    features = np.load(feature_path)
    print(f"=== {name} ===")
    print(f"  nodes: {len(labels)}  features: {features.shape[1]}")
    uniq, counts = np.unique(labels, return_counts=True)
    print(f"  label distribution: {dict(zip(uniq.tolist(), counts.tolist()))}")
    print(f"  fraud rate (all): {(labels == 1).mean():.4f}")
    if name == "AMAZON":
        print(f"  labels in [0,3305): {np.unique(labels[:3305], return_counts=True)}")
        print(f"  labels in [3305:]:  {np.unique(labels[3305:], return_counts=True)}")
        print(f"  fraud rate labelled (>=3305): {(labels[3305:] == 1).mean():.4f}")

    for rel_name, rel_path in rels:
        with open(rel_path, "rb") as f:
            adj = pickle.load(f)
        deg = np.array([len(adj[i]) for i in range(len(adj))])
        edges_dir = int(deg.sum())
        avg_deg = float(deg.mean())

        fr_fr, be_fr = [], []
        for i in range(len(adj)):
            neigh = list(adj[i])
            if not neigh:
                continue
            nbrs = np.array(neigh)
            nbr_labels = labels[nbrs]
            pct_fraud = (nbr_labels == 1).mean()
            if labels[i] == 1:
                fr_fr.append(pct_fraud)
            elif labels[i] == 0:
                be_fr.append(pct_fraud)

        print(
            f"  {rel_name}: dir_edges={edges_dir}, undir~={edges_dir // 2}, "
            f"avg_deg={avg_deg:.2f}, %fraud_nbrs_of_fraud={np.mean(fr_fr) * 100:.2f}, "
            f"%fraud_nbrs_of_benign={np.mean(be_fr) * 100:.2f}"
        )


if __name__ == "__main__":
    run_dataset(
        "AMAZON",
        "data/amz_labels.npy",
        "data/amz_features.npy",
        [
            ("UPU", "data/amz_upu_adjlists.pickle"),
            ("USU", "data/amz_usu_adjlists.pickle"),
            ("UVU", "data/amz_uvu_adjlists.pickle"),
        ],
    )
    run_dataset(
        "YELP",
        "data/yelp_labels.npy",
        "data/yelp_features.npy",
        [
            ("RUR", "data/yelp_rur_adjlists.pickle"),
            ("RTR", "data/yelp_rtr_adjlists.pickle"),
            ("RSR", "data/yelp_rsr_adjlists.pickle"),
        ],
    )
