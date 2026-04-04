"""
Deliverable 1: Compute per-node relational statistics for LLM semantic enrichment.

Runs ONCE as preprocessing. For each node, computes:
- Degree + percentile rank per relation
- Mean label disagreement with neighbors per relation (training labels only)
- Neighbor feature variance per relation
- Cross-relation neighbor overlap (Jaccard)
- Cross-relation degree ratios
- Feature extremeness (z-score stats using training-set statistics)
- 2-hop neighborhood size per relation
- Ego-network density per relation

Usage:
    python -m llm.compute_node_statistics --data amazon
    python -m llm.compute_node_statistics --data yelp
"""

import argparse
import logging
import os
import random
import sys

import numpy as np
from scipy.stats import percentileofscore
from sklearn.model_selection import train_test_split

# add project root to path so we can import utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_data

logger = logging.getLogger(__name__)

DATASET_CONFIG = {
    'amazon': {
        'relation_names': ['UPU', 'USU', 'UVU'],
        'cross_pair_names': ['UPU-USU', 'UPU-UVU', 'USU-UVU'],
        'label_offset': 3305,  # nodes 0-3304 are unlabeled
        'num_features': 25,
    },
    'yelp': {
        'relation_names': ['RUR', 'RTR', 'RSR'],
        'cross_pair_names': ['RUR-RTR', 'RUR-RSR', 'RTR-RSR'],
        'label_offset': 0,  # all nodes are labeled
        'num_features': 32,
    },
}


def get_train_split(labels, data='amazon', val_size=0.15):
    """Replicate the exact train/val/test split from train.py.

    Amazon: nodes 0-3304 are unlabeled; 3305+ are labeled.
    Yelp: all nodes are labeled (offset=0).
    Uses the same random_state=2 and stratification as train.py.
    """
    offset = DATASET_CONFIG[data]['label_offset']
    index = list(range(offset, len(labels)))
    all_labels = labels[offset:]

    idx_train, idx_temp, y_train, y_temp = train_test_split(
        index, all_labels, stratify=all_labels,
        test_size=val_size + 0.60, random_state=2, shuffle=True)

    val_fraction = val_size / (val_size + 0.60)
    idx_val, idx_test, y_val, y_test = train_test_split(
        idx_temp, y_temp, stratify=y_temp,
        test_size=1 - val_fraction, random_state=2, shuffle=True)

    return set(idx_train), set(idx_val), set(idx_test)


def compute_statistics(data='amazon', prefix='data/', output_dir=None):
    """Compute all per-node relational statistics."""
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    if data not in DATASET_CONFIG:
        raise ValueError(f'Unknown dataset: {data}. Must be one of {list(DATASET_CONFIG.keys())}')

    cfg = DATASET_CONFIG[data]
    if output_dir is None:
        output_dir = f'llm_embeddings/{data}'

    logger.info(f'Loading {data} dataset...')
    adj_lists_all, feat_data, labels = load_data(data, prefix)
    homo, relation1, relation2, relation3 = adj_lists_all
    adj_lists = [relation1, relation2, relation3]

    num_nodes = len(labels)
    num_features = feat_data.shape[1]
    num_relations = len(adj_lists)
    relation_names = cfg['relation_names']
    logger.info(f'Dataset: {num_nodes} nodes, {num_features} features, {num_relations} relations ({relation_names})')

    # get training set for label-leakage prevention
    train_set, val_set, test_set = get_train_split(labels, data=data)
    logger.info(f'Split: train={len(train_set)}, val={len(val_set)}, test={len(test_set)}')

    # --- 1. Degree and percentile rank per relation ---
    logger.info('Computing degrees and percentile ranks...')
    degrees = np.zeros((num_nodes, num_relations), dtype=np.float32)
    for r_idx, adj_list in enumerate(adj_lists):
        for node in range(num_nodes):
            degrees[node, r_idx] = len(adj_list.get(node, set()))

    # percentile ranks (computed over all nodes)
    degree_percentiles = np.zeros((num_nodes, num_relations), dtype=np.float32)
    for r_idx in range(num_relations):
        deg_col = degrees[:, r_idx]
        for node in range(num_nodes):
            degree_percentiles[node, r_idx] = percentileofscore(deg_col, deg_col[node], kind='rank')

    # --- 2. Mean label disagreement per relation (training labels only) ---
    logger.info('Computing label disagreement (training labels only)...')
    label_disagreement = np.zeros((num_nodes, num_relations), dtype=np.float32)

    for r_idx, adj_list in enumerate(adj_lists):
        for node in range(num_nodes):
            neighs = adj_list.get(node, set())
            if not neighs:
                continue

            if node in train_set:
                # training node: use actual label disagreement with training neighbors
                node_label = labels[node]
                train_neighs = [n for n in neighs if n in train_set]
                if train_neighs:
                    neigh_labels = np.array([labels[n] for n in train_neighs])
                    label_disagreement[node, r_idx] = np.mean(neigh_labels != node_label)
            else:
                # val/test node: use fraction of suspicious training neighbors as proxy
                train_neighs = [n for n in neighs if n in train_set]
                if train_neighs:
                    neigh_labels = np.array([labels[n] for n in train_neighs])
                    label_disagreement[node, r_idx] = np.mean(neigh_labels == 1)

    # --- 3. Neighbor feature variance per relation ---
    logger.info('Computing neighbor feature variance...')
    neighbor_feat_var = np.zeros((num_nodes, num_relations), dtype=np.float32)

    for r_idx, adj_list in enumerate(adj_lists):
        for node in range(num_nodes):
            neighs = list(adj_list.get(node, set()))
            if len(neighs) < 2:
                continue
            neigh_feats = feat_data[neighs]
            neighbor_feat_var[node, r_idx] = np.mean(np.var(neigh_feats, axis=0))

    # --- 4. Cross-relation neighbor overlap (Jaccard) ---
    logger.info('Computing cross-relation neighbor overlap...')
    # pairs: (UPU,USU), (UPU,UVU), (USU,UVU)
    cross_pairs = [(0, 1), (0, 2), (1, 2)]
    cross_overlap = np.zeros((num_nodes, len(cross_pairs)), dtype=np.float32)

    for pair_idx, (r_a, r_b) in enumerate(cross_pairs):
        adj_a = adj_lists[r_a]
        adj_b = adj_lists[r_b]
        for node in range(num_nodes):
            neighs_a = adj_a.get(node, set())
            neighs_b = adj_b.get(node, set())
            union_size = len(neighs_a | neighs_b)
            if union_size > 0:
                cross_overlap[node, pair_idx] = len(neighs_a & neighs_b) / union_size

    # --- 5. Cross-relation degree ratios ---
    logger.info('Computing cross-relation degree ratios...')
    # ratios: UPU/USU, UPU/UVU, USU/UVU
    degree_ratios = np.zeros((num_nodes, len(cross_pairs)), dtype=np.float32)
    for pair_idx, (r_a, r_b) in enumerate(cross_pairs):
        for node in range(num_nodes):
            deg_b = degrees[node, r_b]
            if deg_b > 0:
                degree_ratios[node, pair_idx] = degrees[node, r_a] / deg_b
            else:
                degree_ratios[node, pair_idx] = 0.0

    # --- 6 & 7. Feature extremeness (z-score using training stats) ---
    logger.info('Computing feature z-scores (training stats only)...')
    train_indices = sorted(train_set)
    train_feats = feat_data[train_indices]
    feat_mean = np.mean(train_feats, axis=0)
    feat_std = np.std(train_feats, axis=0)
    feat_std[feat_std < 1e-8] = 1.0  # avoid division by zero

    z_scores = np.abs((feat_data - feat_mean) / feat_std)

    # number of features with |z| > 2
    num_extreme_features = np.sum(z_scores > 2.0, axis=1).astype(np.float32)
    # mean absolute z-score
    mean_abs_zscore = np.mean(z_scores, axis=1).astype(np.float32)

    # --- 8. 2-hop neighborhood size per relation ---
    logger.info('Computing 2-hop neighborhood sizes...')
    two_hop_size = np.zeros((num_nodes, num_relations), dtype=np.float32)

    for r_idx, adj_list in enumerate(adj_lists):
        for node in range(num_nodes):
            direct_neighs = adj_list.get(node, set())
            if not direct_neighs:
                continue
            two_hop = set()
            for n in direct_neighs:
                two_hop.update(adj_list.get(n, set()))
            two_hop -= direct_neighs
            two_hop.discard(node)
            two_hop_size[node, r_idx] = len(two_hop)
        logger.info(f'  {relation_names[r_idx]}: mean_2hop={two_hop_size[:, r_idx].mean():.1f}, '
                    f'max_2hop={two_hop_size[:, r_idx].max():.0f}')

    # --- 9. Ego-network density per relation ---
    logger.info('Computing ego-network density...')
    ego_density = np.zeros((num_nodes, num_relations), dtype=np.float32)
    MAX_EGO_NEIGHS = 200  # sample if more to keep O(n) manageable

    for r_idx, adj_list in enumerate(adj_lists):
        for node in range(num_nodes):
            neighs = list(adj_list.get(node, set()))
            n = len(neighs)
            if n < 2:
                continue
            # sample for very high-degree nodes
            if n > MAX_EGO_NEIGHS:
                neighs = random.sample(neighs, MAX_EGO_NEIGHS)
                n = MAX_EGO_NEIGHS
            # count edges among neighbors (exclude self-loops and source node)
            neigh_set = set(neighs)
            neigh_set.discard(node)  # exclude source node if present
            neighs_clean = list(neigh_set)
            n = len(neighs_clean)
            if n < 2:
                continue
            edges = 0
            for ni in neighs_clean:
                ni_neighs = adj_list.get(ni, set())
                # only count edges to other neighbors (not self, not source)
                edges += len((ni_neighs & neigh_set) - {ni})
            edges //= 2  # each edge counted twice
            max_edges = n * (n - 1) // 2
            ego_density[node, r_idx] = min(edges / max_edges, 1.0)
        logger.info(f'  {relation_names[r_idx]}: mean_density={ego_density[:, r_idx].mean():.4f}, '
                    f'max_density={ego_density[:, r_idx].max():.4f}')

    # --- Save all statistics ---
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'node_statistics.npz')

    cross_pair_names = cfg['cross_pair_names']

    np.savez(
        output_path,
        degrees=degrees,                        # [N, 3]
        degree_percentiles=degree_percentiles,  # [N, 3]
        label_disagreement=label_disagreement,  # [N, 3]
        neighbor_feat_var=neighbor_feat_var,     # [N, 3]
        cross_overlap=cross_overlap,             # [N, 3]
        degree_ratios=degree_ratios,             # [N, 3]
        num_extreme_features=num_extreme_features,  # [N]
        mean_abs_zscore=mean_abs_zscore,         # [N]
        two_hop_size=two_hop_size,               # [N, 3]
        ego_density=ego_density,                 # [N, 3]
        relation_names=np.array(relation_names),
        cross_pair_names=np.array(cross_pair_names),
        dataset=np.array(data),
    )
    logger.info(f'Saved statistics to {output_path}')

    # --- Sanity check ---
    logger.info('=== Sanity Check ===')
    logger.info(f'Num nodes: {num_nodes}')
    for r_idx, name in enumerate(relation_names):
        logger.info(f'{name}: mean_deg={degrees[:, r_idx].mean():.1f}, '
                    f'max_deg={degrees[:, r_idx].max():.0f}, '
                    f'mean_disagree={label_disagreement[:, r_idx].mean():.4f}, '
                    f'mean_feat_var={neighbor_feat_var[:, r_idx].mean():.4f}')
    for pair_idx, name in enumerate(cross_pair_names):
        logger.info(f'Overlap {name}: mean={cross_overlap[:, pair_idx].mean():.4f}, '
                    f'max={cross_overlap[:, pair_idx].max():.4f}')
        logger.info(f'Deg ratio {name}: mean={degree_ratios[:, pair_idx].mean():.4f}, '
                    f'max={degree_ratios[:, pair_idx].max():.4f}')
    logger.info(f'Extreme features: mean={num_extreme_features.mean():.2f}, '
                f'max={num_extreme_features.max():.0f}')
    logger.info(f'Mean |z-score|: mean={mean_abs_zscore.mean():.4f}, '
                f'max={mean_abs_zscore.max():.4f}')
    for r_idx, name in enumerate(relation_names):
        logger.info(f'{name}: mean_2hop={two_hop_size[:, r_idx].mean():.1f}, '
                    f'mean_ego_density={ego_density[:, r_idx].mean():.4f}')

    return output_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute per-node relational statistics')
    parser.add_argument('--data', type=str, default='amazon', choices=['amazon', 'yelp'],
                        help='Dataset name')
    parser.add_argument('--prefix', type=str, default='data/', help='Data directory prefix')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: llm_embeddings/{data})')
    args = parser.parse_args()
    compute_statistics(args.data, args.prefix, args.output_dir)
