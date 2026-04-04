"""
Load precomputed graph structural features from node_statistics.npz as a PyTorch tensor.

Used by the v2 enrichment pipeline to bypass the sentence-transformer bottleneck
and inject structural features directly into the CAPN policy state.

Usage:
    from llm.graph_features import load_graph_features
    features = load_graph_features('amazon')  # [N, 26] tensor
"""

import os

import numpy as np
import torch


def load_graph_features(data='amazon', stats_dir=None):
    """Load precomputed graph structural features as a normalized tensor.

    :param data: dataset name ('amazon' or 'yelp')
    :param stats_dir: directory containing node_statistics.npz (default: llm_embeddings/{data})
    :return: tensor of shape [N, num_features], float32, normalized to [0, 1]
    """
    if stats_dir is None:
        stats_dir = f'llm_embeddings/{data}'
    stats_path = os.path.join(stats_dir, 'node_statistics.npz')
    stats = np.load(stats_path)

    columns = [
        stats['degrees'],              # [N, 3] per-relation degree
        stats['degree_percentiles'],   # [N, 3] relative rank
        stats['label_disagreement'],   # [N, 3] neighbor homophily
        stats['neighbor_feat_var'],    # [N, 3] neighbor feature variance
        stats['cross_overlap'],        # [N, 3] Jaccard between relation pairs
        stats['degree_ratios'],        # [N, 3] cross-relation degree ratios
        stats['num_extreme_features'][:, None],  # [N, 1]
        stats['mean_abs_zscore'][:, None],       # [N, 1]
    ]

    # add 2-hop and ego-density if present (v2 statistics)
    if 'two_hop_size' in stats:
        columns.append(stats['two_hop_size'])   # [N, 3]
    if 'ego_density' in stats:
        columns.append(stats['ego_density'])    # [N, 3]

    features = np.column_stack(columns)

    # min-max normalize each feature to [0, 1]
    mins = features.min(axis=0, keepdims=True)
    maxs = features.max(axis=0, keepdims=True)
    ranges = maxs - mins
    ranges[ranges < 1e-8] = 1.0  # avoid division by zero for constant features
    features = (features - mins) / ranges

    return torch.tensor(features, dtype=torch.float32)
