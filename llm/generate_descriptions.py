"""
Deliverable 2: Generate text descriptions from node statistics.

Two modes:
  Mode A (template): Deterministic text construction, no API needed.
  Mode B (llm): Anthropic API calls with checkpointing every 500 nodes.

Usage:
    python -m llm.generate_descriptions --mode template
    python -m llm.generate_descriptions --mode llm
"""

import argparse
import json
import logging
import os
import sys
import time

import numpy as np

logger = logging.getLogger(__name__)

DATASET_CONFIG = {
    'amazon': {
        'relation_names': ['UPU', 'USU', 'UVU'],
        'relation_descriptions': {
            'UPU': 'shared-product connections (users who reviewed the same products)',
            'USU': 'same-seller-rating connections (users who gave same rating to same seller within 1 week)',
            'UVU': 'similar-review-text connections (users whose review texts are highly similar)',
        },
        'cross_pair_names': ['UPU-USU', 'UPU-UVU', 'USU-UVU'],
        'platform': 'Online review platform',
        'num_features': 25,
    },
    'yelp': {
        'relation_names': ['RUR', 'RTR', 'RSR'],
        'relation_descriptions': {
            'RUR': 'shared-restaurant connections (users who reviewed the same restaurants)',
            'RTR': 'shared-rating-time connections (users with same star level under same timestamp)',
            'RSR': 'shared-rating-star connections (users who gave same star rating to same restaurant)',
        },
        'cross_pair_names': ['RUR-RTR', 'RUR-RSR', 'RTR-RSR'],
        'platform': 'Restaurant review platform (Yelp)',
        'num_features': 32,
    },
}


def load_statistics(stats_path='llm_embeddings/node_statistics.npz'):
    """Load precomputed node statistics."""
    data = np.load(stats_path, allow_pickle=True)
    return {
        'degrees': data['degrees'],
        'degree_percentiles': data['degree_percentiles'],
        'label_disagreement': data['label_disagreement'],
        'neighbor_feat_var': data['neighbor_feat_var'],
        'cross_overlap': data['cross_overlap'],
        'degree_ratios': data['degree_ratios'],
        'num_extreme_features': data['num_extreme_features'],
        'mean_abs_zscore': data['mean_abs_zscore'],
    }


def generate_template_description(node_idx, stats, data='amazon'):
    """Generate a template-based text description for a single node."""
    cfg = DATASET_CONFIG[data]
    rn = cfg['relation_names']
    rd = cfg['relation_descriptions']
    cpn = cfg['cross_pair_names']

    degrees = stats['degrees'][node_idx]
    percentiles = stats['degree_percentiles'][node_idx]
    disagreement = stats['label_disagreement'][node_idx]
    feat_var = stats['neighbor_feat_var'][node_idx]
    overlap = stats['cross_overlap'][node_idx]
    deg_ratios = stats['degree_ratios'][node_idx]
    num_extreme = stats['num_extreme_features'][node_idx]
    mean_zscore = stats['mean_abs_zscore'][node_idx]

    parts = []

    # connectivity profile
    parts.append(
        f"{cfg['platform']} user with "
        f"{int(degrees[0])} {rd[rn[0]].split('(')[0].strip()} (percentile {percentiles[0]:.0f}), "
        f"{int(degrees[1])} {rd[rn[1]].split('(')[0].strip()} (percentile {percentiles[1]:.0f}), "
        f"and {int(degrees[2])} {rd[rn[2]].split('(')[0].strip()} (percentile {percentiles[2]:.0f})."
    )

    # label disagreement
    parts.append(
        f"Neighborhood label disagreement: "
        f"{disagreement[0]:.2f} {rn[0]}, {disagreement[1]:.2f} {rn[1]}, {disagreement[2]:.2f} {rn[2]}."
    )

    # cross-relation patterns
    parts.append(
        f"Cross-relation neighbor overlap: "
        f"{overlap[0]:.2f} {cpn[0]}, {overlap[1]:.2f} {cpn[1]}, {overlap[2]:.2f} {cpn[2]}."
    )

    # degree ratios
    parts.append(
        f"Degree ratios: "
        f"{deg_ratios[0]:.2f} {rn[0]}/{rn[1]}, {deg_ratios[1]:.2f} {rn[0]}/{rn[2]}, {deg_ratios[2]:.2f} {rn[1]}/{rn[2]}."
    )

    # neighbor feature variance
    parts.append(
        f"Neighbor feature variance: "
        f"{feat_var[0]:.4f} {rn[0]}, {feat_var[1]:.4f} {rn[1]}, {feat_var[2]:.4f} {rn[2]}."
    )

    # feature extremeness
    parts.append(
        f"User has {int(num_extreme)} features with extreme values "
        f"and mean absolute deviation of {mean_zscore:.2f}."
    )

    return " ".join(parts)


def build_llm_prompt(node_idx, stats, data='amazon'):
    """Build a prompt for the LLM to assess a node's relational profile."""
    cfg = DATASET_CONFIG[data]
    rn = cfg['relation_names']
    rd = cfg['relation_descriptions']
    cpn = cfg['cross_pair_names']

    degrees = stats['degrees'][node_idx]
    percentiles = stats['degree_percentiles'][node_idx]
    disagreement = stats['label_disagreement'][node_idx]
    overlap = stats['cross_overlap'][node_idx]
    deg_ratios = stats['degree_ratios'][node_idx]
    num_extreme = stats['num_extreme_features'][node_idx]
    mean_zscore = stats['mean_abs_zscore'][node_idx]

    prompt = f"""You are analyzing a user on a {cfg['platform'].lower()} for potential fraudulent behavior.
The platform has three types of user-to-user relationships:
- {rn[0]}: {rd[rn[0]]}
- {rn[1]}: {rd[rn[1]]}
- {rn[2]}: {rd[rn[2]]}

Here is the relational profile for user #{node_idx}:

CONNECTIVITY:
- {rn[0]}: {int(degrees[0])} connections (percentile {percentiles[0]:.0f})
- {rn[1]}: {int(degrees[1])} connections (percentile {percentiles[1]:.0f})
- {rn[2]}: {int(degrees[2])} connections (percentile {percentiles[2]:.0f})

NEIGHBORHOOD QUALITY (label disagreement with neighbors, higher = more suspicious neighbors):
- {rn[0]}: {disagreement[0]:.3f}
- {rn[1]}: {disagreement[1]:.3f}
- {rn[2]}: {disagreement[2]:.3f}

CROSS-RELATION PATTERNS:
- {cpn[0]} neighbor overlap (Jaccard): {overlap[0]:.3f}
- {cpn[1]} neighbor overlap (Jaccard): {overlap[1]:.3f}
- {cpn[2]} neighbor overlap (Jaccard): {overlap[2]:.3f}
- Degree ratios: {rn[0]}/{rn[1]}={deg_ratios[0]:.3f}, {rn[0]}/{rn[2]}={deg_ratios[1]:.3f}, {rn[1]}/{rn[2]}={deg_ratios[2]:.3f}

FEATURE EXTREMENESS:
- Features with |z-score| > 2: {int(num_extreme)} out of {cfg['num_features']}
- Mean absolute z-score: {mean_zscore:.3f}

In 2-3 sentences, assess whether this user's relational pattern suggests suspicious behavior. \
Focus on cross-relation inconsistencies, unusual connectivity patterns, and neighborhood quality signals."""

    return prompt


def generate_descriptions_template(stats, num_nodes, data='amazon'):
    """Generate all descriptions using template mode."""
    descriptions = {}
    for node_idx in range(num_nodes):
        descriptions[str(node_idx)] = generate_template_description(node_idx, stats, data=data)
        if (node_idx + 1) % 2000 == 0:
            logger.info(f'Template descriptions: {node_idx + 1}/{num_nodes}')
    return descriptions


def generate_descriptions_llm(stats, num_nodes, output_path, checkpoint_interval=500, data='amazon'):
    """Generate descriptions using Anthropic API with checkpointing."""
    try:
        import anthropic
    except ImportError:
        logger.error('anthropic package not installed. Run: pip install anthropic')
        logger.info('Falling back to template mode.')
        return generate_descriptions_template(stats, num_nodes, data=data)

    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        logger.error('ANTHROPIC_API_KEY environment variable not set.')
        logger.info('Falling back to template mode.')
        return generate_descriptions_template(stats, num_nodes, data=data)

    client = anthropic.Anthropic(api_key=api_key)

    # load checkpoint if exists
    descriptions = {}
    checkpoint_path = output_path + '.checkpoint'
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            descriptions = json.load(f)
        logger.info(f'Resumed from checkpoint: {len(descriptions)} nodes already processed')

    for node_idx in range(num_nodes):
        node_key = str(node_idx)
        if node_key in descriptions:
            continue

        prompt = build_llm_prompt(node_idx, stats, data=data)

        try:
            message = client.messages.create(
                model='claude-sonnet-4-20250514',
                max_tokens=200,
                messages=[{'role': 'user', 'content': prompt}],
            )
            descriptions[node_key] = message.content[0].text
        except Exception as e:
            logger.warning(f'LLM API error for node {node_idx}: {e}. Using template fallback.')
            descriptions[node_key] = generate_template_description(node_idx, stats, data=data)

        # checkpoint every N nodes
        if (node_idx + 1) % checkpoint_interval == 0:
            with open(checkpoint_path, 'w') as f:
                json.dump(descriptions, f)
            logger.info(f'Checkpoint saved: {node_idx + 1}/{num_nodes} nodes processed')

        # rate limiting: small delay between API calls
        if node_key not in descriptions or (node_idx + 1) % 10 == 0:
            time.sleep(0.1)

    # clean up checkpoint file
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    return descriptions


def main():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(description='Generate text descriptions from node statistics')
    parser.add_argument('--mode', type=str, required=True, choices=['template', 'llm'],
                        help='Generation mode: template (deterministic) or llm (API-based)')
    parser.add_argument('--data', type=str, default='amazon', choices=['amazon', 'yelp'],
                        help='Dataset name')
    parser.add_argument('--stats-path', type=str, default=None,
                        help='Path to precomputed statistics (default: llm_embeddings/{data}/node_statistics.npz)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: llm_embeddings/{data})')
    parser.add_argument('--checkpoint-interval', type=int, default=500,
                        help='Checkpoint interval for LLM mode')
    args = parser.parse_args()

    if args.stats_path is None:
        args.stats_path = f'llm_embeddings/{args.data}/node_statistics.npz'
    if args.output_dir is None:
        args.output_dir = f'llm_embeddings/{args.data}'

    if not os.path.exists(args.stats_path):
        logger.error(f'Statistics file not found: {args.stats_path}')
        logger.error(f'Run: python -m llm.compute_node_statistics --data {args.data} first')
        sys.exit(1)

    stats = load_statistics(args.stats_path)
    num_nodes = stats['degrees'].shape[0]
    logger.info(f'Loaded statistics for {num_nodes} nodes ({args.data})')

    output_path = os.path.join(args.output_dir, 'node_descriptions.json')

    if args.mode == 'template':
        descriptions = generate_descriptions_template(stats, num_nodes, data=args.data)
    else:
        descriptions = generate_descriptions_llm(
            stats, num_nodes, output_path, args.checkpoint_interval, data=args.data)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(descriptions, f)
    logger.info(f'Saved {len(descriptions)} descriptions to {output_path}')

    # sanity check
    sample_keys = list(descriptions.keys())[:3]
    for key in sample_keys:
        logger.info(f'Sample (node {key}): {descriptions[key][:150]}...')


if __name__ == '__main__':
    main()
