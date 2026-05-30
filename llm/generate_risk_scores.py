"""
Generate per-node fraud risk scores using Claude API reasoning or template heuristics.

Mode A (template): Deterministic heuristic scoring — no API key needed.
Mode B (llm): Claude API reasoning with structured JSON output.

Produces llm_risk_scores.pt [N, 6] with scores:
  structural_anomaly, relation_consistency, neighborhood_risk,
  feature_anomaly, coordination_signal, isolation_score

Usage:
    python -m llm.generate_risk_scores --data amazon --mode template
    python -m llm.generate_risk_scores --data amazon --mode llm
"""

import argparse
import json
import logging
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

SCORE_NAMES = [
    'structural_anomaly',
    'relation_consistency',
    'neighborhood_risk',
    'feature_anomaly',
    'coordination_signal',
    'isolation_score',
]

DATASET_INFO = {
    'amazon': {
        'name': 'Amazon product review',
        'relations': [
            ('UPU', 'users sharing the same product'),
            ('USU', 'users sharing the same star rating'),
            ('UVU', 'users in the same top-reviewer category'),
        ],
    },
    'yelp': {
        'name': 'Yelp restaurant review',
        'relations': [
            ('RUR', 'reviews by the same user'),
            ('RTR', 'reviews submitted around the same time'),
            ('RSR', 'reviews with the same star rating'),
        ],
    },
}


def compute_template_scores(node_stats):
    """Compute heuristic risk scores without API calls.

    :param node_stats: dict with per-node statistics arrays
    :return: numpy array [N, 6]
    """
    num_nodes = node_stats['degrees'].shape[0]
    scores = np.zeros((num_nodes, len(SCORE_NAMES)), dtype=np.float32)

    degrees = node_stats['degrees']                      # [N, 3]
    degree_percentiles = node_stats['degree_percentiles']  # [N, 3]
    label_disagreement = node_stats['label_disagreement']  # [N, 3]
    cross_overlap = node_stats['cross_overlap']            # [N, 3]
    degree_ratios = node_stats['degree_ratios']            # [N, 3]
    num_extreme = node_stats['num_extreme_features']       # [N]
    mean_zscore = node_stats['mean_abs_zscore']            # [N]

    has_2hop = 'two_hop_size' in node_stats
    has_ego = 'ego_density' in node_stats
    two_hop = node_stats.get('two_hop_size', np.zeros_like(degrees))
    ego_dens = node_stats.get('ego_density', np.zeros_like(degrees))

    # max values for normalization
    max_2hop = two_hop.max() if has_2hop and two_hop.max() > 0 else 1.0

    for i in range(num_nodes):
        # structural_anomaly: how extreme are the degree percentiles
        percs = degree_percentiles[i]
        anomaly = max(abs(p - 50) / 50 for p in percs)
        scores[i, 0] = min(anomaly, 1.0)

        # relation_consistency: variance of normalized degrees across relations
        deg = degrees[i]
        deg_max = deg.max()
        if deg_max > 0:
            deg_norm = deg / deg_max
            consistency = 1.0 - np.std(deg_norm) / max(np.mean(deg_norm), 1e-8)
            scores[i, 1] = np.clip(consistency, 0.0, 1.0)
        else:
            scores[i, 1] = 0.5

        # neighborhood_risk: mean label disagreement across relations
        scores[i, 2] = np.clip(np.mean(label_disagreement[i]), 0.0, 1.0)

        # feature_anomaly: normalized mean |z-score|
        scores[i, 3] = min(mean_zscore[i] / 3.0, 1.0)

        # coordination_signal: based on ego-network density
        if has_ego:
            scores[i, 4] = np.clip(np.mean(ego_dens[i]), 0.0, 1.0)
        else:
            # fallback: low cross-relation overlap + high degree → coordination
            scores[i, 4] = np.clip(1.0 - np.mean(cross_overlap[i]), 0.0, 1.0)

        # isolation_score: inverse of 2-hop reach
        if has_2hop:
            scores[i, 5] = 1.0 - min(np.mean(two_hop[i]) / max_2hop, 1.0)
        else:
            # fallback: low degree → isolated
            scores[i, 5] = 1.0 - min(np.mean(deg) / max(degrees.max(), 1.0), 1.0)

    return scores


def build_reasoning_prompt(node_id, node_stats, dataset='amazon'):
    """Build a Claude API prompt for a single node."""
    info = DATASET_INFO[dataset]
    deg = node_stats['degrees']
    percs = node_stats['degree_percentiles']
    ld = node_stats['label_disagreement']
    fv = node_stats['neighbor_feat_var']
    co = node_stats['cross_overlap']
    dr = node_stats['degree_ratios']
    ne = node_stats['num_extreme_features']
    mz = node_stats['mean_abs_zscore']

    has_2hop = 'two_hop_size' in node_stats
    has_ego = 'ego_density' in node_stats

    lines = [
        f'Analyze node #{node_id} in a {info["name"]} network for fraud detection.',
        'Output ONLY a JSON object with 6 scores (0.0 to 1.0).\n',
        'Structural profile:',
    ]

    for r_idx, (rname, rdesc) in enumerate(info['relations']):
        parts = (f'- {rname} ({rdesc}): degree={int(deg[node_id, r_idx])} '
                 f'({percs[node_id, r_idx]:.0f}th pctl), '
                 f'label_disagreement={ld[node_id, r_idx]:.3f}, '
                 f'neighbor_variance={fv[node_id, r_idx]:.3f}')
        if has_2hop:
            parts += f', 2hop_reach={int(node_stats["two_hop_size"][node_id, r_idx])}'
        if has_ego:
            parts += f', ego_density={node_stats["ego_density"][node_id, r_idx]:.3f}'
        lines.append(parts)

    pair_names = ['R1-R2', 'R1-R3', 'R2-R3']
    lines.append(f'- Cross-relation overlap: '
                 + ', '.join(f'{pn}={co[node_id, pi]:.3f}' for pi, pn in enumerate(pair_names)))
    lines.append(f'- Degree ratios: '
                 + ', '.join(f'{pn}={dr[node_id, pi]:.2f}' for pi, pn in enumerate(pair_names)))
    lines.append(f'- Feature anomaly: {int(ne[node_id])} extreme features, '
                 f'mean |z-score|={mz[node_id]:.3f}')

    lines.append('''
Score definitions:
- structural_anomaly: How unusual is connectivity (0=typical, 1=extreme outlier)
- relation_consistency: How uniform is behavior across relations (0=inconsistent, 1=uniform)
- neighborhood_risk: How suspicious are neighbors (0=clean, 1=mostly fraudulent)
- feature_anomaly: How statistically unusual are features (0=normal, 1=extreme)
- coordination_signal: Likelihood of coordinated behavior (0=independent, 1=coordinated)
- isolation_score: Structural isolation (0=embedded, 1=peripheral)

Output ONLY valid JSON:''')

    return '\n'.join(lines)


def build_batch_prompt(node_ids, node_stats, dataset='amazon'):
    """Build a Claude API prompt for a batch of nodes."""
    info = DATASET_INFO[dataset]
    deg = node_stats['degrees']
    percs = node_stats['degree_percentiles']
    ld = node_stats['label_disagreement']
    fv = node_stats['neighbor_feat_var']
    co = node_stats['cross_overlap']
    dr = node_stats['degree_ratios']
    ne = node_stats['num_extreme_features']
    mz = node_stats['mean_abs_zscore']

    has_2hop = 'two_hop_size' in node_stats
    has_ego = 'ego_density' in node_stats

    lines = [
        f'Analyze the following {len(node_ids)} nodes in a {info["name"]} network for fraud detection.',
        'For EACH node, output a JSON object with 6 scores (0.0 to 1.0).',
        'Output a JSON array of objects, one per node, in the same order.\n',
    ]

    for node_id in node_ids:
        lines.append(f'Node #{node_id}:')
        for r_idx, (rname, rdesc) in enumerate(info['relations']):
            parts = (f'  {rname}: deg={int(deg[node_id, r_idx])} '
                     f'({percs[node_id, r_idx]:.0f}th pctl), '
                     f'disagree={ld[node_id, r_idx]:.3f}, var={fv[node_id, r_idx]:.3f}')
            if has_2hop:
                parts += f', 2hop={int(node_stats["two_hop_size"][node_id, r_idx])}'
            if has_ego:
                parts += f', ego={node_stats["ego_density"][node_id, r_idx]:.3f}'
            lines.append(parts)

        pair_names = ['R1-R2', 'R1-R3', 'R2-R3']
        lines.append(f'  Overlap: '
                     + ', '.join(f'{pn}={co[node_id, pi]:.3f}' for pi, pn in enumerate(pair_names)))
        lines.append(f'  Deg ratios: '
                     + ', '.join(f'{pn}={dr[node_id, pi]:.2f}' for pi, pn in enumerate(pair_names)))
        lines.append(f'  Anomaly: {int(ne[node_id])} extreme feats, '
                     f'mean|z|={mz[node_id]:.3f}')
        lines.append('')

    lines.append('''Score definitions:
- structural_anomaly: How unusual is connectivity (0=typical, 1=extreme outlier)
- relation_consistency: How uniform is behavior across relations (0=inconsistent, 1=uniform)
- neighborhood_risk: How suspicious are neighbors (0=clean, 1=mostly fraudulent)
- feature_anomaly: How statistically unusual are features (0=normal, 1=extreme)
- coordination_signal: Likelihood of coordinated behavior (0=independent, 1=coordinated)
- isolation_score: Structural isolation (0=embedded, 1=peripheral)

Output ONLY a valid JSON array of objects (one per node, same order):''')

    return '\n'.join(lines)


def parse_scores_json(text, expected_count=1):
    """Parse Claude response into score arrays. Returns list of dicts."""
    text = text.strip()
    # try parsing as JSON array first
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            return [parsed]
    except json.JSONDecodeError:
        pass

    # try extracting JSON from markdown code blocks
    if '```' in text:
        start = text.find('```')
        end = text.rfind('```')
        if start != end:
            inner = text[start:end].split('\n', 1)[-1]
            try:
                parsed = json.loads(inner)
                if isinstance(parsed, list):
                    return parsed
                if isinstance(parsed, dict):
                    return [parsed]
            except json.JSONDecodeError:
                pass

    logger.warning(f'Failed to parse scores JSON: {text[:200]}')
    return [None] * expected_count


def generate_scores_llm(node_stats, dataset='amazon', output_dir=None,
                        batch_size=5, checkpoint_interval=500):
    """Generate risk scores using Claude API reasoning.

    :param node_stats: dict of numpy arrays from node_statistics.npz
    :param dataset: 'amazon' or 'yelp'
    :param output_dir: directory for output (default: llm_embeddings/{data})
    :param batch_size: number of nodes per API call
    :param checkpoint_interval: save checkpoint every N nodes
    :return: numpy array [N, 6]
    """
    try:
        import anthropic
    except ImportError:
        logger.warning('anthropic package not installed, falling back to template mode')
        return compute_template_scores(node_stats)

    api_key = os.environ.get('ANTHROPIC_API_KEY', '')
    if not api_key:
        logger.warning('ANTHROPIC_API_KEY not set, falling back to template mode')
        return compute_template_scores(node_stats)

    if output_dir is None:
        output_dir = f'llm_embeddings/{dataset}'

    num_nodes = node_stats['degrees'].shape[0]
    scores = np.full((num_nodes, len(SCORE_NAMES)), 0.5, dtype=np.float32)

    # load checkpoint if exists
    checkpoint_path = os.path.join(output_dir, 'risk_scores.checkpoint')
    completed = set()
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            checkpoint = json.load(f)
        for node_id_str, node_scores in checkpoint.items():
            node_id = int(node_id_str)
            if node_id < num_nodes:
                scores[node_id] = [node_scores.get(name, 0.5) for name in SCORE_NAMES]
                completed.add(node_id)
        logger.info(f'Resumed from checkpoint: {len(completed)}/{num_nodes} nodes')
    else:
        checkpoint = {}

    remaining = sorted(set(range(num_nodes)) - completed)
    if not remaining:
        logger.info('All nodes already scored')
        return scores

    logger.info(f'Scoring {len(remaining)} remaining nodes with Claude API (batch_size={batch_size})')

    client = anthropic.Anthropic(api_key=api_key)
    template_scores = compute_template_scores(node_stats)  # fallback

    processed = 0
    for batch_start in range(0, len(remaining), batch_size):
        batch_ids = remaining[batch_start:batch_start + batch_size]

        try:
            if len(batch_ids) == 1:
                prompt = build_reasoning_prompt(batch_ids[0], node_stats, dataset)
            else:
                prompt = build_batch_prompt(batch_ids, node_stats, dataset)

            message = client.messages.create(
                model='claude-haiku-4-5-20251001',
                max_tokens=150 * len(batch_ids),
                temperature=0,
                messages=[{'role': 'user', 'content': prompt}],
            )

            parsed = parse_scores_json(message.content[0].text, expected_count=len(batch_ids))

            for idx, node_id in enumerate(batch_ids):
                if idx < len(parsed) and parsed[idx] is not None:
                    for si, name in enumerate(SCORE_NAMES):
                        val = parsed[idx].get(name, 0.5)
                        scores[node_id, si] = np.clip(float(val), 0.0, 1.0)
                    checkpoint[str(node_id)] = parsed[idx]
                else:
                    # Partial-parse fallback: fill in-memory scores with template
                    # but do NOT checkpoint — this lets a subsequent run retry
                    # these nodes with the LLM instead of marking them "done".
                    scores[node_id] = template_scores[node_id]

        except Exception as e:
            logger.warning(f'API error for batch starting at node {batch_ids[0]}: {e}')
            # Whole-batch API failure (rate limits, quota, network): fill template
            # in-memory only, do NOT checkpoint, so the next run retries these nodes.
            for node_id in batch_ids:
                scores[node_id] = template_scores[node_id]

        processed += len(batch_ids)
        if processed % checkpoint_interval < batch_size:
            os.makedirs(output_dir, exist_ok=True)
            with open(checkpoint_path, 'w') as f:
                json.dump(checkpoint, f)
            logger.info(f'Checkpoint: {len(completed) + processed}/{num_nodes} nodes')

        time.sleep(0.05)

    # final checkpoint save
    os.makedirs(output_dir, exist_ok=True)
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint, f)

    return scores


def generate_risk_scores(data='amazon', mode='template', stats_dir=None, output_dir=None):
    """Main entry point for generating risk scores.

    :param data: dataset name
    :param mode: 'template' or 'llm'
    :param stats_dir: directory containing node_statistics.npz
    :param output_dir: output directory for risk scores
    """
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    if stats_dir is None:
        stats_dir = f'llm_embeddings/{data}'
    if output_dir is None:
        output_dir = stats_dir

    stats_path = os.path.join(stats_dir, 'node_statistics.npz')
    if not os.path.exists(stats_path):
        raise FileNotFoundError(
            f'Node statistics not found at {stats_path}. '
            f'Run: python -m llm.compute_node_statistics --data {data}')

    logger.info(f'Loading node statistics from {stats_path}')
    stats = dict(np.load(stats_path))
    num_nodes = stats['degrees'].shape[0]
    logger.info(f'Dataset: {data}, {num_nodes} nodes')

    if mode == 'template':
        logger.info('Generating risk scores using template heuristics...')
        scores = compute_template_scores(stats)
    elif mode == 'llm':
        logger.info('Generating risk scores using Claude API reasoning...')
        scores = generate_scores_llm(stats, dataset=data, output_dir=output_dir)
    else:
        raise ValueError(f'Unknown mode: {mode}. Must be "template" or "llm"')

    # save as tensor
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'llm_risk_scores.pt')
    torch.save(torch.tensor(scores, dtype=torch.float32), output_path)
    logger.info(f'Saved risk scores to {output_path} (shape: [{num_nodes}, {len(SCORE_NAMES)}])')

    # sanity check
    logger.info('=== Risk Score Statistics ===')
    for si, name in enumerate(SCORE_NAMES):
        col = scores[:, si]
        logger.info(f'{name}: mean={col.mean():.4f}, std={col.std():.4f}, '
                    f'min={col.min():.4f}, max={col.max():.4f}')

    return output_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate per-node fraud risk scores')
    parser.add_argument('--data', type=str, default='amazon', choices=['amazon', 'yelp'],
                        help='Dataset name')
    parser.add_argument('--mode', type=str, default='template', choices=['template', 'llm'],
                        help='Generation mode: template (heuristic) or llm (Claude API)')
    parser.add_argument('--stats-dir', type=str, default=None,
                        help='Directory containing node_statistics.npz')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: same as stats-dir)')
    parser.add_argument('--batch-size', type=int, default=5,
                        help='Nodes per API call (llm mode only)')
    args = parser.parse_args()
    generate_risk_scores(args.data, args.mode, args.stats_dir, args.output_dir)
