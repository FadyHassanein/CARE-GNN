"""
Deliverable 3: Encode text descriptions into fixed-size vectors using Sentence-BERT.

Uses all-MiniLM-L6-v2 (384-dim) with L2 normalization.

Usage:
    python -m llm.encode_embeddings
"""

import argparse
import json
import logging
import os
import sys

import numpy as np
import torch

logger = logging.getLogger(__name__)


def encode_descriptions(descriptions_path='llm_embeddings/node_descriptions.json',
                        output_dir='llm_embeddings',
                        model_name='all-MiniLM-L6-v2',
                        batch_size=256):
    """Encode text descriptions into sentence embeddings."""
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.error('sentence-transformers not installed. Run: pip install sentence-transformers')
        sys.exit(1)

    # load descriptions
    if not os.path.exists(descriptions_path):
        logger.error(f'Descriptions file not found: {descriptions_path}')
        logger.error('Run: python -m llm.generate_descriptions --mode template first')
        sys.exit(1)

    with open(descriptions_path, 'r') as f:
        descriptions = json.load(f)

    # sort by node index to ensure consistent ordering
    num_nodes = len(descriptions)
    texts = [descriptions[str(i)] for i in range(num_nodes)]
    logger.info(f'Loaded {num_nodes} descriptions')

    # load sentence-transformer model
    logger.info(f'Loading sentence-transformer model: {model_name}')
    model = SentenceTransformer(model_name)

    # encode in batches
    logger.info(f'Encoding {num_nodes} descriptions (batch_size={batch_size})...')
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,  # L2 normalization
    )

    # convert to torch tensor
    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)

    # save
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'llm_semantic_embeddings.pt')
    torch.save(embeddings_tensor, output_path)
    logger.info(f'Saved embeddings to {output_path}')

    # sanity check
    logger.info('=== Sanity Check ===')
    logger.info(f'Shape: {embeddings_tensor.shape}')
    logger.info(f'Dtype: {embeddings_tensor.dtype}')
    norms = torch.norm(embeddings_tensor, dim=1)
    logger.info(f'Mean norm: {norms.mean():.6f}')
    logger.info(f'Min norm: {norms.min():.6f}, Max norm: {norms.max():.6f}')
    logger.info(f'Min value: {embeddings_tensor.min():.6f}, Max value: {embeddings_tensor.max():.6f}')
    logger.info(f'Mean value: {embeddings_tensor.mean():.6f}, Std value: {embeddings_tensor.std():.6f}')

    return output_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Encode descriptions into sentence embeddings')
    parser.add_argument('--data', type=str, default='amazon', choices=['amazon', 'yelp'],
                        help='Dataset name')
    parser.add_argument('--descriptions-path', type=str, default=None,
                        help='Path to node descriptions JSON (default: llm_embeddings/{data}/node_descriptions.json)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: llm_embeddings/{data})')
    parser.add_argument('--model', type=str, default='all-MiniLM-L6-v2',
                        help='Sentence-transformer model name')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Encoding batch size')
    args = parser.parse_args()
    if args.descriptions_path is None:
        args.descriptions_path = f'llm_embeddings/{args.data}/node_descriptions.json'
    if args.output_dir is None:
        args.output_dir = f'llm_embeddings/{args.data}'
    encode_descriptions(args.descriptions_path, args.output_dir, args.model, args.batch_size)
