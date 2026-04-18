"""
Generate per-node fraud risk scores by analysing raw review TEXT (not graph statistics).

This is different from `generate_risk_scores.py` which analyses graph statistics.
Here we use the actual review text content to detect textual fraud signals that
behavioural/structural features cannot capture (copy-paste language, promotional tone,
generic templates, etc.).

Mode A (template): Deterministic heuristics computed from text properties.
Mode B (llm):      Claude Haiku API analysing the review content.

Produces text_risk_scores.pt [N, 6] with scores in [0, 1]:
  generic_language, sentiment_mismatch, promotional_tone,
  copy_paste_signal, detail_authenticity, behavioral_anomaly

Usage:
    python -m llm.generate_text_risk_scores --data yelp --mode template
    python -m llm.generate_text_risk_scores --data yelp --mode llm
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

SCORE_NAMES = [
    'generic_language',       # 0=specific/detailed, 1=vague/generic
    'sentiment_mismatch',     # 0=consistent, 1=contradictory
    'promotional_tone',       # 0=organic, 1=promotional/advertorial
    'copy_paste_signal',      # 0=unique, 1=formulaic
    'detail_authenticity',    # 0=likely fake, 1=clearly authentic (inverted)
    'behavioral_anomaly',     # 0=normal, 1=anomalous (caps/emoji/repetition)
]

# word lists for heuristic scoring
POSITIVE_WORDS = {
    'great', 'amazing', 'awesome', 'excellent', 'perfect', 'best', 'love',
    'wonderful', 'fantastic', 'incredible', 'outstanding', 'superb', 'brilliant',
    'delicious', 'beautiful', 'fabulous', 'extraordinary', 'marvelous', 'good',
    'nice', 'lovely', 'yummy', 'tasty', 'favorite',
}
NEGATIVE_WORDS = {
    'terrible', 'awful', 'horrible', 'worst', 'bad', 'poor', 'disappointing',
    'disgusting', 'nasty', 'rude', 'slow', 'cold', 'stale', 'bland', 'dirty',
    'overpriced', 'mediocre', 'avoid', 'never', 'hate', 'waste',
}
PROMO_PHRASES = {
    'highly recommend', 'must try', 'must visit', 'definitely recommend',
    'you have to', 'one of the best', 'the best', 'five stars', '5 stars',
    'worth it', 'worth every', 'exceeded expectations', 'top notch',
    'top-notch', 'a plus', 'a+',
}
GENERIC_PHRASES = {
    'great place', 'nice place', 'good food', 'great food', 'nice food',
    'great service', 'nice service', 'good service', 'friendly staff',
    'good experience', 'nice experience', 'will come back', 'come back again',
    'will return', 'good value', 'great value',
}


def _tokenize(text):
    """Cheap tokenizer - lowercase words only."""
    return re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?", text.lower())


def _compute_one(text):
    """Return six heuristic scores for a single review text."""
    if not text or len(text.strip()) == 0:
        return np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)

    text = text.strip()
    text_lower = text.lower()
    tokens = _tokenize(text)
    n_tokens = max(len(tokens), 1)
    unique_ratio = len(set(tokens)) / n_tokens
    char_count = max(len(text), 1)

    # --- 1. generic_language --------------------------------------------------
    # short reviews and heavy use of generic phrases score high
    length_penalty = 1.0 - min(n_tokens / 150.0, 1.0)     # <=150 words -> 0, longer -> 1
    generic_hits = sum(1 for p in GENERIC_PHRASES if p in text_lower)
    generic_score = np.clip(0.4 * length_penalty + 0.6 * min(generic_hits / 3.0, 1.0), 0.0, 1.0)

    # --- 2. sentiment_mismatch ------------------------------------------------
    # high mismatch when positive and negative words co-occur in near-equal proportion
    pos = sum(1 for t in tokens if t in POSITIVE_WORDS)
    neg = sum(1 for t in tokens if t in NEGATIVE_WORDS)
    total_sent = pos + neg
    if total_sent > 0:
        balance = min(pos, neg) / total_sent               # 0 if purely one-sided, 0.5 if balanced
        mismatch_score = np.clip(balance * 2.0, 0.0, 1.0)
    else:
        mismatch_score = 0.5                                # unknown -> neutral

    # --- 3. promotional_tone --------------------------------------------------
    promo_hits = sum(1 for p in PROMO_PHRASES if p in text_lower)
    # strong positives without balancing detail
    pos_density = pos / n_tokens
    promo_score = np.clip(0.5 * min(promo_hits / 2.0, 1.0) + 0.5 * min(pos_density * 10.0, 1.0), 0.0, 1.0)

    # --- 4. copy_paste_signal -------------------------------------------------
    # low unique_ratio -> repeated words; most_common token frequency also indicates template
    if n_tokens >= 5:
        most_common_freq = Counter(tokens).most_common(1)[0][1] / n_tokens
        copy_score = np.clip(0.6 * (1.0 - unique_ratio) + 0.4 * most_common_freq * 3.0, 0.0, 1.0)
    else:
        copy_score = 0.5

    # --- 5. detail_authenticity (higher = more authentic) ---------------------
    # specific details: digits, proper nouns (capitalised non-initial tokens), long reviews
    digit_count = sum(1 for c in text if c.isdigit())
    # count capitalised words that aren't the first word of a sentence
    sentences = re.split(r'[.!?]+', text)
    proper_noun_count = 0
    for sent in sentences:
        words = sent.strip().split()
        if len(words) <= 1:
            continue
        for w in words[1:]:
            if w and w[0].isupper() and not w.isupper():
                proper_noun_count += 1

    digit_density = min(digit_count / char_count * 100, 1.0)
    proper_noun_density = min(proper_noun_count / n_tokens * 5, 1.0)
    length_bonus = min(n_tokens / 100.0, 1.0)
    authenticity = np.clip(0.3 * digit_density + 0.4 * proper_noun_density + 0.3 * length_bonus,
                           0.0, 1.0)

    # --- 6. behavioral_anomaly -------------------------------------------------
    # excessive caps, exclamations, emoji-like patterns, repeated characters
    caps_ratio = sum(1 for c in text if c.isupper()) / char_count
    exclaim_ratio = text.count('!') / char_count
    repeat_char = len(re.findall(r'(.)\1{3,}', text))       # "sooooo"
    emoji_like = len(re.findall(r'[:;=][\-\^]?[\)\(DPp]|<3', text))
    anomaly_score = np.clip(
        0.35 * min(caps_ratio * 5, 1.0) +
        0.25 * min(exclaim_ratio * 50, 1.0) +
        0.2 * min(repeat_char / 3.0, 1.0) +
        0.2 * min(emoji_like / 3.0, 1.0),
        0.0, 1.0,
    )

    return np.array([generic_score, mismatch_score, promo_score,
                     copy_score, authenticity, anomaly_score], dtype=np.float32)


def compute_template_scores(texts):
    """Compute heuristic scores for all reviews without any API call.

    :param texts: list of strings, length N
    :return: numpy array [N, 6]
    """
    num_nodes = len(texts)
    scores = np.zeros((num_nodes, len(SCORE_NAMES)), dtype=np.float32)
    for i, text in enumerate(texts):
        scores[i] = _compute_one(text)
        if (i + 1) % 10000 == 0:
            logger.info(f'  template scoring: {i+1}/{num_nodes}')
    return scores


def build_batch_prompt(node_ids, texts, max_chars=800):
    """Build Claude API prompt for a batch of reviews.

    Truncates texts to max_chars to keep prompts bounded.
    """
    lines = [
        f'You are a fraud detection expert. Analyse each of the following {len(node_ids)} '
        'Yelp reviews and output a JSON array (one object per review, in the same order).',
        '',
        'For each review, output a JSON object with 6 scores in [0.0, 1.0]:',
        '- generic_language:     0=specific/detailed, 1=vague/template-like',
        '- sentiment_mismatch:   0=tone and wording agree, 1=contradictory signals',
        '- promotional_tone:     0=organic, 1=advertorial/marketing-style',
        '- copy_paste_signal:    0=unique wording, 1=formulaic/automated feel',
        '- detail_authenticity:  0=fake-sounding, 1=concrete and believable (inverted)',
        '- behavioral_anomaly:   0=normal prose, 1=unusual (all caps, emoji spam, etc.)',
        '',
        'Return ONLY the JSON array, nothing else.',
        '',
    ]
    for node_id, text in zip(node_ids, texts):
        snippet = text.strip().replace('\n', ' ')
        if len(snippet) > max_chars:
            snippet = snippet[:max_chars] + '...'
        lines.append(f'Review #{node_id}: "{snippet}"')
        lines.append('')
    lines.append('JSON array output:')
    return '\n'.join(lines)


def parse_scores_json(text, expected_count=1):
    """Parse Claude response into list of score dicts.

    Same pattern as generate_risk_scores.parse_scores_json.
    """
    text = text.strip()
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
        match = re.search(r'```(?:json)?\s*(.+?)```', text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(1))
                if isinstance(parsed, list):
                    return parsed
                if isinstance(parsed, dict):
                    return [parsed]
            except json.JSONDecodeError:
                pass

    # try finding a JSON array in the text
    array_match = re.search(r'\[.*\]', text, re.DOTALL)
    if array_match:
        try:
            parsed = json.loads(array_match.group(0))
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass

    logger.warning(f'Failed to parse scores JSON: {text[:200]}')
    return [None] * expected_count


def generate_scores_llm(texts, dataset='yelp', output_dir=None,
                        batch_size=10, checkpoint_interval=500,
                        model='claude-haiku-4-5-20251001'):
    """Generate text risk scores using Claude API.

    :param texts: list of N review texts
    :param dataset: dataset name (used for output path defaults)
    :param output_dir: directory for output and checkpoint
    :param batch_size: reviews per API call (default 10)
    :param checkpoint_interval: save checkpoint every N nodes processed
    :param model: Anthropic model id
    :return: numpy array [N, 6]
    """
    try:
        import anthropic
    except ImportError:
        logger.warning('anthropic package not installed, falling back to template mode')
        return compute_template_scores(texts)

    api_key = os.environ.get('ANTHROPIC_API_KEY', '')
    if not api_key:
        logger.warning('ANTHROPIC_API_KEY not set, falling back to template mode')
        return compute_template_scores(texts)

    if output_dir is None:
        output_dir = f'llm_embeddings/{dataset}'
    os.makedirs(output_dir, exist_ok=True)

    num_nodes = len(texts)
    scores = np.full((num_nodes, len(SCORE_NAMES)), 0.5, dtype=np.float32)

    # checkpointing
    checkpoint_path = os.path.join(output_dir, 'text_risk_scores.checkpoint')
    completed = set()
    checkpoint = {}
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            checkpoint = json.load(f)
        for node_id_str, node_scores in checkpoint.items():
            node_id = int(node_id_str)
            if node_id < num_nodes:
                scores[node_id] = [node_scores.get(name, 0.5) for name in SCORE_NAMES]
                completed.add(node_id)
        logger.info(f'Resumed from checkpoint: {len(completed)}/{num_nodes} nodes')

    remaining = sorted(set(range(num_nodes)) - completed)
    if not remaining:
        logger.info('All nodes already scored')
        return scores

    logger.info(f'Scoring {len(remaining)} remaining nodes with Claude API '
                f'(batch_size={batch_size}, model={model})')

    # template fallback computed lazily per-batch to save memory on Yelp (45k nodes)
    client = anthropic.Anthropic(api_key=api_key)
    processed = 0
    last_ckpt = 0

    for batch_start in range(0, len(remaining), batch_size):
        batch_ids = remaining[batch_start:batch_start + batch_size]
        batch_texts = [texts[i] for i in batch_ids]

        try:
            prompt = build_batch_prompt(batch_ids, batch_texts)
            message = client.messages.create(
                model=model,
                max_tokens=200 * len(batch_ids),
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
                    tpl = _compute_one(batch_texts[idx])
                    scores[node_id] = tpl
                    checkpoint[str(node_id)] = {name: float(tpl[si])
                                                for si, name in enumerate(SCORE_NAMES)}

        except Exception as e:
            logger.warning(f'API error for batch starting at node {batch_ids[0]}: {e}')
            for i, node_id in enumerate(batch_ids):
                tpl = _compute_one(batch_texts[i])
                scores[node_id] = tpl
                checkpoint[str(node_id)] = {name: float(tpl[si])
                                            for si, name in enumerate(SCORE_NAMES)}

        processed += len(batch_ids)
        if processed - last_ckpt >= checkpoint_interval:
            with open(checkpoint_path, 'w') as f:
                json.dump(checkpoint, f)
            logger.info(f'Checkpoint: {len(completed) + processed}/{num_nodes} nodes')
            last_ckpt = processed

        time.sleep(0.05)

    # final save
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint, f)

    return scores


def generate_text_risk_scores(data='yelp', mode='template',
                              texts_path=None, output_dir=None,
                              batch_size=10):
    """Main entry point.

    :param data: dataset name (must have review text file)
    :param mode: 'template' or 'llm'
    :param texts_path: path to json list of review texts (ordered by node id)
    :param output_dir: directory to write text_risk_scores.pt
    :param batch_size: batch size for llm mode
    """
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    if texts_path is None:
        texts_path = f'data/{data}_review_texts.json'
    if output_dir is None:
        output_dir = f'llm_embeddings/{data}'

    if not os.path.exists(texts_path):
        raise FileNotFoundError(
            f'Review texts file not found at {texts_path}. '
            f'For Yelp, run the recovery pipeline described in doc/text_embedding_experiment_results.md'
        )

    logger.info(f'Loading review texts from {texts_path}')
    with open(texts_path, 'r', encoding='utf-8') as f:
        texts = json.load(f)
    num_nodes = len(texts)
    logger.info(f'Loaded {num_nodes} review texts')

    if mode == 'template':
        logger.info('Computing template text risk scores (no API calls)...')
        scores = compute_template_scores(texts)
    elif mode == 'llm':
        logger.info('Generating text risk scores via Claude API...')
        scores = generate_scores_llm(texts, dataset=data, output_dir=output_dir,
                                     batch_size=batch_size)
    else:
        raise ValueError(f'Unknown mode: {mode}')

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'text_risk_scores.pt')
    torch.save(torch.tensor(scores, dtype=torch.float32), output_path)
    logger.info(f'Saved text risk scores to {output_path} '
                f'(shape: [{num_nodes}, {len(SCORE_NAMES)}])')

    logger.info('=== Text Risk Score Statistics ===')
    for si, name in enumerate(SCORE_NAMES):
        col = scores[:, si]
        logger.info(f'{name}: mean={col.mean():.4f}, std={col.std():.4f}, '
                    f'min={col.min():.4f}, max={col.max():.4f}')

    return output_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate text-based per-node fraud risk scores')
    parser.add_argument('--data', type=str, default='yelp', choices=['yelp', 'amazon'],
                        help='Dataset name (must have review texts available)')
    parser.add_argument('--mode', type=str, default='template', choices=['template', 'llm'],
                        help='Scoring mode')
    parser.add_argument('--texts-path', type=str, default=None,
                        help='Path to review texts JSON (default: data/{data}_review_texts.json)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: llm_embeddings/{data})')
    parser.add_argument('--batch-size', type=int, default=10,
                        help='Reviews per Claude API call (llm mode only)')
    args = parser.parse_args()
    generate_text_risk_scores(args.data, args.mode, args.texts_path,
                              args.output_dir, args.batch_size)
