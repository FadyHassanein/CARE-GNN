"""
LLM Prior Generation for CAPN.

Generates domain-specific relation priors using an LLM (offline, one-time).
The priors encode per-relation importance and camouflage susceptibility scores
that warm-start the CAPN policy network.

Usage:
    python llm_priors.py --data yelp --output data/llm_priors/yelp_priors.json

If no LLM API is available, use --use-defaults to generate hand-crafted priors
based on domain knowledge from the CARE-GNN paper.
"""

import argparse
import json
import logging
import os

logger = logging.getLogger(__name__)

# Domain knowledge from CARE-GNN paper (Table 1 in paper, README)
# These are hand-crafted priors based on the reported feature/label similarity statistics
YELP_DEFAULTS = {
    "dataset": "yelp",
    "relations": [
        {
            "name": "R-U-R",
            "description": "Reviews posted by the same user",
            "importance": 0.85,
            "camouflage_risk": 0.55,
            "notes": "High feature similarity (0.991) but moderate label similarity (0.909). "
                     "Fraudsters sharing user accounts create camouflage through feature mimicry."
        },
        {
            "name": "R-T-R",
            "description": "Reviews sharing the same product rating (stars)",
            "importance": 0.60,
            "camouflage_risk": 0.35,
            "notes": "Moderate relation — star ratings are easy to manipulate but provide "
                     "weak structural signal for fraud detection."
        },
        {
            "name": "R-S-R",
            "description": "Reviews with the same text sentiment polarity score",
            "importance": 0.70,
            "camouflage_risk": 0.45,
            "notes": "Sentiment is harder to fake consistently. Provides moderate signal "
                     "for detecting coordinated fraud campaigns."
        }
    ]
}

AMAZON_DEFAULTS = {
    "dataset": "amazon",
    "relations": [
        {
            "name": "U-P-U",
            "description": "Users reviewing at least one same product",
            "importance": 0.90,
            "camouflage_risk": 0.70,
            "notes": "Low label similarity (0.167) despite feature similarity (0.711). "
                     "Strong camouflage effect — fraudsters target same products as legitimate users."
        },
        {
            "name": "U-S-U",
            "description": "Users having at least one same star rating within a week",
            "importance": 0.55,
            "camouflage_risk": 0.30,
            "notes": "Weak relation — same star ratings are common and provide "
                     "limited discriminative signal for fraud."
        },
        {
            "name": "U-V-U",
            "description": "Users with top-5% mutual review text TF-IDF similarity",
            "importance": 0.80,
            "camouflage_risk": 0.50,
            "notes": "Text similarity captures copy-paste fraud patterns. "
                     "Moderate camouflage risk as sophisticated fraudsters vary their text."
        }
    ]
}

# LLM prompt template for generating priors
PROMPT_TEMPLATE = """You are an expert in graph-based fraud detection. I need your analysis of relation types
in a fraud detection graph for the {dataset} dataset.

For each relation below, provide two scores on a scale of 0.0 to 1.0:
1. **importance**: How useful is this relation for detecting fraudulent nodes?
   (1.0 = very useful, 0.0 = not useful)
2. **camouflage_risk**: How susceptible is this relation to camouflage attacks where
   fraudsters mimic legitimate behavior? (1.0 = very susceptible, 0.0 = not susceptible)

Relations:
{relations_description}

Respond in JSON format:
{{
    "relations": [
        {{"name": "...", "importance": 0.X, "camouflage_risk": 0.X}},
        ...
    ]
}}

Consider that:
- Camouflaged fraudsters deliberately make their features similar to legitimate users
- Relations with high feature similarity but low label similarity indicate camouflage
- Some relations are easier to manipulate (e.g., star ratings) than others (e.g., text similarity)
"""


def get_default_priors(dataset):
    """Get hand-crafted default priors based on paper domain knowledge."""
    if dataset == 'yelp':
        return YELP_DEFAULTS
    elif dataset == 'amazon':
        return AMAZON_DEFAULTS
    else:
        raise ValueError(f'No default priors for dataset: {dataset}')


def generate_prompt(dataset):
    """Generate the LLM prompt for a given dataset."""
    defaults = get_default_priors(dataset)
    relations_desc = "\n".join([
        f"- {r['name']}: {r['description']}" for r in defaults['relations']
    ])
    return PROMPT_TEMPLATE.format(dataset=dataset, relations_description=relations_desc)


def save_priors(priors, output_path):
    """Save priors to JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(priors, f, indent=2)
    logger.info(f'Priors saved to {output_path}')


def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='Generate LLM priors for CAPN')
    parser.add_argument('--data', type=str, required=True, choices=['yelp', 'amazon'],
                        help='Dataset name')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file path')
    parser.add_argument('--use-defaults', action='store_true', default=True,
                        help='Use hand-crafted defaults instead of LLM API')
    parser.add_argument('--show-prompt', action='store_true', default=False,
                        help='Print the LLM prompt (for manual use with ChatGPT/Claude)')
    args = parser.parse_args()

    if args.output is None:
        args.output = f'data/llm_priors/{args.data}_priors.json'

    if args.show_prompt:
        print("=" * 60)
        print("Copy this prompt to ChatGPT/Claude/etc:")
        print("=" * 60)
        print(generate_prompt(args.data))
        print("=" * 60)
        return

    if args.use_defaults:
        priors = get_default_priors(args.data)
        logger.info(f'Using hand-crafted default priors for {args.data}')
    else:
        logger.info('LLM API integration not implemented. Use --use-defaults or --show-prompt')
        logger.info('To use with an LLM, run with --show-prompt, copy the prompt, and save the response.')
        priors = get_default_priors(args.data)

    save_priors(priors, args.output)

    # print summary
    for r in priors['relations']:
        logger.info(f"  {r['name']}: importance={r['importance']}, camouflage_risk={r['camouflage_risk']}")


if __name__ == '__main__':
    main()
