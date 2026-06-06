"""Pin text-risk artifacts with provenance, build the template variant, and
report template-vs-LLM divergence. One-time backfill + reusable audit tool.

What it does (for a dataset, default yelp):
  1. Detects the source of the existing canonical ``text_risk_scores.pt``
     (LLM vs template) from its value quantization + checkpoint presence,
     writes a ``.meta.json`` provenance sidecar, and pins a named copy
     ``text_risk_scores_<tag>.pt``.
  2. Generates the deterministic ``text_risk_scores_template.pt`` variant
     (no API calls), without touching the canonical.
  3. Reports per-score correlation between the two variants and each variant's
     correlation with the fraud label — evidence for how different "template"
     and "Haiku" scores actually are (Blocker B3).

Usage:  python scripts/pin_text_risk_artifacts.py --data yelp
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm.artifacts import (build_meta, describe, mode_tag, read_meta,  # noqa: E402
                           variant_path, write_meta)
from llm.generate_text_risk_scores import (SCORE_NAMES,  # noqa: E402
                                           generate_text_risk_scores)


def quantization_fraction(arr, step=0.05, tol=1e-4):
    """Fraction of values that sit (within tol) on a multiple of `step`.
    Claude outputs coarse scores (~all multiples of 0.05); template scores
    are continuous, so this cleanly separates the two sources."""
    r = arr / step
    return float(np.mean(np.abs(r - np.round(r)) < tol))


def infer_source(arr, output_dir):
    qf = quantization_fraction(arr)
    ckpt = os.path.join(output_dir, 'text_risk_scores.checkpoint')
    if qf > 0.9 or os.path.exists(ckpt):
        return 'llm', 'claude-haiku-4-5-20251001', qf
    return 'template', 'deterministic-template-v1', qf


def _pearson(x, y):
    if x.std() < 1e-9 or y.std() < 1e-9:
        return float('nan')
    return float(np.corrcoef(x, y)[0, 1])


def pin_existing(canonical, out_dir, data):
    cur = torch.load(canonical, weights_only=True)
    arr = cur.float().numpy()
    if read_meta(canonical):
        print(f'Canonical already pinned: {describe(canonical)}')
        return
    mode, model_id, qf = infer_source(arr, out_dir)
    tag = mode_tag(mode, model_id)
    print(f'Inferred canonical source: mode={mode} model={model_id} tag={tag} '
          f'(quantization-to-0.05 fraction={qf:.4f})')
    meta = build_meta(
        cur, mode=mode, model=model_id, dataset=data, score_names=SCORE_NAMES,
        source_texts=f'data/{data}_review_texts.json',
        extra={'pinned_by': 'scripts/pin_text_risk_artifacts.py',
               'quantization_0p05_fraction': round(qf, 6),
               'note': 'provenance inferred during backfill (no original sidecar existed)'})
    cmeta = dict(meta)
    cmeta['canonical_of'] = tag
    write_meta(canonical, cmeta)
    pinned = variant_path(canonical, tag)
    torch.save(cur, pinned)
    write_meta(pinned, meta)
    print(f'Pinned existing canonical -> {pinned}  [{describe(pinned)}]')
    print(f'Labelled canonical        -> {canonical}  [{describe(canonical)}]')


def build_template_variant(canonical, data):
    tpl_path = variant_path(canonical, 'template')
    if os.path.exists(tpl_path) and read_meta(tpl_path):
        print(f'Template variant already present: {describe(tpl_path)}')
        return
    print('Generating deterministic template variant (no API, no canonical overwrite)...')
    generate_text_risk_scores(data, mode='template', write_canonical=False)


def report(canonical, data):
    haiku_path = next((variant_path(canonical, t) for t in ('haiku', 'sonnet', 'opus', 'llm')
                       if os.path.exists(variant_path(canonical, t))), None)
    tpl_path = variant_path(canonical, 'template')
    if not (haiku_path and os.path.exists(tpl_path)):
        print('Could not locate both variants for comparison; skipping report.')
        return
    hk = torch.load(haiku_path, weights_only=True).float().numpy()
    tp = torch.load(tpl_path, weights_only=True).float().numpy()

    print(f'\n=== template ({os.path.basename(tpl_path)}) vs LLM '
          f'({os.path.basename(haiku_path)}) divergence ===')
    for i, name in enumerate(SCORE_NAMES):
        print(f'  {name:20s} pearson r={_pearson(tp[:, i], hk[:, i]):+.3f}  '
              f'mean|delta|={np.abs(tp[:, i] - hk[:, i]).mean():.3f}')
    print(f'  overall mean|delta| = {np.abs(tp - hk).mean():.4f}')

    labels_path = f'data/{data}_labels.npy'
    if os.path.exists(labels_path):
        y = np.load(labels_path).astype(np.float64)
        if y.shape[0] == hk.shape[0]:
            print('\n=== correlation with fraud label (Pearson r) ===')
            print(f'  {"score":20s} {"template":>10s} {"LLM":>10s}')
            for i, name in enumerate(SCORE_NAMES):
                print(f'  {name:20s} {_pearson(tp[:, i], y):>+10.3f} {_pearson(hk[:, i], y):>+10.3f}')
        else:
            print(f'\nlabel shape {y.shape} != scores {hk.shape}; skipping label correlation')
    else:
        print(f'\nno labels at {labels_path}; skipping label correlation')


def main():
    ap = argparse.ArgumentParser(description='Pin text-risk artifacts + report divergence')
    ap.add_argument('--data', default='yelp')
    args = ap.parse_args()
    out_dir = f'llm_embeddings/{args.data}'
    canonical = os.path.join(out_dir, 'text_risk_scores.pt')
    if not os.path.exists(canonical):
        print(f'No canonical artifact at {canonical}; nothing to pin.')
        return
    pin_existing(canonical, out_dir, args.data)
    build_template_variant(canonical, args.data)
    report(canonical, args.data)
    print('\nDone.')


if __name__ == '__main__':
    main()
