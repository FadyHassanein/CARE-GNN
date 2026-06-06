"""Provenance + checksum helpers for cached LLM score artifacts.

Every cached score tensor (e.g. ``text_risk_scores.pt``) gets a sidecar
``<path>.meta.json`` recording *how it was produced* — mode, model, a SHA-256 of
the tensor, timestamp and source. This exists because the two scoring modes
(deterministic template vs. Claude API) previously wrote to the *same* filename
with no record of which one, so a later ``--mode llm`` run silently replaced the
template scores that a result was reported against (and reproducing the
documented command loaded the wrong file).

With a sidecar, training logs exactly which artifact backed a run, and the
generator can refuse to silently overwrite a differently-sourced cache.
"""

import hashlib
import json
import os
from datetime import datetime, timezone


def sha256_tensor(t):
    """Stable SHA-256 over a tensor's raw bytes (CPU, contiguous, float32)."""
    arr = t.detach().to('cpu').contiguous().float().numpy()
    return hashlib.sha256(arr.tobytes()).hexdigest()


def mode_tag(mode, model=None):
    """Short filename tag for a (mode, model) pair.

    ``template`` -> ``'template'``; ``llm`` -> the model family
    (``haiku``/``sonnet``/``opus``) when recognisable, else a sanitized id.
    """
    if mode == 'template':
        return 'template'
    if model:
        low = model.lower()
        for fam in ('haiku', 'sonnet', 'opus'):
            if fam in low:
                return fam
        return ''.join(c if c.isalnum() else '-' for c in low)[:24].strip('-') or 'llm'
    return 'llm'


def variant_path(canonical_path, tag):
    """``.../text_risk_scores.pt`` + ``'template'`` ->
    ``.../text_risk_scores_template.pt``."""
    root, ext = os.path.splitext(canonical_path)
    return f'{root}_{tag}{ext}'


def meta_path(pt_path):
    return pt_path + '.meta.json'


def write_meta(pt_path, meta):
    with open(meta_path(pt_path), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, sort_keys=True)


def read_meta(pt_path):
    mp = meta_path(pt_path)
    if not os.path.exists(mp):
        return None
    try:
        with open(mp, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def utc_now_iso():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def build_meta(scores_tensor, *, mode, model, dataset, score_names,
               source_texts=None, extra=None, created_at=None):
    """Assemble a provenance dict for a score tensor."""
    meta = {
        'mode': mode,
        'model': model,
        'tag': mode_tag(mode, model),
        'dataset': dataset,
        'n_nodes': int(scores_tensor.shape[0]),
        'dims': int(scores_tensor.shape[1]) if scores_tensor.ndim > 1 else 1,
        'score_names': list(score_names),
        'sha256': sha256_tensor(scores_tensor),
        'source_texts': source_texts,
        'created_at': created_at or utc_now_iso(),
        'generator': 'llm/generate_text_risk_scores.py',
    }
    if extra:
        meta.update(extra)
    return meta


def describe(pt_path):
    """One-line provenance string for logging."""
    meta = read_meta(pt_path)
    if not meta:
        return 'provenance UNKNOWN (no .meta.json sidecar)'
    s = (f"mode={meta.get('mode')} model={meta.get('model')} "
         f"sha256={str(meta.get('sha256'))[:12]} created={meta.get('created_at')}")
    if 'llm_fallback_nodes' in meta:
        s += f" llm_fallback_nodes={meta['llm_fallback_nodes']}"
    return s
