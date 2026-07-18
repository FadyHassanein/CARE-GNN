"""DECISIVE head-to-head: is the LLM needed, or does a zero-cost heuristic match it?

Reviewer's Path A. For each backbone, at the paper's exact frozen 25/15/60
protocol (5 seeds where stochastic), compare:
  raw32              (reference)
  raw32 + template6  (deterministic heuristic scores)
  raw32 + haiku6     (Claude Haiku scores — the paper's canonical tensor)
and paired-test the haiku-minus-template delta. If that delta is n.s., the LLM
is statistically indistinguishable from a free heuristic and the paper's
"LLM-derived semantic signal" framing cannot stand as written.

Output: results/experiments/llm_vs_template.json
"""

import json
import sys

import numpy as np
import torch
from scipy import stats
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')
from baseline_table import bootstrap_delta_ci, fit_score  # noqa: E402
from probe_yelp_headroom import frozen_split  # noqa: E402
from utils import load_data, normalize  # noqa: E402

SEEDS = [72, 73, 74, 75, 76]


def xgb_score(seed, X, y, idx_tr, idx_te):
    clf = XGBClassifier(n_estimators=300, learning_rate=0.08, max_leaf_nodes=31,
                        scale_pos_weight=float((y[idx_tr] == 0).sum() / (y[idx_tr] == 1).sum()),
                        random_state=seed, n_jobs=-1)
    clf.fit(X[idx_tr], y[idx_tr])
    p = clf.predict_proba(X[idx_te])[:, 1]
    return p, roc_auc_score(y[idx_te], p), 0.0


def main():
    _, feat_data, labels = load_data('yelp')
    feat = normalize(feat_data)
    feat = np.asarray(feat.todense()) if hasattr(feat, 'todense') else np.asarray(feat)
    y = np.asarray(labels).ravel()
    tmpl = torch.load('llm_embeddings/yelp/text_risk_scores_template.pt', weights_only=True).numpy()
    haiku = torch.load('llm_embeddings/yelp/text_risk_scores_haiku.pt', weights_only=True).numpy()
    idx_tr, _, idx_te, _, _, _ = frozen_split(len(y), y)

    # how different are the two score tensors at all?
    corr = [float(np.corrcoef(tmpl[:, i], haiku[:, i])[0, 1]) for i in range(6)]
    print(f'per-column template-vs-haiku Pearson r: {[round(c, 2) for c in corr]}')
    print(f'mean |template - haiku| = {np.abs(tmpl - haiku).mean():.3f} on [0,1]\n')

    feats = {'raw': feat,
             'raw+template6': np.hstack([feat, tmpl]),
             'raw+haiku6': np.hstack([feat, haiku])}
    scorer = {'LR': fit_score, 'MLP': fit_score, 'GBM': fit_score, 'XGB': xgb_score}
    out = {}
    for kind in ['LR', 'MLP', 'GBM', 'XGB']:
        seeds = [2] if kind == 'LR' else SEEDS
        rows, preds = {}, {}
        for fname, X in feats.items():
            if kind in ('LR', 'MLP', 'GBM'):
                runs = [fit_score(kind, s, X, y, idx_tr, idx_te) for s in seeds]
            else:
                runs = [xgb_score(s, X, y, idx_tr, idx_te) for s in seeds]
            rows[fname] = [r[1] for r in runs]
            preds[fname] = runs[0][0]
        d = np.mean(rows['raw+haiku6']) - np.mean(rows['raw+template6'])
        if kind == 'LR':
            lo, hi = bootstrap_delta_ci(y[idx_te], preds['raw+template6'], preds['raw+haiku6'])
            sig = {'method': 'bootstrap 95% CI', 'ci': [lo, hi], 'sig': bool(lo > 0 or hi < 0)}
        else:
            t, p = stats.ttest_rel(rows['raw+haiku6'], rows['raw+template6'])
            sig = {'method': 'paired t (5 seeds)', 't': float(t), 'p': float(p),
                   'sig': bool(p < 0.05)}
        out[kind] = {
            'raw': float(np.mean(rows['raw'])),
            'template6': float(np.mean(rows['raw+template6'])),
            'haiku6': float(np.mean(rows['raw+haiku6'])),
            'haiku_minus_template_pp': float(d * 100),
            'significance': sig,
        }
        print(f'{kind:4s} raw={out[kind]["raw"]:.4f}  +template6={out[kind]["template6"]:.4f}  '
              f'+haiku6={out[kind]["haiku6"]:.4f}  haiku-tmpl={d * 100:+.2f}pp  {sig}', flush=True)

    with open('results/experiments/llm_vs_template.json', 'w', encoding='utf-8') as f:
        json.dump({'protocol': 'frozen YelpChi 25/15/60, paper protocol; haiku vs '
                               'template text6 head-to-head', 'per_column_corr': corr,
                   'mean_abs_diff': float(np.abs(tmpl - haiku).mean()),
                   'models': out}, f, indent=2)
    print('-> results/experiments/llm_vs_template.json')


if __name__ == '__main__':
    main()
