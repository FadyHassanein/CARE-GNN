"""Tabular baselines +/- LLM text6 on the FROZEN YelpChi split, with significance.

Extends probe_yelp_headroom.py into the paper's baseline table: LR / MLP / GBM
on raw32 vs raw32+text6, 5 seeds where the model is stochastic, paired t-test
on the text6 delta. LR is deterministic -> paired bootstrap CI over test nodes.

Output: results/experiments/tabular_baselines.json + printed table.
"""

import json
import sys

import numpy as np
import torch
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')
from probe_yelp_headroom import frozen_split  # noqa: E402
from utils import load_data, normalize  # noqa: E402

SEEDS = [72, 73, 74, 75, 76]


def make_model(kind, seed):
    if kind == 'LR':
        return LogisticRegression(max_iter=2000, class_weight='balanced'), True
    if kind == 'MLP':
        return MLPClassifier(hidden_layer_sizes=(64, 64), max_iter=300,
                             early_stopping=True, random_state=seed), True
    if kind == 'GBM':
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                              max_leaf_nodes=31, l2_regularization=1.0,
                                              random_state=seed), False
    raise ValueError(kind)


def fit_score(kind, seed, X, y, idx_tr, idx_te):
    model, scale = make_model(kind, seed)
    Xtr, Xte = X[idx_tr], X[idx_te]
    if scale:
        sc = StandardScaler().fit(Xtr)
        Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    model.fit(Xtr, y[idx_tr])
    p = model.predict_proba(Xte)[:, 1]
    return p, roc_auc_score(y[idx_te], p), average_precision_score(y[idx_te], p)


def bootstrap_delta_ci(y, p_base, p_aug, n_boot=1000, seed=2):
    """Paired bootstrap CI over test nodes for AUC(aug)-AUC(base)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        deltas.append(roc_auc_score(y[idx], p_aug[idx]) - roc_auc_score(y[idx], p_base[idx]))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return float(lo), float(hi)


def main():
    _, feat_data, labels = load_data('yelp')
    feat = normalize(feat_data)
    feat = np.asarray(feat.todense()) if hasattr(feat, 'todense') else np.asarray(feat)
    y = np.asarray(labels).ravel()
    text6 = torch.load('llm_embeddings/yelp/text_risk_scores.pt', weights_only=True).numpy()
    idx_tr, _, idx_te, _, _, _ = frozen_split(len(y), y)

    feats = {'raw32': feat, 'raw32+text6': np.hstack([feat, text6])}
    out = {}
    for kind in ['LR', 'MLP', 'GBM']:
        seeds = [2] if kind == 'LR' else SEEDS  # LR deterministic
        rows = {}
        preds = {}
        for fname, X in feats.items():
            runs = [fit_score(kind, s, X, y, idx_tr, idx_te) for s in seeds]
            aucs = [r[1] for r in runs]
            aps = [r[2] for r in runs]
            preds[fname] = runs[0][0]
            rows[fname] = {
                'auc_mean': float(np.mean(aucs)),
                'auc_std': float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0,
                'ap_mean': float(np.mean(aps)),
                'ap_std': float(np.std(aps, ddof=1)) if len(aps) > 1 else 0.0,
                'per_seed_auc': [float(a) for a in aucs],
            }
        delta = rows['raw32+text6']['auc_mean'] - rows['raw32']['auc_mean']
        if kind == 'LR':
            lo, hi = bootstrap_delta_ci(y[idx_te], preds['raw32'], preds['raw32+text6'])
            sig = {'method': 'paired bootstrap 95% CI', 'ci': [lo, hi], 'sig': lo > 0}
        else:
            base = rows['raw32']['per_seed_auc']
            aug = rows['raw32+text6']['per_seed_auc']
            t, p = stats.ttest_rel(aug, base)
            sig = {'method': 'paired t-test (5 seeds)', 't': float(t), 'p': float(p),
                   'sig': bool(p < 0.05 and delta > 0)}
        out[kind] = {'rows': rows, 'text6_delta_pp': delta * 100, 'significance': sig}
        print(f'{kind:4s} raw={rows["raw32"]["auc_mean"]:.4f}  '
              f'+text6={rows["raw32+text6"]["auc_mean"]:.4f}  '
              f'delta={delta * 100:+.2f}pp  {sig}', flush=True)

    with open('results/experiments/tabular_baselines.json', 'w', encoding='utf-8') as f:
        json.dump({'protocol': 'frozen YelpChi 25/15/60 split (random_state=2), '
                               'Haiku text_risk_scores.pt', 'models': out}, f, indent=2)
    print('-> results/experiments/tabular_baselines.json')


if __name__ == '__main__':
    main()
