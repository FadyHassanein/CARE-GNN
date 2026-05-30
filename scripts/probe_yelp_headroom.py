"""Probe the signal ceiling on the FROZEN YelpChi split (no GNN training).

Replicates train.py's exact split (two-stage stratified, random_state=2, 25/15/60)
and measures test-set AUC for linear (LogisticRegression) and nonlinear
(HistGradientBoosting) classifiers on various feature combinations.

Purpose: adjudicate (a) does structural-6 help or hurt vs text-6, and
(b) is there nonlinear headroom the single-layer GNN (~0.787 AUC) is leaving on
the table — i.e. is the bottleneck the FUSION, or the SIGNAL?
"""

import sys
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, '.')
from utils import load_data, normalize

VAL_SIZE = 0.15


def frozen_split(n, labels):
    index = list(range(n))
    idx_train, idx_temp, y_train, y_temp = train_test_split(
        index, labels, stratify=labels, test_size=VAL_SIZE + 0.60,
        random_state=2, shuffle=True)
    val_fraction = VAL_SIZE / (VAL_SIZE + 0.60)
    idx_val, idx_test, y_val, y_test = train_test_split(
        idx_temp, y_temp, stratify=y_temp, test_size=1 - val_fraction,
        random_state=2, shuffle=True)
    return (np.array(idx_train), np.array(idx_val), np.array(idx_test),
            np.array(y_train), np.array(y_val), np.array(y_test))


def evaluate(X, y_all, idx_tr, idx_te, name):
    Xtr, Xte = X[idx_tr], X[idx_te]
    ytr, yte = y_all[idx_tr], y_all[idx_te]

    # Linear
    sc = StandardScaler().fit(Xtr)
    lr = LogisticRegression(max_iter=2000, class_weight='balanced')
    lr.fit(sc.transform(Xtr), ytr)
    p_lr = lr.predict_proba(sc.transform(Xte))[:, 1]
    auc_lr = roc_auc_score(yte, p_lr)
    ap_lr = average_precision_score(yte, p_lr)

    # Nonlinear
    gb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                        max_leaf_nodes=31, l2_regularization=1.0,
                                        random_state=2)
    gb.fit(Xtr, ytr)
    p_gb = gb.predict_proba(Xte)[:, 1]
    auc_gb = roc_auc_score(yte, p_gb)
    ap_gb = average_precision_score(yte, p_gb)

    print(f'  {name:28s} dim={X.shape[1]:4d}  LR AUC={auc_lr:.4f} AP={ap_lr:.4f}  '
          f'GBM AUC={auc_gb:.4f} AP={ap_gb:.4f}')
    return {'name': name, 'lr_auc': auc_lr, 'gb_auc': auc_gb}


def main():
    print('Loading YelpChi...')
    _, feat_data, labels = load_data('yelp')
    feat = normalize(feat_data)
    feat = np.asarray(feat.todense()) if hasattr(feat, 'todense') else np.asarray(feat)
    labels = np.asarray(labels).ravel()
    n = len(labels)

    text6 = torch.load('llm_embeddings/yelp/text_risk_scores.pt', weights_only=True).numpy()
    struct6 = torch.load('llm_embeddings/yelp/llm_risk_scores.pt', weights_only=True).numpy()
    print(f'n={n} feat={feat.shape} text6={text6.shape} struct6={struct6.shape}')

    # structural-6 per-column std (verifier claimed 4/6 near-constant)
    print('\nstructural-6 per-column [mean, std]:')
    for i in range(struct6.shape[1]):
        print(f'  col{i}: mean={struct6[:, i].mean():.3f} std={struct6[:, i].std():.3f}')
    print('text-6 per-column [mean, std]:')
    for i in range(text6.shape[1]):
        print(f'  col{i}: mean={text6[:, i].mean():.3f} std={text6[:, i].std():.3f}')

    idx_tr, idx_val, idx_te, _, _, _ = frozen_split(n, labels)
    print(f'\nsplit: train={len(idx_tr)} val={len(idx_val)} test={len(idx_te)} '
          f'(test fraud rate={labels[idx_te].mean():.3f})')

    print('\n=== Test-set AUC by feature set (GNN+CAPN+LLM reference ~0.787) ===')
    feature_sets = [
        ('raw32', feat),
        ('text6', text6),
        ('struct6', struct6),
        ('raw + text6', np.hstack([feat, text6])),
        ('raw + struct6', np.hstack([feat, struct6])),
        ('raw + text6 + struct6', np.hstack([feat, text6, struct6])),
        ('text6 + struct6', np.hstack([text6, struct6])),
    ]
    results = [evaluate(X, labels, idx_tr, idx_te, name) for name, X in feature_sets]

    print('\n=== verdicts ===')
    by = {r['name']: r for r in results}
    d_text = by['raw + text6']['lr_auc'] - by['raw32']['lr_auc']
    d_struct = by['raw + struct6']['lr_auc'] - by['raw32']['lr_auc']
    d_both = by['raw + text6 + struct6']['lr_auc'] - by['raw + text6']['lr_auc']
    print(f'  LR: text6 adds {d_text:+.4f} over raw; struct6 adds {d_struct:+.4f}; '
          f'struct6 on top of text6 adds {d_both:+.4f}')
    gbm_best = max(r['gb_auc'] for r in results)
    print(f'  best GBM AUC = {gbm_best:.4f} vs GNN ~0.787 '
          f'(gap {gbm_best - 0.787:+.4f} = nonlinear headroom the 1-layer GNN may miss)')


if __name__ == '__main__':
    main()
