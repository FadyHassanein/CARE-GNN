"""XGB-Graph (GADBench-style) +/- LLM text6 on the FROZEN YelpChi split.

GADBench's winning YelpChi method: parameter-free L-hop mean aggregation of
node features over the homogeneous graph, concat [h0 || h1 || ... || hL],
feed a tree ensemble. Here: XGBoost (n_estimators picked by val AUC via early
stopping — our protocol's val split) and HistGBM (5 seeds, paired t-test).

Rows per model: raw32 / raw32+text6, each with and without hop aggregation.
text6 is treated as node features, so it is aggregated over neighbours too.

Output: results/experiments/xgb_graph_table.json
"""

import json
import sys

import numpy as np
import scipy.sparse as sp
import torch
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier

sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')
from baseline_table import bootstrap_delta_ci  # noqa: E402
from probe_yelp_headroom import frozen_split  # noqa: E402
from utils import load_data, normalize  # noqa: E402

SEEDS = [72, 73, 74, 75, 76]
HOPS = 2  # GADBench default (performance saturates at 2)


def adjlist_to_norm_csr(adj, n):
    """Row-normalized (mean-agg) sparse adjacency from a dict-of-sets adjlist."""
    rows, cols = [], []
    for v, nbrs in adj.items():
        for u in nbrs:
            if u != v:  # self excluded; h0 already carries the node itself
                rows.append(v)
                cols.append(u)
    a = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                      shape=(n, n))
    deg = np.asarray(a.sum(1)).ravel()
    deg[deg == 0] = 1.0
    return sp.diags(1.0 / deg) @ a


def hop_features(X, a_norm, hops=HOPS):
    """[h0 || h1 || ... || hL] with mean aggregation."""
    out = [X]
    h = X
    for _ in range(hops):
        h = a_norm @ h
        out.append(h)
    return np.hstack(out)


def eval_xgb(X, y, idx_tr, idx_val, idx_te):
    """XGBoost, n_estimators by early stopping on val AUC (deterministic)."""
    clf = XGBClassifier(n_estimators=1000, eval_metric='auc',
                        early_stopping_rounds=50,
                        scale_pos_weight=float((y[idx_tr] == 0).sum() / (y[idx_tr] == 1).sum()),
                        random_state=2, n_jobs=-1)
    clf.fit(X[idx_tr], y[idx_tr], eval_set=[(X[idx_val], y[idx_val])], verbose=False)
    p = clf.predict_proba(X[idx_te])[:, 1]
    return p, roc_auc_score(y[idx_te], p), average_precision_score(y[idx_te], p)


def eval_gbm(X, y, idx_tr, idx_te, seed):
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                         max_leaf_nodes=31, l2_regularization=1.0,
                                         random_state=seed)
    clf.fit(X[idx_tr], y[idx_tr])
    p = clf.predict_proba(X[idx_te])[:, 1]
    return p, roc_auc_score(y[idx_te], p), average_precision_score(y[idx_te], p)


def main():
    adj_lists, feat_data, labels = load_data('yelp')
    homo = adj_lists[0]
    feat = normalize(feat_data)
    feat = np.asarray(feat.todense()) if hasattr(feat, 'todense') else np.asarray(feat)
    y = np.asarray(labels).ravel()
    n = len(y)
    text6 = torch.load('llm_embeddings/yelp/text_risk_scores.pt', weights_only=True).numpy()
    idx_tr, idx_val, idx_te, _, _, _ = frozen_split(n, y)

    a_norm = adjlist_to_norm_csr(homo, n)
    raw = feat.astype(np.float32)
    raw_t6 = np.hstack([feat, text6]).astype(np.float32)

    feature_sets = {
        'raw32': raw,
        'raw32+text6': raw_t6,
        'agg(raw32)': hop_features(raw, a_norm),
        'agg(raw32+text6)': hop_features(raw_t6, a_norm),
    }

    out = {'xgb': {}, 'gbm': {}}
    preds_xgb, preds_gbm = {}, {}
    print(f'n={n}  edges(homo)={a_norm.nnz}  hops={HOPS}')
    for name, X in feature_sets.items():
        p, auc, ap = eval_xgb(X, y, idx_tr, idx_val, idx_te)
        preds_xgb[name] = p
        out['xgb'][name] = {'auc': float(auc), 'ap': float(ap)}
        print(f'XGB  {name:18s} dim={X.shape[1]:4d}  AUC={auc:.4f}  AP={ap:.4f}', flush=True)

        runs = [eval_gbm(X, y, idx_tr, idx_te, s) for s in SEEDS]
        aucs = [r[1] for r in runs]
        aps = [r[2] for r in runs]
        preds_gbm[name] = runs[0][0]
        out['gbm'][name] = {
            'auc_mean': float(np.mean(aucs)), 'auc_std': float(np.std(aucs, ddof=1)),
            'ap_mean': float(np.mean(aps)), 'ap_std': float(np.std(aps, ddof=1)),
            'per_seed_auc': [float(a) for a in aucs],
        }
        print(f'GBM  {name:18s} dim={X.shape[1]:4d}  AUC={np.mean(aucs):.4f}±{np.std(aucs, ddof=1):.4f}  '
              f'AP={np.mean(aps):.4f}', flush=True)

    # significance of the text6 delta, with and without aggregation
    sig = {}
    for base, aug in [('raw32', 'raw32+text6'), ('agg(raw32)', 'agg(raw32+text6)')]:
        lo, hi = bootstrap_delta_ci(y[idx_te], preds_xgb[base], preds_xgb[aug])
        sig[f'xgb: {aug} - {base}'] = {
            'delta_pp': (out['xgb'][aug]['auc'] - out['xgb'][base]['auc']) * 100,
            'bootstrap_ci': [lo, hi], 'sig': bool(lo > 0)}
        t, p = stats.ttest_rel(out['gbm'][aug]['per_seed_auc'],
                               out['gbm'][base]['per_seed_auc'])
        sig[f'gbm: {aug} - {base}'] = {
            'delta_pp': (out['gbm'][aug]['auc_mean'] - out['gbm'][base]['auc_mean']) * 100,
            't': float(t), 'p': float(p),
            'sig': bool(p < 0.05 and t > 0)}
    out['significance'] = sig
    for k, v in sig.items():
        print(f'{k}: {v}')

    with open('results/experiments/xgb_graph_table.json', 'w', encoding='utf-8') as f:
        json.dump({'protocol': f'frozen YelpChi 25/15/60 split, homo graph, '
                               f'mean-agg L={HOPS}, XGB early-stop on val AUC, '
                               f'Haiku text_risk_scores.pt',
                   'results': out}, f, indent=2)
    print('-> results/experiments/xgb_graph_table.json')


if __name__ == '__main__':
    main()
