"""Throwaway probe for M8: quantify label leakage in node_statistics.npz.

For each saved statistic, compute its AUC vs the true label SEPARATELY on the
train / val / test splits. A statistic computed without test labels should show
similar train and test AUC; a big train>>test gap is a leak signature (the
statistic encodes a node's own label on the train split). Run from repo root.
"""
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm.compute_node_statistics import get_train_split  # noqa: E402

LABELS = {'yelp': 'data/yelp_labels.npy', 'amazon': 'data/amz_labels.npy'}


def col_auc(values, y):
    try:
        if len(np.unique(y)) < 2:
            return float('nan')
        return roc_auc_score(y, values)
    except Exception:
        return float('nan')


def split_auc(arr1d, labels, tr, va, te):
    out = {}
    for name, s in (('train', tr), ('val', va), ('test', te)):
        idx = sorted(s)
        out[name] = col_auc(arr1d[idx], labels[idx])
    return out


for data in ['yelp', 'amazon']:
    npz_path = f'llm_embeddings/{data}/node_statistics.npz'
    if not os.path.exists(npz_path):
        print(f'[{data}] no node_statistics.npz; skip'); continue
    npz = np.load(npz_path, allow_pickle=True)
    labels = np.load(LABELS[data])
    tr, va, te = get_train_split(labels, data=data)
    print(f'\n========== {data}  (train={len(tr)} val={len(va)} test={len(te)}) ==========')

    # focus: label_disagreement (the suspect "neighborhood_risk")
    ld = npz['label_disagreement']  # [N,3]
    ld_mean = ld.mean(axis=1)
    a = split_auc(ld_mean, labels, tr, va, te)
    print(f'label_disagreement (mean over 3 rel):  '
          f"train={a['train']:.3f}  val={a['val']:.3f}  test={a['test']:.3f}   "
          f"(train-test gap={a['train']-a['test']:+.3f})")
    for r in range(ld.shape[1]):
        ar = split_auc(ld[:, r], labels, tr, va, te)
        print(f'   relation {r}:                          '
              f"train={ar['train']:.3f}  val={ar['val']:.3f}  test={ar['test']:.3f}")

    # derived risk score dim 2 (neighborhood_risk) if present
    rs_path = f'llm_embeddings/{data}/llm_risk_scores.pt'
    if os.path.exists(rs_path):
        import torch
        rs = torch.load(rs_path, weights_only=True).numpy()
        if rs.shape[1] > 2:
            ar = split_auc(rs[:, 2], labels, tr, va, te)
            print(f'llm_risk_scores dim2 (neighborhood_risk): '
                  f"train={ar['train']:.3f}  val={ar['val']:.3f}  test={ar['test']:.3f}   "
                  f"(train-test gap={ar['train']-ar['test']:+.3f})")

    # scan every other stat for leak signatures (train>>test)
    print('  -- scan of all stats (test AUC, train-test gap) --')
    for key in npz.files:
        if key in ('relation_names', 'cross_pair_names', 'dataset'):
            continue
        arr = npz[key]
        if arr.ndim == 1:
            cols = {key: arr}
        else:
            cols = {f'{key}[{i}]': arr[:, i] for i in range(arr.shape[1])}
        for cname, cvals in cols.items():
            a = split_auc(cvals, labels, tr, va, te)
            gap = a['train'] - a['test']
            flag = '  <-- LEAK?' if (not np.isnan(gap) and gap > 0.05) else ''
            if (not np.isnan(a['test']) and abs(a['test'] - 0.5) > 0.03) or flag:
                print(f"     {cname:28s} test={a['test']:.3f}  gap={gap:+.3f}{flag}")
