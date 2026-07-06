"""BWGNN (Tang et al., ICML 2022) in pure torch, on YelpChi.

Beta-wavelet spectral GNN: MLP -> C+1 parallel polynomial filters of the
sym-normalized Laplacian -> concat -> MLP -> weighted CE. Homo variant runs
on the homogeneous graph; Hetero runs filters per relation then max-pools.
No DGL: filters are sparse matmuls (h = sum_k theta_k L^k h).

Modes:
  --validate   40/20/40 random splits (paper protocol) to check the reimpl
               against published YelpChi numbers (Hetero ~0.90, Homo ~0.84).
  (default)    frozen 25/15/60 split (our paper protocol), 5 model seeds,
               raw32 vs raw32+text6, paired t-test on the text6 delta.

Output: results/experiments/bwgnn_table.json (default mode).
"""

import argparse
import json
import sys
from math import comb

import numpy as np
import scipy.sparse as sp
import scipy.special
import torch
import torch.nn as nn
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')
from probe_yelp_headroom import frozen_split  # noqa: E402
from utils import load_data  # noqa: E402

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEEDS = [72, 73, 74, 75, 76]


def beta_wavelet_thetas(d):
    """Polynomial coefficients of (x/2)^i (1-x/2)^(d-i) / B(i+1, d-i+1)."""
    thetas = []
    for i in range(d + 1):
        coeff = np.zeros(d + 1)
        # (x/2)^i * (1-x/2)^(d-i) = sum_j C(d-i,j) (-1)^j x^(i+j) / 2^(i+j)
        for j in range(d - i + 1):
            coeff[i + j] = comb(d - i, j) * ((-1) ** j) / (2 ** (i + j))
        coeff /= scipy.special.beta(i + 1, d + 1 - i)
        thetas.append(torch.tensor(coeff, dtype=torch.float32))
    return thetas


def adjlist_to_csr(adj, n):
    rows, cols = [], []
    for v, nbrs in adj.items():
        for u in nbrs:
            if u != v:
                rows.append(v)
                cols.append(u)
    return sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                         shape=(n, n))


class SparseLaplacian:
    """L x = x - D^-1/2 A D^-1/2 x, as two cheap ops."""

    def __init__(self, a_csr):
        deg = np.asarray(a_csr.sum(1)).ravel()
        deg[deg == 0] = 1.0
        d_inv_sqrt = 1.0 / np.sqrt(deg)
        coo = a_csr.tocoo()
        idx = torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long)
        val = torch.tensor(coo.data, dtype=torch.float32)
        self.a = torch.sparse_coo_tensor(idx, val, a_csr.shape).coalesce().to(DEVICE)
        self.d_inv_sqrt = torch.tensor(d_inv_sqrt, dtype=torch.float32,
                                       device=DEVICE).unsqueeze(1)

    def __call__(self, x):
        return x - self.d_inv_sqrt * torch.sparse.mm(self.a, self.d_inv_sqrt * x)


class BWGNN(nn.Module):
    def __init__(self, in_dim, laplacians, hidden=64, order=2):
        """laplacians: list of SparseLaplacian (1 = Homo, >1 = Hetero max-pool)."""
        super().__init__()
        self.laplacians = laplacians
        self.thetas = beta_wavelet_thetas(order)
        self.mlp_in = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                                    nn.Linear(hidden, hidden), nn.ReLU())
        self.mlp_out = nn.Sequential(
            nn.Linear(hidden * len(self.thetas), hidden), nn.ReLU(),
            nn.Linear(hidden, 2))

    def _propagate(self, lap, h):
        outs = []
        for theta in self.thetas:
            acc = theta[0] * h
            hk = h
            for k in range(1, len(theta)):
                hk = lap(hk)
                acc = acc + theta[k] * hk
            outs.append(acc)
        return torch.cat(outs, dim=1)

    def forward(self, x):
        h = self.mlp_in(x)
        per_rel = [self.mlp_out(self._propagate(lap, h)) for lap in self.laplacians]
        return per_rel[0] if len(per_rel) == 1 else torch.stack(per_rel).max(0).values


class GatedBWGNN(BWGNN):
    """BWGNN with the framework's gated-residual text injection (Eq. gate):
    x_tilde = x + sigmoid(g) (+) (W_t t), preserving the raw feature dim.
    g is a per-dim gate initialised at -2 so sigmoid(g)~0.12 (start near raw)."""

    def __init__(self, raw_dim, text_dim, laplacians, hidden=64, order=2):
        super().__init__(raw_dim, laplacians, hidden=hidden, order=order)
        self.w_t = nn.Linear(text_dim, raw_dim, bias=False)
        self.gate = nn.Parameter(torch.full((raw_dim,), -2.0))

    def forward(self, xt_pair):
        x, t = xt_pair
        x = x + torch.sigmoid(self.gate) * self.w_t(t)
        return super().forward(x)


def _train_eval(model, xt, y, idx_tr, idx_val, idx_te, epochs, lr):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    yt = torch.tensor(y, dtype=torch.long, device=DEVICE)
    w = torch.tensor([1.0, (y[idx_tr] == 0).sum() / (y[idx_tr] == 1).sum()],
                     dtype=torch.float32, device=DEVICE)
    tr = torch.tensor(idx_tr, device=DEVICE)
    best = None
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        loss = nn.functional.cross_entropy(model(xt)[tr], yt[tr], weight=w)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            p = torch.softmax(model(xt), 1)[:, 1].cpu().numpy()
        val_auc = roc_auc_score(y[idx_val], p[idx_val])
        if best is None or val_auc > best[0]:
            best = (val_auc, roc_auc_score(y[idx_te], p[idx_te]),
                    average_precision_score(y[idx_te], p[idx_te]))
    return best


def run_gate_once(feat, text6, y, laps, idx_tr, idx_val, idx_te, seed,
                  epochs=100, lr=0.01):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = GatedBWGNN(feat.shape[1], text6.shape[1], laps).to(DEVICE)
    xt = (torch.tensor(feat, dtype=torch.float32, device=DEVICE),
          torch.tensor(text6, dtype=torch.float32, device=DEVICE))
    return _train_eval(model, xt, y, idx_tr, idx_val, idx_te, epochs, lr)


def run_once(X, y, laps, idx_tr, idx_val, idx_te, seed, epochs=100, lr=0.01):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = BWGNN(X.shape[1], laps).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    xt = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    yt = torch.tensor(y, dtype=torch.long, device=DEVICE)
    w = torch.tensor([1.0, (y[idx_tr] == 0).sum() / (y[idx_tr] == 1).sum()],
                     dtype=torch.float32, device=DEVICE)
    tr = torch.tensor(idx_tr, device=DEVICE)

    best = None  # (val_auc, test_auc, test_ap)
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        logits = model(xt)
        loss = nn.functional.cross_entropy(logits[tr], yt[tr], weight=w)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            p = torch.softmax(model(xt), 1)[:, 1].cpu().numpy()
        val_auc = roc_auc_score(y[idx_val], p[idx_val])
        if best is None or val_auc > best[0]:
            best = (val_auc, roc_auc_score(y[idx_te], p[idx_te]),
                    average_precision_score(y[idx_te], p[idx_te]))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--validate', action='store_true',
                    help='paper-protocol 40/20/40 sanity check vs published numbers')
    ap.add_argument('--gate-probe', action='store_true',
                    help='compare raw vs concat(text6) vs gated(text6) on BWGNN-Hetero')
    ap.add_argument('--epochs', type=int, default=100)
    args = ap.parse_args()

    adj_lists, feat_data, labels = load_data('yelp')
    # DGL FraudDataset (used by the official BWGNN code) column-standardizes
    # features; row-normalization breaks the spectral filters' scale.
    feat = StandardScaler().fit_transform(np.asarray(feat_data))
    y = np.asarray(labels).ravel()
    n = len(y)
    text6 = torch.load('llm_embeddings/yelp/text_risk_scores.pt', weights_only=True).numpy()
    text6 = StandardScaler().fit_transform(text6)  # same scale as the features

    lap_homo = [SparseLaplacian(adjlist_to_csr(adj_lists[0], n))]
    lap_het = [SparseLaplacian(adjlist_to_csr(a, n)) for a in adj_lists[1:4]]

    if args.validate:
        print('=== validation: 40/20/40 random splits, published Yelp: '
              'Hetero AUC ~0.905, Homo ~0.840 ===')
        for name, laps in [('Hetero', lap_het), ('Homo', lap_homo)]:
            aucs = []
            for seed in SEEDS:
                idx = np.arange(n)
                idx_tr, idx_rest, _, y_rest = train_test_split(
                    idx, y, stratify=y, train_size=0.40, random_state=seed)
                idx_val, idx_te, _, _ = train_test_split(
                    idx_rest, y_rest, stratify=y_rest, train_size=1 / 3,
                    random_state=seed)
                v, a, p = run_once(feat, y, laps, idx_tr, idx_val, idx_te, seed,
                                   epochs=args.epochs)
                aucs.append(a)
                print(f'  {name} seed {seed}: test AUC={a:.4f} AP={p:.4f} '
                      f'(val {v:.4f})', flush=True)
            print(f'  {name}: AUC {np.mean(aucs):.4f} ± {np.std(aucs, ddof=1):.4f}')
        return

    idx_tr, idx_val, idx_te, _, _, _ = frozen_split(n, y)

    if args.gate_probe:
        # Does our gated injection (Eq. gate) beat plain concat on a modern
        # backbone? Compare on BWGNN-Hetero, 5 seeds, val-AUC checkpoint.
        raw = feat.astype(np.float32)
        concat = np.hstack([feat, text6]).astype(np.float32)
        variants = {}
        for name, runner in [
            ('raw32', lambda s: run_once(raw, y, lap_het, idx_tr, idx_val, idx_te, s,
                                         epochs=args.epochs)),
            ('concat(text6)', lambda s: run_once(concat, y, lap_het, idx_tr, idx_val,
                                                 idx_te, s, epochs=args.epochs)),
            ('gate(text6)', lambda s: run_gate_once(raw, text6.astype(np.float32), y,
                                                    lap_het, idx_tr, idx_val, idx_te, s,
                                                    epochs=args.epochs)),
        ]:
            runs = [runner(s) for s in SEEDS]
            variants[name] = {'auc': [r[1] for r in runs], 'ap': [r[2] for r in runs]}
            print(f'BWGNN-Hetero {name:14s} AUC={np.mean(variants[name]["auc"]):.4f}±'
                  f'{np.std(variants[name]["auc"], ddof=1):.4f}  '
                  f'AP={np.mean(variants[name]["ap"]):.4f}', flush=True)
        out = {}
        for a, b in [('concat(text6)', 'raw32'), ('gate(text6)', 'raw32'),
                     ('gate(text6)', 'concat(text6)')]:
            t, p = stats.ttest_rel(variants[a]['auc'], variants[b]['auc'])
            delta = (np.mean(variants[a]['auc']) - np.mean(variants[b]['auc'])) * 100
            out[f'{a} - {b}'] = {'delta_pp': float(delta), 't': float(t),
                                 'p': float(p), 'sig': bool(p < 0.05 and delta > 0)}
            print(f'  {a} - {b}: {delta:+.2f}pp  t={t:.2f} p={p:.4f} '
                  f'{"SIG" if p < 0.05 else "n.s."}')
        with open('results/experiments/bwgnn_gate_probe.json', 'w', encoding='utf-8') as f:
            json.dump({'protocol': 'BWGNN-Hetero, frozen 25/15/60, 5 seeds, val-AUC ckpt',
                       'variants': {k: {'auc_mean': float(np.mean(v['auc'])),
                                        'auc_std': float(np.std(v['auc'], ddof=1)),
                                        'ap_mean': float(np.mean(v['ap'])),
                                        'per_seed_auc': v['auc']}
                                    for k, v in variants.items()},
                       'tests': out}, f, indent=2)
        print('-> results/experiments/bwgnn_gate_probe.json')
        return

    feats = {'raw32': feat.astype(np.float32),
             'raw32+text6': np.hstack([feat, text6]).astype(np.float32)}
    out = {}
    for vname, laps in [('BWGNN-Hetero', lap_het), ('BWGNN-Homo', lap_homo)]:
        out[vname] = {}
        for fname, X in feats.items():
            runs = [run_once(X, y, laps, idx_tr, idx_val, idx_te, s,
                             epochs=args.epochs) for s in SEEDS]
            aucs = [r[1] for r in runs]
            aps = [r[2] for r in runs]
            out[vname][fname] = {
                'auc_mean': float(np.mean(aucs)), 'auc_std': float(np.std(aucs, ddof=1)),
                'ap_mean': float(np.mean(aps)), 'ap_std': float(np.std(aps, ddof=1)),
                'per_seed_auc': [float(a) for a in aucs],
            }
            print(f'{vname:13s} {fname:12s} AUC={np.mean(aucs):.4f}±'
                  f'{np.std(aucs, ddof=1):.4f}  AP={np.mean(aps):.4f}', flush=True)
        t, p = stats.ttest_rel(out[vname]['raw32+text6']['per_seed_auc'],
                               out[vname]['raw32']['per_seed_auc'])
        delta = (out[vname]['raw32+text6']['auc_mean']
                 - out[vname]['raw32']['auc_mean']) * 100
        out[vname]['text6_delta'] = {'delta_pp': delta, 't': float(t), 'p': float(p),
                                     'sig': bool(p < 0.05 and delta > 0)}
        print(f'{vname}: text6 delta {delta:+.2f}pp  t={t:.2f} p={p:.4f}')

    with open('results/experiments/bwgnn_table.json', 'w', encoding='utf-8') as f:
        json.dump({'protocol': 'frozen YelpChi 25/15/60 split, val-AUC checkpoint, '
                               '100 epochs, h=64 C=2, Haiku text6', 'results': out},
                  f, indent=2)
    print('-> results/experiments/bwgnn_table.json')


if __name__ == '__main__':
    main()
