"""GraphConsis (Liu et al., SIGIR 2020) in pure torch, on YelpChi.

Consistency-based neighbour selection: for each node, neighbours are scored by
feature consistency s(u,v) = exp(-||x_u - x_v||^2), hard-filtered at s <= eps,
then sampled with probability proportional to s (without replacement). Two
GraphSAGE-style layers (mean-agg + concat-self + linear), a scalar relation
gate per relation, L2-normalised, linear classifier.

Fidelity notes (paper and official code disagree; we follow the DGFraud/
DGFraud-TF2 code, which produced the published reproductions):
  - consistency on RAW input features at every layer (code), not per-layer
    hidden states (paper Eq. 3)
  - s uses the squared L2 norm (paper Eq. 3 + TF2)
  - eps = 0.001 as a hard filter on raw scores (TF2)
  - relation combination: un-normalised scalar gate alpha_r from
    [h_r || t_r] @ a, multiply, sum over relations, l2-normalize (code),
    not neighbour-level softmax attention (paper Eq. 5-6)
  - context embedding: DISABLED (default 0 in both official codebases)
  - fan-out [10, 5] (paper / DGFraud README run)

Validation target: published GraphConsis-on-YelpChi reproductions span
AUC 0.62 (CARE-GNN paper, unified 1-layer protocol) to 0.6983 +/- 0.0302
(PC-GNN paper, authors' code, 40/20/40, 10 runs). We anchor on the PC-GNN
protocol and accept the band [0.62, 0.73].

Modes:
  --validate   40/20/40 random splits (PC-GNN protocol) vs published band
  (default)    frozen 25/15/60 split, 5 seeds, raw32 vs raw32+text6

The sampler takes eps as a PER-NODE tensor so the CAPN policy can later set
per-node thresholds (the framework-integration hook).
"""

import argparse
import json
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')
from probe_yelp_headroom import frozen_split  # noqa: E402
from utils import load_data, normalize  # noqa: E402

SEEDS = [72, 73, 74, 75, 76]
MAX_DEGREE = 128     # adjacency cap (DGFraud)
FANOUT = [10, 5]     # samples per layer
EPS = 1e-3           # consistency filter threshold
HID = 128            # hidden dim (DGFraud-TF2 nhid)


def build_neigh_arrays(adj, n):
    """Ragged neighbour lists -> (padded id array, mask), capped at MAX_DEGREE."""
    ids = np.zeros((n, MAX_DEGREE), dtype=np.int64)
    mask = np.zeros((n, MAX_DEGREE), dtype=bool)
    rng = np.random.default_rng(0)  # deterministic cap
    for v in range(n):
        nbrs = np.fromiter((u for u in adj.get(v, ()) if u != v), dtype=np.int64)
        if len(nbrs) > MAX_DEGREE:
            nbrs = rng.choice(nbrs, MAX_DEGREE, replace=False)
        ids[v, :len(nbrs)] = nbrs
        mask[v, :len(nbrs)] = True
    return ids, mask


class ConsisSampler:
    """Consistency-filtered, consistency-proportional neighbour sampler.

    eps may be a scalar or a per-node array — the CAPN integration hook.
    """

    def __init__(self, feat, rel_arrays, rng):
        self.feat = feat                      # raw (row-normalised) features
        self.rel_arrays = rel_arrays          # [(ids, mask)] per relation
        self.rng = rng

    def sample(self, nodes, r_idx, k, eps=EPS, return_stats=False):
        nodes = np.asarray(nodes)
        ids, mask = self.rel_arrays[r_idx]
        nbr_ids = ids[nodes]                                  # (B, MAX_DEGREE)
        nbr_mask = mask[nodes]
        diff = self.feat[nbr_ids] - self.feat[nodes][:, None, :]
        s = np.exp(-np.square(diff).sum(-1)) * nbr_mask       # consistency
        eps_arr = np.broadcast_to(np.asarray(eps, dtype=s.dtype), (len(nodes),))
        valid = s > eps_arr[:, None]
        # Gumbel-top-k == weighted sampling without replacement, fully vectorised
        with np.errstate(divide='ignore'):
            keys = np.where(valid, np.log(s), -np.inf)
        keys = keys - np.log(-np.log(self.rng.random(s.shape)))
        keys[~valid] = -np.inf
        top = np.argpartition(-keys, kth=min(k, keys.shape[1] - 1), axis=1)[:, :k]
        picked = np.take_along_axis(nbr_ids, top, axis=1)
        ok = np.take_along_axis(valid, top, axis=1)
        # pad: repeat first valid pick; nodes with no valid neighbour -> self
        first = picked[np.arange(len(nodes)), ok.argmax(1)]
        fill = np.where(ok.any(1), first, nodes)
        out = np.where(ok, picked, fill[:, None])
        if return_stats:
            s_kept = np.take_along_axis(s, top, axis=1)
            denom = np.maximum(ok.sum(1), 1)
            mean_s = (s_kept * ok).sum(1) / denom             # per-node kept consistency
            return out, mean_s
        return out


class GraphConsis(nn.Module):
    def __init__(self, in_dim, num_rel, hid=HID, num_layers=2):
        super().__init__()
        self.num_rel = num_rel
        self.num_layers = num_layers
        dims = [in_dim] + [hid] * num_layers
        # per-layer: shared linear over [agg || self]
        self.lins = nn.ModuleList(
            nn.Linear(2 * dims[i], dims[i + 1]) for i in range(num_layers))
        self.rel_vecs = nn.ParameterList(
            nn.Parameter(torch.empty(num_rel, dims[i + 1])) for i in range(num_layers))
        self.att = nn.ParameterList(
            nn.Parameter(torch.empty(2 * dims[i + 1], 1)) for i in range(num_layers))
        for p in list(self.rel_vecs) + list(self.att):
            nn.init.xavier_uniform_(p)
        self.clf = nn.Linear(hid, 2)

    def layer(self, li, h_self, h_neigh):
        """h_self (B,d); h_neigh: list of R tensors (B,k,d), one per relation."""
        outs = []
        for r in range(self.num_rel):
            agg = h_neigh[r].mean(1)                          # mean aggregator
            out = self.lins[li](torch.cat([agg, h_self], 1))  # (B,hid)
            t = self.rel_vecs[li][r].expand(out.size(0), -1)
            alpha = torch.cat([out, t], 1) @ self.att[li]     # scalar gate
            outs.append(alpha * out)
        h = torch.stack(outs).sum(0)
        return F.normalize(h, p=2, dim=1)


def forward_batch(model, sampler, feat_t, nodes, eps=EPS):
    """2-layer minibatch forward with per-relation fan-out sampling.

    eps: scalar or per-node array over `nodes`; applied only at the centre
    nodes' own sampling step (the CAPN-controllable decision). Inner hops use
    the global EPS.
    """
    device = feat_t.device
    R = model.num_rel
    B, k1 = len(nodes), FANOUT[0]
    nodes = np.asarray(nodes)

    def layer1_states(node_arr, k):
        """Layer-1 states for a flat node array: aggregate each node's own
        k-sampled neighbourhood in every relation, combine relations."""
        nb = [feat_t[torch.as_tensor(sampler.sample(node_arr, r, k, EPS),
                                     device=device)] for r in range(R)]
        return model.layer(0, feat_t[torch.as_tensor(node_arr, device=device)], nb)

    # layer-2 neighbour sets for the centre nodes (per-node eps applies here)
    hop1 = [sampler.sample(nodes, r, k1, eps) for r in range(R)]
    h1 = [layer1_states(hop1[r].reshape(-1), FANOUT[1]).view(B, k1, -1)
          for r in range(R)]
    h_center = layer1_states(nodes, FANOUT[1])
    h2 = model.layer(1, h_center, h1)
    return model.clf(h2)


def run_once(feat, y, rel_arrays, idx_tr, idx_val, idx_te, seed,
             epochs=30, lr=0.01, batch_size=512, device='cuda'):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = torch.device(device if torch.cuda.is_available() or device == 'cpu' else 'cpu')
    sampler = ConsisSampler(feat, rel_arrays, rng)
    feat_t = torch.tensor(feat, dtype=torch.float32, device=dev)
    model = GraphConsis(feat.shape[1], len(rel_arrays)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    yt = torch.tensor(y, dtype=torch.long, device=dev)

    def predict(idx):
        model.eval()
        ps = []
        with torch.no_grad():
            for i in range(0, len(idx), 2048):
                chunk = idx[i:i + 2048]
                logits = forward_batch(model, sampler, feat_t, chunk)
                ps.append(torch.softmax(logits, 1)[:, 1].cpu().numpy())
        return np.concatenate(ps)

    best = None
    for _ in range(epochs):
        model.train()
        order = rng.permutation(idx_tr)
        for i in range(0, len(order), batch_size):
            batch = order[i:i + batch_size]
            opt.zero_grad()
            logits = forward_batch(model, sampler, feat_t, batch)
            loss = F.cross_entropy(logits, yt[torch.as_tensor(batch, device=dev)])
            loss.backward()
            opt.step()
        p_val = predict(idx_val)
        val_auc = roc_auc_score(y[idx_val], p_val)
        if best is None or val_auc > best[0]:
            p_te = predict(idx_te)
            best = (val_auc, roc_auc_score(y[idx_te], p_te),
                    average_precision_score(y[idx_te], p_te))
    return best


# ===================== CAPN-integrated variant =====================
# The framework wrapper: the IDENTICAL CAPN machinery from capn.py
# (PolicyNetwork, ValueNetwork, ShapedRewardComputer, TextFeatureGate) is
# attached unchanged. Backbone-specific parts are only (a) the state features
# and (b) the actuation point: the per-node consistency threshold eps that
# replaces GraphConsis's global eps=1e-3.
from capn import (PolicyNetwork, ValueNetwork,  # noqa: E402
                  ShapedRewardComputer, TextFeatureGate)

LAMBDA_POLICY = 0.15   # matches the main framework runs (v3 protocol)
LAMBDA_CRITIC = 0.1
WARMUP_EP, RAMP_EP = 5, 5


def precompute_selection_stats(feat, rel_arrays, chunk=4096):
    """Per (node, relation): [deg/MAX_DEGREE, mean_s, std_s, frac(s>EPS)].

    Consistency depends only on raw features, so this is computed once.
    """
    n = feat.shape[0]
    stats = np.zeros((n, len(rel_arrays), 4), dtype=np.float32)
    for r, (ids, mask) in enumerate(rel_arrays):
        for lo in range(0, n, chunk):
            sl = slice(lo, min(lo + chunk, n))
            nbr = ids[sl]
            m = mask[sl]
            diff = feat[nbr] - feat[sl][:, None, :]
            s = np.exp(-np.square(diff).sum(-1)) * m
            deg = np.maximum(m.sum(1), 1)
            mean_s = s.sum(1) / deg
            var_s = (np.square(s - mean_s[:, None]) * m).sum(1) / deg
            stats[sl, r, 0] = m.sum(1) / MAX_DEGREE
            stats[sl, r, 1] = mean_s
            stats[sl, r, 2] = np.sqrt(var_s)
            stats[sl, r, 3] = (s > EPS).sum(1) / deg
    return stats


def forward_batch_capn(model, sampler, feat_t, nodes, eps_per_rel):
    """forward_batch with per-relation per-node eps; also returns the mean
    kept-consistency per relation (the reward signal)."""
    device = feat_t.device
    R = model.num_rel
    B, k1 = len(nodes), FANOUT[0]
    nodes = np.asarray(nodes)

    def layer1_states(node_arr, k):
        nb = [feat_t[torch.as_tensor(sampler.sample(node_arr, r, k, EPS),
                                     device=device)] for r in range(R)]
        return model.layer(0, feat_t[torch.as_tensor(node_arr, device=device)], nb)

    hop1, mean_s = [], []
    for r in range(R):
        picked, ms = sampler.sample(nodes, r, k1, eps_per_rel[r], return_stats=True)
        hop1.append(picked)
        mean_s.append(float(ms.mean()))
    h1 = [layer1_states(hop1[r].reshape(-1), FANOUT[1]).view(B, k1, -1)
          for r in range(R)]
    h_center = layer1_states(nodes, FANOUT[1])
    h2 = model.layer(1, h_center, h1)
    return model.clf(h2), mean_s


def run_capn_once(feat, text6, y, rel_arrays, sel_stats, idx_tr, idx_val, idx_te,
                  seed, use_llm=True, epochs=30, lr=0.01, batch_size=512,
                  device='cuda'):
    """GraphConsis + framework: CAPN per-node eps policy (+ LLM gate/state)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = torch.device(device if torch.cuda.is_available() or device == 'cpu' else 'cpu')
    R = len(rel_arrays)
    sampler = ConsisSampler(feat, rel_arrays, rng)   # sampling always on raw feats
    feat_t_raw = torch.tensor(feat, dtype=torch.float32, device=dev)
    yt = torch.tensor(y, dtype=torch.long, device=dev)

    state_dim = feat.shape[1] + 4 + (text6.shape[1] if use_llm else 0)
    state_base = np.concatenate([feat, text6], 1).astype(np.float32) if use_llm else feat
    state_t = torch.tensor(state_base, dtype=torch.float32, device=dev)
    stats_t = torch.tensor(sel_stats, dtype=torch.float32, device=dev)

    model = GraphConsis(feat.shape[1], R).to(dev)
    policy = PolicyNetwork(state_dim, hidden_dim=64, num_relations=R).to(dev)
    value = ValueNetwork(state_dim, hidden_dim=64).to(dev)
    gate = TextFeatureGate(feat.shape[1], text6.shape[1]).to(dev) if use_llm else None
    reward_computer = ShapedRewardComputer()
    text_t = torch.tensor(text6, dtype=torch.float32, device=dev) if use_llm else None

    groups = [{'params': model.parameters(), 'lr': lr},
              {'params': policy.parameters(), 'lr': 3e-3},
              {'params': value.parameters(), 'lr': 1e-3}]
    if gate is not None:
        groups.append({'params': gate.parameters(), 'lr': lr})
    opt = torch.optim.Adam(groups)

    def gated_features():
        return gate(feat_t_raw, text_t) if gate is not None else feat_t_raw

    def states_for(batch_idx):
        b = torch.as_tensor(batch_idx, device=dev)
        return [torch.cat([state_t[b], stats_t[b, r]], 1) for r in range(R)]

    def eps_for(batch_idx, train_flag):
        S = states_for(batch_idx)
        policy.reset_episode()
        eps_list, logp = [], []
        for r in range(R):
            e, lp = policy(S[r], r, deterministic=not train_flag)
            policy.store_action(lp, e)
            eps_list.append(e.detach().cpu().numpy())
            logp.append(lp)
        return S, eps_list

    def predict(idx):
        model.eval(); policy.eval()
        ft = gated_features()
        ps = []
        for i in range(0, len(idx), 2048):
            chunk = idx[i:i + 2048]
            _, eps_list = eps_for(chunk, train_flag=False)
            with torch.no_grad():
                logits, _ = forward_batch_capn(model, sampler, ft.detach(), chunk, eps_list)
            ps.append(torch.softmax(logits, 1)[:, 1].cpu().numpy())
        return np.concatenate(ps)

    best = None
    for ep in range(epochs):
        model.train(); policy.train()
        lam = 0.0 if ep < WARMUP_EP else LAMBDA_POLICY * min(1.0, (ep - WARMUP_EP + 1) / RAMP_EP)
        reward_computer.reset_epoch()
        order = rng.permutation(idx_tr)
        for i in range(0, len(order), batch_size):
            batch = order[i:i + batch_size]
            opt.zero_grad()
            S, eps_list = eps_for(batch, train_flag=True)
            logits, mean_s = forward_batch_capn(model, sampler, gated_features(),
                                                batch, eps_list)
            yb = yt[torch.as_tensor(batch, device=dev)]
            ce = F.cross_entropy(logits, yb)
            avg_dist = 1.0 - float(np.mean(mean_s))            # inconsistency
            batch_acc = float((logits.argmax(1) == yb).float().mean())
            raw_r, _ = reward_computer.compute_reward(
                avg_dist, batch_acc, policy._thresholds, return_raw=True)
            V = torch.stack([value(S[r]).squeeze(-1) for r in range(R)], 1)  # [B,R]
            adv = (raw_r - V).detach()
            p_loss = policy.get_policy_loss(adv)
            c_loss = ((V - raw_r) ** 2).mean()
            (ce + lam * p_loss + LAMBDA_CRITIC * c_loss).backward()
            opt.step()
        p_val = predict(idx_val)
        val_auc = roc_auc_score(y[idx_val], p_val)
        if best is None or val_auc > best[0]:
            p_te = predict(idx_te)
            best = (val_auc, roc_auc_score(y[idx_te], p_te),
                    average_precision_score(y[idx_te], p_te))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--validate', action='store_true',
                    help='PC-GNN protocol 40/20/40; published band AUC [0.62, 0.73]')
    ap.add_argument('--smoke', action='store_true', help='1 seed, 2 epochs, CPU')
    ap.add_argument('--capn', action='store_true',
                    help='framework-integrated rows: +CAPN and +CAPN+LLM, frozen split')
    ap.add_argument('--capn-smoke', action='store_true',
                    help='2-epoch CPU smoke of the CAPN-integrated variant')
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    adj_lists, feat_data, labels = load_data('yelp')
    feat = normalize(feat_data)   # L1 row-norm, as in DGFraud-TF2 + CARE pipeline
    feat = np.asarray(feat.todense() if hasattr(feat, 'todense') else feat,
                      dtype=np.float32)
    y = np.asarray(labels).ravel()
    n = len(y)
    # relations only (skip homo at index 0)
    rel_arrays = [build_neigh_arrays(a, n) for a in adj_lists[1:4]]
    text6 = torch.load('llm_embeddings/yelp/text_risk_scores.pt', weights_only=True).numpy()

    if args.smoke:
        idx_tr, idx_val, idx_te, _, _, _ = frozen_split(n, y)
        b = run_once(feat, y, rel_arrays, idx_tr[:2000], idx_val[:2000], idx_te[:4000],
                     72, epochs=2, device='cpu')
        print(f'smoke: val={b[0]:.4f} test AUC={b[1]:.4f} AP={b[2]:.4f}')
        return

    if args.capn_smoke:
        idx_tr, idx_val, idx_te, _, _, _ = frozen_split(n, y)
        sel_stats = precompute_selection_stats(feat, rel_arrays)
        b = run_capn_once(feat, text6, y, rel_arrays, sel_stats,
                          idx_tr[:2000], idx_val[:2000], idx_te[:4000],
                          72, use_llm=True, epochs=2, device='cpu')
        print(f'capn smoke: val={b[0]:.4f} test AUC={b[1]:.4f} AP={b[2]:.4f}')
        return

    if args.capn:
        idx_tr, idx_val, idx_te, _, _, _ = frozen_split(n, y)
        sel_stats = precompute_selection_stats(feat, rel_arrays)
        out = {}
        for vname, use_llm in [('GraphConsis+CAPN', False),
                               ('GraphConsis+CAPN+LLM (framework)', True)]:
            runs = [run_capn_once(feat, text6, y, rel_arrays, sel_stats,
                                  idx_tr, idx_val, idx_te, s, use_llm=use_llm,
                                  epochs=args.epochs, device=args.device)
                    for s in SEEDS]
            aucs = [r[1] for r in runs]
            aps = [r[2] for r in runs]
            out[vname] = {'auc_mean': float(np.mean(aucs)),
                          'auc_std': float(np.std(aucs, ddof=1)),
                          'ap_mean': float(np.mean(aps)),
                          'ap_std': float(np.std(aps, ddof=1)),
                          'per_seed_auc': [float(a) for a in aucs]}
            print(f'{vname:34s} AUC={np.mean(aucs):.4f}±{np.std(aucs, ddof=1):.4f}  '
                  f'AP={np.mean(aps):.4f}', flush=True)
        # paired tests vs the standalone rows from the stage-2 run
        try:
            with open('results/experiments/graphconsis_table.json', encoding='utf-8') as f:
                stand = json.load(f)['results']
            comparisons = [
                ('RL pillar (CAPN vs standalone)', 'GraphConsis+CAPN', stand['raw32']),
                ('framework vs standalone', 'GraphConsis+CAPN+LLM (framework)', stand['raw32']),
                ('RL on top of LLM (framework vs +text6)',
                 'GraphConsis+CAPN+LLM (framework)', stand['raw32+text6']),
            ]
            for name, a, b in comparisons:
                t, p = stats.ttest_rel(out[a]['per_seed_auc'], b['per_seed_auc'])
                d = (out[a]['auc_mean'] - b['auc_mean']) * 100
                out[name] = {'delta_pp': float(d), 't': float(t), 'p': float(p),
                             'sig': bool(p < 0.05)}
                print(f'{name}: {d:+.2f}pp  t={t:.2f} p={p:.4f} '
                      f'{"SIG" if p < 0.05 else "n.s."}')
        except FileNotFoundError:
            print('standalone graphconsis_table.json not found — tests skipped')
        with open('results/experiments/graphconsis_capn_table.json', 'w',
                  encoding='utf-8') as f:
            json.dump({'protocol': 'frozen YelpChi 25/15/60, val-AUC ckpt, identical '
                                   'capn.py machinery, lambda_policy 0.15 warmup 5 '
                                   'ramp 5, per-node eps actuation', 'results': out},
                      f, indent=2)
        print('-> results/experiments/graphconsis_capn_table.json')
        return

    if args.validate:
        print('=== validation: 40/20/40, published band AUC [0.62, 0.73] '
              '(PC-GNN repro 0.6983±0.0302) ===')
        aucs = []
        for seed in SEEDS:
            idx = np.arange(n)
            idx_tr, idx_rest, _, y_rest = train_test_split(
                idx, y, stratify=y, train_size=0.40, random_state=seed)
            idx_val, idx_te, _, _ = train_test_split(
                idx_rest, y_rest, stratify=y_rest, train_size=1 / 3, random_state=seed)
            v, a, p = run_once(feat, y, rel_arrays, idx_tr, idx_val, idx_te, seed,
                               epochs=args.epochs, device=args.device)
            aucs.append(a)
            print(f'  seed {seed}: test AUC={a:.4f} AP={p:.4f} (val {v:.4f})', flush=True)
        print(f'  GraphConsis: AUC {np.mean(aucs):.4f} ± {np.std(aucs, ddof=1):.4f}')
        return

    idx_tr, idx_val, idx_te, _, _, _ = frozen_split(n, y)
    feats = {'raw32': feat,
             'raw32+text6': np.hstack([feat, text6]).astype(np.float32)}
    out = {}
    for fname, X in feats.items():
        runs = [run_once(X, y, rel_arrays, idx_tr, idx_val, idx_te, s,
                         epochs=args.epochs, device=args.device) for s in SEEDS]
        aucs = [r[1] for r in runs]
        aps = [r[2] for r in runs]
        out[fname] = {'auc_mean': float(np.mean(aucs)), 'auc_std': float(np.std(aucs, ddof=1)),
                      'ap_mean': float(np.mean(aps)), 'ap_std': float(np.std(aps, ddof=1)),
                      'per_seed_auc': [float(a) for a in aucs]}
        print(f'GraphConsis {fname:12s} AUC={np.mean(aucs):.4f}±{np.std(aucs, ddof=1):.4f}  '
              f'AP={np.mean(aps):.4f}', flush=True)
    t, p = stats.ttest_rel(out['raw32+text6']['per_seed_auc'], out['raw32']['per_seed_auc'])
    delta = (out['raw32+text6']['auc_mean'] - out['raw32']['auc_mean']) * 100
    out['text6_delta'] = {'delta_pp': float(delta), 't': float(t), 'p': float(p),
                          'sig': bool(p < 0.05 and delta > 0)}
    print(f'text6 delta {delta:+.2f}pp  t={t:.2f} p={p:.4f}')
    with open('results/experiments/graphconsis_table.json', 'w', encoding='utf-8') as f:
        json.dump({'protocol': 'frozen YelpChi 25/15/60, val-AUC ckpt, fan-out [10,5], '
                               'eps=1e-3, Haiku text6', 'results': out}, f, indent=2)
    print('-> results/experiments/graphconsis_table.json')


if __name__ == '__main__':
    main()
