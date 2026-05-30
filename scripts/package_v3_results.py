"""Package the v3 YelpChi improvement: parse authoritative logs, write a results
JSON, and generate thesis-ready figures.

v3 = nonlinear feature encoder + ego/neighbour separation + cosine LR (80 epochs),
on top of the full CARE-GNN + CAPN + real-Haiku-LLM framework.

Run: python scripts/package_v3_results.py
"""

import os
import re
import json
import statistics as st

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family': 'serif', 'font.size': 11, 'axes.titlesize': 13,
                     'axes.labelsize': 12, 'figure.dpi': 150})

GNN = re.compile(r'GNN F1: ([\d.]+), Acc: [\d.]+, Recall: [\d.]+, AUC: ([\d.]+), AP: ([\d.]+)')
VAL = re.compile(r'Validation AUC: ([\d.]+)')
EPOCH = re.compile(r'Epoch: (\d+),')

BASE = {72: 'logs/sweep_lambda_policy_realhaiku/lp_0.15.log',
        73: 'logs/multiseed_realhaiku_lp015/seed_73.log',
        74: 'logs/multiseed_realhaiku_lp015/seed_74.log'}
V3 = {s: f'logs/experiments/v3_80ep/seed_{s}.log' for s in (72, 73, 74)}


def best_ckpt(path):
    gnn, best = [], None
    for line in open(path, encoding='utf-8', errors='replace'):
        m = GNN.search(line)
        if m:
            gnn.append((float(m.group(1)), float(m.group(2)), float(m.group(3))))
            continue
        v = VAL.search(line)
        if v and len(gnn) >= 2:
            va = float(v.group(1))
            f1, auc, ap = gnn[-2]
            if best is None or va > best[0]:
                best = (va, f1, auc, ap)
    return {'val_auc': best[0], 'f1': best[1], 'auc': best[2], 'ap': best[3]}


def trajectory(path):
    """(epoch, val_auc) pairs."""
    epochs, last_epoch, out = [], 0, []
    for line in open(path, encoding='utf-8', errors='replace'):
        e = EPOCH.search(line)
        if e:
            last_epoch = int(e.group(1))
            continue
        v = VAL.search(line)
        if v:
            out.append((last_epoch, float(v.group(1))))
    return out


def stats(d):
    a = [d[s]['auc'] for s in d]; p = [d[s]['ap'] for s in d]; f = [d[s]['f1'] for s in d]
    return {'auc_mean': st.mean(a), 'auc_std': st.pstdev(a),
            'ap_mean': st.mean(p), 'ap_std': st.pstdev(p),
            'f1_mean': st.mean(f), 'f1_std': st.pstdev(f), 'per_seed': d}


def main():
    os.makedirs('figures/v3', exist_ok=True)
    os.makedirs('results/experiments', exist_ok=True)

    base = {s: best_ckpt(p) for s, p in BASE.items()}
    v3 = {s: best_ckpt(p) for s, p in V3.items()}
    bs, vs = stats(base), stats(v3)

    # seed-72 method progression ladder
    ladder = [
        ('Baseline\n(31ep)', best_ckpt(BASE[72])['auc']),
        ('+ encoder\n(31ep)', best_ckpt('logs/experiments/feat_encoder_v1/seed_72.log')['auc']),
        ('+ ego-sep\n+ cosine (31ep)', best_ckpt('logs/experiments/v2_combo/seed_72.log')['auc']),
        ('+ 80ep\n(v3)', best_ckpt(V3[72])['auc']),
    ]

    summary = {
        'config': 'CARE-GNN + CAPN(actor-critic, lp=0.15) + real-Haiku text gate '
                  '+ nonlinear feature encoder + ego-separation + cosine LR, 80 epochs',
        'baseline_realhaiku_31ep': bs, 'v3': vs,
        'delta_auc': vs['auc_mean'] - bs['auc_mean'],
        'delta_vs_documented_row4': vs['auc_mean'] - 0.7856,
        'documented_row4': {'auc': 0.7856, 'ap': 0.4108, 'f1': 0.6099},
        'seed72_progression': ladder,
    }
    json.dump(summary, open('results/experiments/v3_final_summary.json', 'w'), indent=2)

    # ---- Figure 1: AUC + AP comparison, mean +/- std ----
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, key, title in [(axes[0], 'auc', 'Test AUC'), (axes[1], 'ap', 'Test AP')]:
        means = [0.7856 if key == 'auc' else 0.4108, bs[f'{key}_mean'], vs[f'{key}_mean']]
        errs = [0.0022 if key == 'auc' else 0.0052, bs[f'{key}_std'], vs[f'{key}_std']]
        labels = ['Documented\nRow 4', 'Baseline\n(real-Haiku)', 'v3 (ours)']
        colors = ['#9aa0a6', '#5b8fd4', '#2e7d32']
        bars = ax.bar(labels, means, yerr=errs, capsize=5, color=colors, edgecolor='black', linewidth=0.6)
        ax.set_title(title)
        ax.set_ylim(min(means) - max(errs) - 0.01, max(means) + max(errs) + 0.01)
        for b, m in zip(bars, means):
            ax.text(b.get_x() + b.get_width() / 2, m + max(errs) * 0.4, f'{m:.4f}',
                    ha='center', va='bottom', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
    fig.suptitle('YelpChi: v3 vs baselines (3 seeds, mean ± std)', fontweight='bold')
    fig.tight_layout()
    fig.savefig('figures/v3/fig1_comparison.png', bbox_inches='tight')
    plt.close(fig)

    # ---- Figure 2: validation-AUC trajectory, seed 72 ----
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for path, lab, col in [(BASE[72], 'Baseline (31ep)', '#5b8fd4'), (V3[72], 'v3 (80ep)', '#2e7d32')]:
        tr = trajectory(path)
        if tr:
            xs, ys = zip(*tr)
            ax.plot(xs, ys, marker='o', ms=3, label=lab, color=col)
    ax.axhline(0.7856, ls='--', color='#9aa0a6', lw=1, label='Documented Row 4')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Validation AUC')
    ax.set_title('Validation-AUC trajectory (seed 72)\ncold-start → capacity engages → cosine settles')
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig('figures/v3/fig2_trajectory.png', bbox_inches='tight')
    plt.close(fig)

    # ---- Figure 3: seed-72 method progression ----
    fig, ax = plt.subplots(figsize=(7, 4.5))
    names = [l[0] for l in ladder]; vals = [l[1] for l in ladder]
    colors = ['#9aa0a6', '#d98b8b', '#e0b760', '#2e7d32']
    bars = ax.bar(names, vals, color=colors, edgecolor='black', linewidth=0.6)
    ax.axhline(ladder[0][1], ls='--', color='#9aa0a6', lw=1)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.001, f'{v:.4f}', ha='center', va='bottom', fontsize=9)
    ax.set_ylabel('Test AUC'); ax.set_ylim(0.77, 0.802)
    ax.set_title('Component progression (seed 72): capacity alone hurts;\nego-separation + cosine + budget unlock it')
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout(); fig.savefig('figures/v3/fig3_progression.png', bbox_inches='tight')
    plt.close(fig)

    # ---- console report ----
    print('=== v3 final summary ===')
    print(f'baseline (real-Haiku, 31ep): AUC {bs["auc_mean"]:.4f} +/- {bs["auc_std"]:.4f} | '
          f'AP {bs["ap_mean"]:.4f} | F1 {bs["f1_mean"]:.4f}')
    print(f'v3 (enc+ego+cosine, 80ep):   AUC {vs["auc_mean"]:.4f} +/- {vs["auc_std"]:.4f} | '
          f'AP {vs["ap_mean"]:.4f} | F1 {vs["f1_mean"]:.4f}')
    print(f'delta AUC vs baseline:   {summary["delta_auc"]:+.4f}')
    print(f'delta AUC vs documented: {summary["delta_vs_documented_row4"]:+.4f}')
    print('per-seed v3:', {s: round(v3[s]['auc'], 4) for s in v3})
    print('figures -> figures/v3/  | summary -> results/experiments/v3_final_summary.json')


if __name__ == '__main__':
    main()
