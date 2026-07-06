"""Assemble the v3 4-row YelpChi table + paired t-tests into one summary JSON.

Reads results/experiments/v3_row{1..4}_5seed.json (from run_v3_table.sh) and
writes results/experiments/yelp_v3_n5_summary.json in the same shape as
yelp_corrected_n5_summary.json.
"""

import json

import numpy as np
from scipy import stats

ROWS = {
    'R1 GNN': 'v3_row1_5seed',
    'R2 GNN+RL': 'v3_row2_5seed',
    'R3 GNN+LLM': 'v3_row3_5seed',
    'R4 Full': 'v3_row4_5seed',
}
TESTS = [
    ('Full vs CARE-GNN', 'R4 Full', 'R1 GNN'),
    ('LLM pillar (R3-R1)', 'R3 GNN+LLM', 'R1 GNN'),
    ('RL alone (R2-R1)', 'R2 GNN+RL', 'R1 GNN'),
    ('RL on top of LLM (R4-R3)', 'R4 Full', 'R3 GNN+LLM'),
]


def main():
    rows = {}
    for name, tag in ROWS.items():
        with open(f'results/experiments/{tag}.json', encoding='utf-8') as f:
            d = json.load(f)
        per_seed = sorted(d['per_seed'], key=lambda r: r['seed'])
        rows[name] = {
            'auc_mean': d['auc_mean'], 'auc_std': d['auc_std'],
            'ap_mean': d['ap_mean'], 'ap_std': d['ap_std'],
            'f1_mean': d['f1_mean'], 'f1_std': d['f1_std'],
            'per_seed_auc': [r['auc'] for r in per_seed],
            'seeds': [r['seed'] for r in per_seed],
        }

    tests = []
    for name, a, b in TESTS:
        xa, xb = np.array(rows[a]['per_seed_auc']), np.array(rows[b]['per_seed_auc'])
        assert rows[a]['seeds'] == rows[b]['seeds'], f'seed mismatch in {name}'
        t, p = stats.ttest_rel(xa, xb)
        tests.append({'name': name, 'delta_pp': float((xa - xb).mean() * 100),
                      't': float(t), 'p': float(p), 'sig': bool(p < 0.05)})

    out = {
        'protocol': 'YelpChi 25/15/60 split, val-selected checkpoint, Haiku text '
                    'scores, v3 backbone (feat-encoder + ego-sep + cosine), 80 '
                    'epochs, lambda_policy=0.15 on CAPN rows, n=5 seeds 72-76',
        'rows': rows,
        'paired_tests': tests,
    }
    path = 'results/experiments/yelp_v3_n5_summary.json'
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)

    for name, r in rows.items():
        print(f'{name:12s} AUC {r["auc_mean"]:.4f} ± {r["auc_std"]:.4f}   '
              f'AP {r["ap_mean"]:.4f}   F1 {r["f1_mean"]:.4f}')
    for t in tests:
        print(f'{t["name"]:26s} {t["delta_pp"]:+.2f}pp  t={t["t"]:.2f}  '
              f'p={t["p"]:.4f}  {"SIG" if t["sig"] else "n.s."}')
    print(f'-> {path}')


if __name__ == '__main__':
    main()
