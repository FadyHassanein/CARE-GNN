"""Parametric multi-seed YelpChi experiment runner + parser.

Runs train.py for seeds 72/73/74 (the thesis convention), parses the test-set
metrics at each best-validation checkpoint, and writes a JSON summary with
mean +/- std compared to the documented Row-4 baseline.

The point: one command per experiment, identical methodology every time, so
iterating on architecture changes is fast and apples-to-apples.

Usage (from project root):
    python scripts/run_yelp_experiment.py --tag baseline_realhaiku_lp015 -- \
        --use-capn --use-actor-critic --gnn-warmup-epochs 5 \
        --lambda-policy-ramp-epochs 5 --llm-priors-file data/llm_priors/yelp_priors.json \
        --text-enrichment gate --text-state-enrichment --lambda-policy 0.15

Everything after the standalone `--` is passed through verbatim to train.py.
`--data yelp --model CARE --seed <s>` are added automatically per seed.
"""

import argparse
import json
import os
import re
import statistics as st
import subprocess
import sys
from datetime import datetime

PYTHON = r'C:\Users\fadyh\miniconda3\envs\torch_cuda\python.exe'
SEEDS = [72, 73, 74]

# Documented Row 4 (Full framework) baseline, doc/gnn_rl_llm_ablation_results.md
BASELINE = {'auc': 0.7856, 'auc_std': 0.0022, 'ap': 0.4108, 'f1': 0.6099}

# eval block format per epoch in the train.py log:
#   GNN F1: .., AUC: <test_auc>, AP: <test_ap>     <- TEST (idx_test)
#   Label F1: ..                                    <- test label module
#   GNN F1: .., AUC: <val_auc>, AP: <val_ap>        <- VALIDATION
#   Label F1: ..                                    <- val label module
#   Validation AUC: <val_auc>
GNN_LINE = re.compile(r'GNN F1: ([\d.]+), Acc: [\d.]+, Recall: [\d.]+, AUC: ([\d.]+), AP: ([\d.]+)')
VAL_LINE = re.compile(r'Validation AUC: ([\d.]+)')


def parse_best_checkpoint(log_path):
    """Return test (auc, ap, f1) at the epoch with the highest validation AUC."""
    gnn_lines = []          # rolling list of (f1, auc, ap) from every "GNN F1" line
    best = None             # (val_auc, test_f1, test_auc, test_ap)
    with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            m = GNN_LINE.search(line)
            if m:
                gnn_lines.append((float(m.group(1)), float(m.group(2)), float(m.group(3))))
                continue
            v = VAL_LINE.search(line)
            if v and len(gnn_lines) >= 2:
                val_auc = float(v.group(1))
                test_f1, test_auc, test_ap = gnn_lines[-2]   # [-2]=test, [-1]=val
                if best is None or val_auc > best[0]:
                    best = (val_auc, test_f1, test_auc, test_ap)
    if best is None:
        return None
    return {'val_auc': best[0], 'f1': best[1], 'auc': best[2], 'ap': best[3]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True, help='experiment name (log + result filenames)')
    ap.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    ap.add_argument('passthrough', nargs=argparse.REMAINDER,
                    help='args after `--` forwarded to train.py')
    args = ap.parse_args()

    extra = args.passthrough
    if extra and extra[0] == '--':
        extra = extra[1:]

    log_dir = os.path.join('logs', 'experiments', args.tag)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join('results', 'experiments'), exist_ok=True)

    per_seed = []
    for seed in args.seeds:
        log_path = os.path.join(log_dir, f'seed_{seed}.log')
        cmd = [PYTHON, 'train.py', '--data', 'yelp', '--model', 'CARE',
               '--seed', str(seed)] + extra
        print(f'[{datetime.now():%H:%M:%S}] seed {seed}: {" ".join(cmd)}', flush=True)
        with open(log_path, 'w', encoding='utf-8') as logf:
            subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, check=False)
        res = parse_best_checkpoint(log_path)
        if res is None:
            print(f'  WARNING: could not parse metrics from {log_path}', flush=True)
            continue
        res['seed'] = seed
        per_seed.append(res)
        print(f'  seed {seed}: test AUC={res["auc"]:.4f} AP={res["ap"]:.4f} '
              f'F1={res["f1"]:.4f} (val={res["val_auc"]:.4f})', flush=True)

    if not per_seed:
        print('No results parsed; aborting summary.', flush=True)
        sys.exit(1)

    aucs = [r['auc'] for r in per_seed]
    aps = [r['ap'] for r in per_seed]
    f1s = [r['f1'] for r in per_seed]
    summary = {
        'tag': args.tag,
        'extra_flags': extra,
        'seeds': [r['seed'] for r in per_seed],
        'per_seed': per_seed,
        'auc_mean': st.mean(aucs), 'auc_std': st.pstdev(aucs) if len(aucs) > 1 else 0.0,
        'ap_mean': st.mean(aps), 'ap_std': st.pstdev(aps) if len(aps) > 1 else 0.0,
        'f1_mean': st.mean(f1s), 'f1_std': st.pstdev(f1s) if len(f1s) > 1 else 0.0,
        'baseline': BASELINE,
        'auc_delta_vs_baseline': st.mean(aucs) - BASELINE['auc'],
    }
    out_path = os.path.join('results', 'experiments', f'{args.tag}.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print('\n' + '=' * 64)
    print(f'EXPERIMENT: {args.tag}')
    for r in per_seed:
        print(f'  seed {r["seed"]}: AUC={r["auc"]:.4f}  AP={r["ap"]:.4f}  F1={r["f1"]:.4f}')
    n = len(aucs)
    print(f'  AUC mean +/- std: {summary["auc_mean"]:.4f} +/- {summary["auc_std"]:.4f}  (n={n})')
    print(f'  AP  mean +/- std: {summary["ap_mean"]:.4f} +/- {summary["ap_std"]:.4f}')
    print(f'  F1  mean +/- std: {summary["f1_mean"]:.4f} +/- {summary["f1_std"]:.4f}')
    print(f'  documented baseline AUC: {BASELINE["auc"]:.4f} +/- {BASELINE["auc_std"]:.4f}')
    print(f'  delta vs baseline: {summary["auc_delta_vs_baseline"]:+.4f}')
    print(f'  -> {out_path}')
    print('=' * 64)


if __name__ == '__main__':
    main()
