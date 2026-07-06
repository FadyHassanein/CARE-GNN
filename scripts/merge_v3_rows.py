"""Merge the paused/refilled v3 rows into canonical 5-seed row JSONs.

R4: seeds 72-75 parsed from logs/experiments/v3_row4_5seed/seed_*.log (row
JSON was never written — runner killed at pause) + seed 76 from the
v3_row4_s76 run. R1: existing 2-seed JSON (72,73) + v3_row1_fill (74-76).
Writes v3_row{1,4}_5seed.json in run_yelp_experiment format, then
summarize_v3_table.py produces the final table.
"""

import json
import os
import statistics as st
import sys

sys.path.insert(0, 'scripts')
from run_yelp_experiment import parse_best_checkpoint  # noqa: E402


def load_per_seed(tag):
    with open(f'results/experiments/{tag}.json', encoding='utf-8') as f:
        return json.load(f)['per_seed']


def write_row(tag, per_seed, extra_flags):
    per_seed = sorted(per_seed, key=lambda r: r['seed'])
    seeds = [r['seed'] for r in per_seed]
    assert seeds == [72, 73, 74, 75, 76], f'{tag}: bad seed set {seeds}'
    aucs = [r['auc'] for r in per_seed]
    aps = [r['ap'] for r in per_seed]
    f1s = [r['f1'] for r in per_seed]
    out = {
        'tag': tag, 'extra_flags': extra_flags, 'seeds': seeds,
        'per_seed': per_seed,
        'auc_mean': st.mean(aucs), 'auc_std': st.stdev(aucs),
        'ap_mean': st.mean(aps), 'ap_std': st.stdev(aps),
        'f1_mean': st.mean(f1s), 'f1_std': st.stdev(f1s),
    }
    path = f'results/experiments/{tag}.json'
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print(f'{tag}: AUC {out["auc_mean"]:.4f} ± {out["auc_std"]:.4f}  (n=5) -> {path}')


def main():
    # R4: 72-75 from logs, 76 from the solo run
    r4 = []
    for s in (72, 73, 74, 75):
        log = f'logs/experiments/v3_row4_5seed/seed_{s}.log'
        res = parse_best_checkpoint(log)
        assert res, f'unparseable {log}'
        res['seed'] = s
        r4.append(res)
    r4 += load_per_seed('v3_row4_s76')
    write_row('v3_row4_5seed', r4, ['(merged: pause at seed 75; s76 solo)'])

    # R1: 72,73 from the original row + 74-76 from the fill
    if os.path.exists('results/experiments/v3_row1_fill.json'):
        r1 = load_per_seed('v3_row1_5seed') + load_per_seed('v3_row1_fill')
        write_row('v3_row1_5seed', r1, ['(merged: 72,73 original + 74-76 fill)'])
    else:
        print('v3_row1_fill.json not present yet — R1 left unmerged')


if __name__ == '__main__':
    main()
