"""
Generate thesis-ready figures from ablation study results.

Usage:
    python generate_figures.py                         # all figures from all results
    python generate_figures.py --study capn             # specific study
    python generate_figures.py --results-dir results/ablation --output-dir figures/thesis
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch

# Publication style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

METRIC_LABELS = {
    'gnn_auc': 'AUC-ROC',
    'gnn_ap': 'AP',
    'gnn_f1': 'F1 (macro)',
    'gnn_accuracy': 'Accuracy',
    'gnn_recall': 'Recall (macro)',
    'gnn_precision': 'Precision (macro)',
    'label_auc': 'Label AUC',
    'label_ap': 'Label AP',
    'label_f1': 'Label F1',
}

COLORS = {
    'CARE-GNN': '#2196F3',
    'CAPN': '#FF5722',
    'CAPN+LLM': '#4CAF50',
    'default': ['#2196F3', '#FF5722', '#4CAF50', '#FFC107', '#9C27B0', '#00BCD4', '#795548'],
}


def load_results(results_dir):
    """Load all ablation results from directory tree."""
    all_results = {}
    for study_dir in sorted(glob.glob(os.path.join(results_dir, '*'))):
        if not os.path.isdir(study_dir):
            continue
        study_name = os.path.basename(study_dir)
        # load the most recent JSON
        json_files = sorted(glob.glob(os.path.join(study_dir, '*.json')))
        if json_files:
            with open(json_files[-1], 'r') as f:
                all_results[study_name] = json.load(f)
            print(f'  Loaded {study_name}: {json_files[-1]}')
    return all_results


def grouped_bar_chart(results, metrics, title, save_path, figsize=(10, 5)):
    """Grouped bar chart comparing multiple experiments across multiple metrics."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    exp_names = list(results.keys())
    n_exps = len(exp_names)
    n_metrics = len(metrics)

    x = np.arange(n_metrics)
    width = 0.8 / n_exps
    colors = COLORS['default'][:n_exps]

    fig, ax = plt.subplots(figsize=figsize)

    for i, name in enumerate(exp_names):
        vals = [results[name].get('best_metrics', {}).get(m, 0) for m in metrics]
        offset = (i - n_exps / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=name, color=colors[i],
                       edgecolor='white', linewidth=0.5)
        # value labels
        for bar, val in zip(bars, vals):
            if val > 0.01:
                ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.005,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=7, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS.get(m, m) for m in metrics])
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_ylim(0, min(1.05, max(
        max(results[n].get('best_metrics', {}).get(m, 0) for m in metrics for n in exp_names) + 0.1, 0.5)))
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f'  Saved: {save_path}')


def radar_chart(results, metrics, title, save_path):
    """Spider/radar chart comparing experiments across metrics."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    exp_names = list(results.keys())
    labels = [METRIC_LABELS.get(m, m) for m in metrics]
    n = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]
    colors = COLORS['default'][:len(exp_names)]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    for i, name in enumerate(exp_names):
        vals = [results[name].get('best_metrics', {}).get(m, 0) for m in metrics]
        vals += vals[:1]
        ax.plot(angles, vals, 'o-', linewidth=2, color=colors[i], label=name, markersize=4)
        ax.fill(angles, vals, alpha=0.1, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.set_title(title, y=1.08, fontsize=14)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), framealpha=0.9)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f'  Saved: {save_path}')


def delta_bar_chart(baseline_metrics, improved_metrics, baseline_name, improved_name,
                    metrics, title, save_path):
    """Bar chart showing improvement deltas (positive = better)."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    deltas = []
    labels = []
    for m in metrics:
        b = baseline_metrics.get(m, 0)
        imp = improved_metrics.get(m, 0)
        deltas.append(imp - b)
        labels.append(METRIC_LABELS.get(m, m))

    colors = ['#4CAF50' if d >= 0 else '#F44336' for d in deltas]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(deltas)), deltas, color=colors, edgecolor='white', linewidth=0.5)

    for bar, d in zip(bars, deltas):
        va = 'bottom' if d >= 0 else 'top'
        offset = 0.002 if d >= 0 else -0.002
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + offset,
                f'{d:+.4f}', ha='center', va=va, fontsize=9, fontweight='bold')

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.set_ylabel('Delta (improvement)')
    ax.set_title(f'{title}\n({improved_name} vs {baseline_name})')
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f'  Saved: {save_path}')


def latex_table(results, metrics, caption, label):
    """Generate LaTeX table from results."""
    exp_names = list(results.keys())
    header = ' & '.join(['Method'] + [METRIC_LABELS.get(m, m) for m in metrics])

    rows = []
    # find best per metric
    best_vals = {}
    for m in metrics:
        vals = [(n, results[n].get('best_metrics', {}).get(m, 0)) for n in exp_names]
        best_name = max(vals, key=lambda x: x[1])[0]
        best_vals[m] = best_name

    for name in exp_names:
        bm = results[name].get('best_metrics', {})
        cells = [name.replace('_', '\\_')]
        for m in metrics:
            val = bm.get(m, 0)
            cell = f'{val:.4f}'
            if best_vals[m] == name:
                cell = f'\\textbf{{{cell}}}'
            cells.append(cell)
        rows.append(' & '.join(cells))

    n_cols = len(metrics) + 1
    col_spec = 'l' + 'c' * len(metrics)
    nl = '\n'
    bs = '\\'
    row_sep = f' {bs}{bs}{nl}'

    lines = [
        f'{bs}begin{{table}}[ht]',
        f'{bs}centering',
        f'{bs}caption{{{caption}}}',
        f'{bs}label{{{label}}}',
        f'{bs}begin{{tabular}}{{{col_spec}}}',
        f'{bs}toprule',
        f'{header} {bs}{bs}',
        f'{bs}midrule',
    ]
    for row in rows:
        lines.append(f'{row} {bs}{bs}')
    lines += [
        f'{bs}bottomrule',
        f'{bs}end{{tabular}}',
        f'{bs}end{{table}}',
    ]
    return nl.join(lines)


def sensitivity_line_plot(results, param_name, metrics, title, save_path):
    """Line plot for hyperparameter sensitivity (e.g., lambda_policy)."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    exp_names = list(results.keys())
    # extract param values from experiment names
    param_vals = []
    for name in exp_names:
        config = results[name].get('config', {})
        if param_name in config:
            param_vals.append(config[param_name])
        else:
            # try parsing from name
            for part in name.split('='):
                try:
                    param_vals.append(float(part.strip()))
                    break
                except ValueError:
                    continue

    if len(param_vals) != len(exp_names):
        print(f'  Warning: could not extract {param_name} from all experiments')
        return

    sort_idx = np.argsort(param_vals)
    param_vals = [param_vals[i] for i in sort_idx]
    exp_names = [exp_names[i] for i in sort_idx]

    colors = COLORS['default'][:len(metrics)]
    fig, ax = plt.subplots(figsize=(8, 5))

    for i, m in enumerate(metrics):
        vals = [results[n].get('best_metrics', {}).get(m, 0) for n in exp_names]
        ax.plot(param_vals, vals, 'o-', color=colors[i], linewidth=2,
                markersize=6, label=METRIC_LABELS.get(m, m))

    ax.set_xlabel(param_name.replace('_', ' ').title())
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.legend()
    if len(param_vals) > 1 and max(param_vals) / max(min(param_vals), 1e-10) > 10:
        ax.set_xscale('log')
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f'  Saved: {save_path}')


def summary_dashboard(all_results, save_path):
    """Single-page dashboard with key metrics from all studies."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    studies = list(all_results.keys())
    n_studies = len(studies)
    if n_studies == 0:
        return

    fig = plt.figure(figsize=(16, 4 * ((n_studies + 1) // 2)))
    gs = gridspec.GridSpec((n_studies + 1) // 2, 2, hspace=0.4, wspace=0.3)
    key_metrics = ['gnn_auc', 'gnn_f1', 'gnn_ap', 'gnn_recall']

    for idx, study in enumerate(studies):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        results = all_results[study]
        exp_names = list(results.keys())
        n = len(exp_names)
        x = np.arange(len(key_metrics))
        width = 0.8 / max(n, 1)
        colors = COLORS['default'][:n]

        for i, name in enumerate(exp_names):
            vals = [results[name].get('best_metrics', {}).get(m, 0) for m in key_metrics]
            offset = (i - n / 2 + 0.5) * width
            ax.bar(x + offset, vals, width, label=name, color=colors[i], edgecolor='white')

        ax.set_xticks(x)
        ax.set_xticklabels([METRIC_LABELS.get(m, m) for m in key_metrics], fontsize=8)
        ax.set_title(study.replace('_', ' ').title(), fontsize=11, fontweight='bold')
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7, loc='upper right')

    fig.suptitle('CAPN Ablation Study Dashboard', fontsize=16, fontweight='bold')
    fig.savefig(save_path)
    plt.close(fig)
    print(f'  Saved: {save_path}')


def generate_all_figures(results_dir, output_dir):
    """Generate all thesis figures from results."""
    print(f'Loading results from {results_dir}...')
    all_results = load_results(results_dir)

    if not all_results:
        print('No results found!')
        return

    os.makedirs(output_dir, exist_ok=True)
    gnn_metrics = ['gnn_auc', 'gnn_ap', 'gnn_f1', 'gnn_accuracy', 'gnn_recall', 'gnn_precision']
    key_metrics = ['gnn_auc', 'gnn_ap', 'gnn_f1', 'gnn_recall']
    all_metrics = ['gnn_auc', 'gnn_ap', 'gnn_f1', 'gnn_recall', 'label_auc', 'label_ap']

    # -- Dashboard --
    print('\nGenerating dashboard...')
    summary_dashboard(all_results, os.path.join(output_dir, 'dashboard.pdf'))

    # -- Per-study figures --
    for study_name, results in all_results.items():
        print(f'\nGenerating figures for: {study_name}')
        study_dir = os.path.join(output_dir, study_name)

        # Grouped bar chart (all GNN metrics)
        grouped_bar_chart(results, gnn_metrics,
                          f'{study_name.replace("_", " ").title()} — GNN Metrics',
                          os.path.join(study_dir, 'bar_gnn_metrics.pdf'))

        # Grouped bar chart (including label metrics)
        grouped_bar_chart(results, all_metrics,
                          f'{study_name.replace("_", " ").title()} — All Metrics',
                          os.path.join(study_dir, 'bar_all_metrics.pdf'),
                          figsize=(12, 5))

        # Radar chart
        radar_chart(results, key_metrics,
                    f'{study_name.replace("_", " ").title()}',
                    os.path.join(study_dir, 'radar.pdf'))

        # Delta chart (first experiment as baseline)
        exp_names = list(results.keys())
        if len(exp_names) >= 2:
            baseline = results[exp_names[0]].get('best_metrics', {})
            for other_name in exp_names[1:]:
                other = results[other_name].get('best_metrics', {})
                safe_name = other_name.replace(' ', '_').replace('(', '').replace(')', '').lower()
                delta_bar_chart(baseline, other, exp_names[0], other_name,
                                gnn_metrics,
                                f'Improvement: {other_name}',
                                os.path.join(study_dir, f'delta_{safe_name}.pdf'))

        # Lambda sensitivity (if applicable)
        if 'lambda' in study_name:
            sensitivity_line_plot(results, 'lambda_policy', key_metrics,
                                  'Policy Loss Weight Sensitivity',
                                  os.path.join(study_dir, 'sensitivity.pdf'))

    # -- LaTeX tables --
    print('\nGenerating LaTeX tables...')
    tables_dir = os.path.join(output_dir, 'tables')
    os.makedirs(tables_dir, exist_ok=True)

    for study_name, results in all_results.items():
        table = latex_table(results, key_metrics,
                           f'{study_name.replace("_", " ").title()} Results',
                           f'tab:{study_name}')
        table_path = os.path.join(tables_dir, f'{study_name}.tex')
        with open(table_path, 'w') as f:
            f.write(table)
        print(f'  Saved: {table_path}')

    # -- Combined table (all studies) --
    full_table_path = os.path.join(tables_dir, 'all_results.tex')
    with open(full_table_path, 'w') as f:
        for study_name, results in all_results.items():
            table = latex_table(results, key_metrics,
                               f'{study_name.replace("_", " ").title()} Results',
                               f'tab:{study_name}')
            f.write(table + '\n\n')
    print(f'  Saved: {full_table_path}')

    print(f'\nDone! All figures saved to {output_dir}/')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate thesis figures from ablation results')
    parser.add_argument('--results-dir', type=str, default='results/ablation',
                        help='Directory containing ablation result JSON files')
    parser.add_argument('--output-dir', type=str, default='figures/thesis',
                        help='Directory to save generated figures')
    parser.add_argument('--study', type=str, default=None,
                        help='Generate figures for a specific study only')
    args = parser.parse_args()

    if args.study:
        study_dir = os.path.join(args.results_dir, args.study)
        if os.path.isdir(study_dir):
            all_results = {args.study: load_results_single(study_dir)}
        else:
            print(f'Study directory not found: {study_dir}')
            sys.exit(1)
    else:
        generate_all_figures(args.results_dir, args.output_dir)
