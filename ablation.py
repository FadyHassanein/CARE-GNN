import argparse
import json
import logging
import os
import sys
import time

from train import train, setup_logging, parse_args

logger = logging.getLogger(__name__)

"""
    Ablation study framework for CARE-GNN.
    Systematically evaluates different model configurations:
    - Inter-aggregator types (Att, Weight, Mean, GNN)
    - With/without RL module
    - Different loss functions (CE, Focal, Weighted CE)
    - Number of layers (1, 2, 3 for MULTI_CARE)
    - Individual relations
"""


def run_ablation(base_args, experiments, output_dir='results/ablation'):
    """
    Run a set of ablation experiments.
    :param base_args: base argument namespace
    :param experiments: list of dicts, each mapping arg names to values to override
    :param output_dir: directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)
    all_results = {}

    for i, exp in enumerate(experiments):
        exp_name = exp.pop('name', f'experiment_{i}')
        logger.info(f'\n{"="*60}')
        logger.info(f'Running experiment: {exp_name}')
        logger.info(f'{"="*60}')

        # create a copy of base args and override with experiment-specific args
        exp_args = argparse.Namespace(**vars(base_args))
        for key, value in exp.items():
            setattr(exp_args, key, value)

        logger.info(f'Config: {exp}')

        try:
            start_time = time.time()
            model, performance_log = train(exp_args)
            elapsed = time.time() - start_time

            if performance_log:
                best_metrics = max(performance_log, key=lambda m: m.get('gnn_auc', 0))
            else:
                best_metrics = {}

            all_results[exp_name] = {
                'config': exp,
                'best_metrics': {k: v.tolist() if hasattr(v, 'tolist') else v
                                 for k, v in best_metrics.items()
                                 if not k.endswith('confusion_matrix')},
                'elapsed_seconds': elapsed,
            }
            logger.info(f'Experiment {exp_name} completed in {elapsed:.1f}s')
            if best_metrics:
                logger.info(f'Best GNN AUC: {best_metrics.get("gnn_auc", "N/A")}')

        except Exception as e:
            logger.error(f'Experiment {exp_name} failed: {e}')
            all_results[exp_name] = {'error': str(e)}

        # restore the name for results
        exp['name'] = exp_name

    # save results
    results_file = os.path.join(output_dir, f'ablation_results_{time.strftime("%Y%m%d_%H%M%S")}.json')
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f'Results saved to {results_file}')

    return all_results


def get_inter_aggregator_experiments():
    """Compare different inter-relation aggregator types."""
    return [
        {'name': 'CARE-GNN (Threshold)', 'inter': 'GNN'},
        {'name': 'CARE-Weight', 'inter': 'Weight'},
        {'name': 'CARE-Mean', 'inter': 'Mean'},
        {'name': 'CARE-Att', 'inter': 'Att'},
    ]


def get_loss_function_experiments():
    """Compare different loss functions."""
    return [
        {'name': 'CrossEntropy', 'loss': 'ce'},
        {'name': 'FocalLoss', 'loss': 'focal'},
        {'name': 'WeightedCE', 'loss': 'weighted_ce'},
    ]


def get_multi_layer_experiments():
    """Compare different numbers of GNN layers."""
    return [
        {'name': '1-Layer CARE', 'model': 'CARE'},
        {'name': '2-Layer CARE', 'model': 'MULTI_CARE', 'num_layers': 2},
        {'name': '3-Layer CARE', 'model': 'MULTI_CARE', 'num_layers': 3},
    ]


def get_baseline_experiments():
    """Compare CARE-GNN against GraphSAGE baseline."""
    return [
        {'name': 'GraphSAGE', 'model': 'SAGE'},
        {'name': 'CARE-GNN', 'model': 'CARE', 'inter': 'GNN'},
    ]


if __name__ == '__main__':
    setup_logging()

    parser = argparse.ArgumentParser(description='Run CARE-GNN ablation studies')
    parser.add_argument('--study', type=str, required=True,
                        choices=['inter', 'loss', 'layers', 'baseline', 'all'],
                        help='Which ablation study to run')
    parser.add_argument('--data', type=str, default='yelp', help='Dataset')
    parser.add_argument('--num-epochs', type=int, default=31, help='Epochs per experiment')
    parser.add_argument('--output-dir', type=str, default='results/ablation', help='Output directory')

    ab_args = parser.parse_args()

    # get base args for training
    sys.argv = ['train.py', '--data', ab_args.data, '--num-epochs', str(ab_args.num_epochs)]
    base_args = parse_args()

    # select experiments
    studies = {
        'inter': get_inter_aggregator_experiments,
        'loss': get_loss_function_experiments,
        'layers': get_multi_layer_experiments,
        'baseline': get_baseline_experiments,
    }

    if ab_args.study == 'all':
        for study_name, study_fn in studies.items():
            logger.info(f'\n\nStarting study: {study_name}')
            experiments = study_fn()
            run_ablation(base_args, experiments, os.path.join(ab_args.output_dir, study_name))
    else:
        experiments = studies[ab_args.study]()
        run_ablation(base_args, experiments, ab_args.output_dir)
