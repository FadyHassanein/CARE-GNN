import argparse
import inspect
import json
import logging
import os
import sys
import time
import traceback

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
            tb = traceback.format_exc()
            logger.error(f'Experiment {exp_name} failed: {e}\n{tb}')
            all_results[exp_name] = {'error': str(e) or repr(e), 'traceback': tb}

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


# ===================== CAPN Ablation Experiments =====================

def get_capn_full_experiments():
    """A1: CAPN vs original RL — the main comparison."""
    return [
        {'name': 'CARE-GNN (original RL)', 'model': 'CARE', 'use_capn': False},
        {'name': 'CAPN (policy network)', 'model': 'CARE', 'use_capn': True},
    ]


def get_capn_label_predictor_experiments():
    """A2: Isolate label predictor effect from policy network."""
    return [
        {'name': 'Linear predictor (CARE-GNN)', 'model': 'CARE', 'use_capn': False, 'use_mlp_label': False},
        {'name': 'MLP predictor (no policy)', 'model': 'CARE', 'use_capn': False, 'use_mlp_label': True},
        {'name': 'MLP predictor + policy (CAPN)', 'model': 'CARE', 'use_capn': True},
    ]


def get_capn_reward_experiments():
    """A5: Shaped vs binary reward."""
    return [
        {'name': 'Binary reward (original)', 'model': 'CARE', 'use_capn': False},
        {'name': 'Shaped reward (CAPN)', 'model': 'CARE', 'use_capn': True},
        {'name': 'Shaped (dist only)', 'model': 'CARE', 'use_capn': True,
         'reward_w1': 1.0, 'reward_w2': 0.0, 'reward_w3': 0.0},
        {'name': 'Shaped (acc only)', 'model': 'CARE', 'use_capn': True,
         'reward_w1': 0.0, 'reward_w2': 1.0, 'reward_w3': 0.0},
    ]


def get_capn_llm_prior_experiments(dataset='yelp'):
    """A6: With vs without LLM priors."""
    return [
        {'name': 'CAPN (no priors)', 'model': 'CARE', 'use_capn': True, 'llm_priors_file': ''},
        {'name': 'CAPN (LLM priors)', 'model': 'CARE', 'use_capn': True,
         'llm_priors_file': f'data/llm_priors/{dataset}_priors.json'},
    ]


def get_capn_lambda_experiments():
    """Sensitivity analysis for policy loss weight."""
    return [
        {'name': 'lambda_policy=0.01', 'model': 'CARE', 'use_capn': True, 'lambda_policy': 0.01},
        {'name': 'lambda_policy=0.05', 'model': 'CARE', 'use_capn': True, 'lambda_policy': 0.05},
        {'name': 'lambda_policy=0.1', 'model': 'CARE', 'use_capn': True, 'lambda_policy': 0.1},
        {'name': 'lambda_policy=0.5', 'model': 'CARE', 'use_capn': True, 'lambda_policy': 0.5},
    ]


def get_capn_llm_state_experiments():
    """A7: LLM semantic state enrichment — with vs without LLM embeddings in policy state."""
    return [
        {'name': 'CAPN (base state)', 'model': 'CARE', 'use_capn': True, 'use_llm_state': False},
        {'name': 'CAPN (LLM state)', 'model': 'CARE', 'use_capn': True, 'use_llm_state': True},
    ]


def run_studies(base_args, study_names, studies, output_dir, dataset='yelp'):
    """Run a list of studies with given base args."""
    for study_name in study_names:
        logger.info(f'\n\nStarting study: {study_name}')
        gen_fn = studies[study_name]
        # pass dataset to generators that need it
        if 'dataset' in inspect.signature(gen_fn).parameters:
            experiments = gen_fn(dataset=dataset)
        else:
            experiments = gen_fn()
        run_ablation(base_args, experiments, os.path.join(output_dir, study_name))


if __name__ == '__main__':
    setup_logging()

    parser = argparse.ArgumentParser(description='Run CARE-GNN ablation studies')
    parser.add_argument('--study', type=str, required=True,
                        choices=['inter', 'loss', 'layers', 'baseline',
                                 'capn', 'capn_label', 'capn_reward', 'capn_llm', 'capn_lambda',
                                 'capn_llm_state',
                                 'all', 'all_capn'],
                        help='Which ablation study to run')
    parser.add_argument('--data', type=str, default='yelp',
                        help='Dataset: yelp, amazon, or both')
    parser.add_argument('--num-epochs', type=int, default=31, help='Epochs per experiment')
    parser.add_argument('--output-dir', type=str, default='results/ablation', help='Output directory')

    ab_args = parser.parse_args()

    studies = {
        'inter': get_inter_aggregator_experiments,
        'loss': get_loss_function_experiments,
        'layers': get_multi_layer_experiments,
        'baseline': get_baseline_experiments,
        'capn': get_capn_full_experiments,
        'capn_label': get_capn_label_predictor_experiments,
        'capn_reward': get_capn_reward_experiments,
        'capn_llm': get_capn_llm_prior_experiments,
        'capn_lambda': get_capn_lambda_experiments,
        'capn_llm_state': get_capn_llm_state_experiments,
    }

    capn_studies = ['capn', 'capn_label', 'capn_reward', 'capn_llm', 'capn_lambda', 'capn_llm_state']

    # determine which studies to run
    if ab_args.study == 'all':
        study_list = list(studies.keys())
    elif ab_args.study == 'all_capn':
        study_list = capn_studies
    else:
        study_list = [ab_args.study]

    # determine which datasets to run on
    if ab_args.data == 'both':
        datasets = ['amazon', 'yelp']
    else:
        datasets = [ab_args.data]

    for dataset in datasets:
        logger.info(f'\n{"#"*60}')
        logger.info(f'Dataset: {dataset}')
        logger.info(f'{"#"*60}')

        sys.argv = ['train.py', '--data', dataset, '--num-epochs', str(ab_args.num_epochs)]
        if dataset == 'amazon':
            sys.argv += ['--batch-size', '256']
        base_args = parse_args()

        output_dir = os.path.join(ab_args.output_dir, dataset) if len(datasets) > 1 else ab_args.output_dir
        run_studies(base_args, study_list, studies, output_dir, dataset=dataset)
