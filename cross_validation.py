import argparse
import logging
import sys
import time

import numpy as np
from sklearn.model_selection import StratifiedKFold

from train import train, setup_logging, parse_args, select_best_metrics
from utils import seed_everything

logger = logging.getLogger(__name__)

"""
    K-fold cross-validation for CARE-GNN.
    Runs stratified k-fold CV and reports mean +/- std of all metrics.
"""


def run_cross_validation(args, num_folds=5):
    """
    Run stratified k-fold cross-validation.
    :param args: training arguments
    :param num_folds: number of folds
    :returns: dict of metric means and stds
    """
    seed_everything(args.seed)

    # disable validation split during CV (each fold has its own val/test)
    args.use_validation = False
    args.save_best = False

    all_fold_metrics = []

    for fold in range(num_folds):
        logger.info(f'\n{"="*60}')
        logger.info(f'Fold {fold + 1}/{num_folds}')
        logger.info(f'{"="*60}')

        # use different random state for each fold
        args.seed = args.seed + fold
        seed_everything(args.seed)

        try:
            start_time = time.time()
            model, performance_log = train(args)
            elapsed = time.time() - start_time

            # select the epoch by best validation score (NOT best test AUC)
            best_metrics = select_best_metrics(performance_log)
            if best_metrics:
                # remove non-numeric entries
                fold_metrics = {k: v for k, v in best_metrics.items()
                                if isinstance(v, (int, float))}
                all_fold_metrics.append(fold_metrics)
                logger.info(f'Fold {fold + 1} completed in {elapsed:.1f}s, '
                            f'GNN AUC: {fold_metrics.get("gnn_auc", "N/A")}')
            else:
                logger.warning(f'Fold {fold + 1} produced no metrics')

        except Exception as e:
            logger.error(f'Fold {fold + 1} failed: {e}')

    if not all_fold_metrics:
        logger.error('No successful folds')
        return {}

    # compute mean and std for each metric
    results = {}
    metric_keys = all_fold_metrics[0].keys()
    for key in metric_keys:
        values = [m[key] for m in all_fold_metrics if key in m]
        if values:
            results[f'{key}_mean'] = np.mean(values)
            # sample std (ddof=1) across folds; 0.0 when only one fold
            results[f'{key}_std'] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0

    # print summary
    logger.info(f'\n{"="*60}')
    logger.info(f'Cross-Validation Results ({num_folds} folds)')
    logger.info(f'{"="*60}')
    for key in metric_keys:
        mean_key = f'{key}_mean'
        std_key = f'{key}_std'
        if mean_key in results:
            logger.info(f'{key}: {results[mean_key]:.4f} +/- {results[std_key]:.4f}')

    return results


if __name__ == '__main__':
    setup_logging()

    parser = argparse.ArgumentParser(description='Run CARE-GNN cross-validation')
    parser.add_argument('--num-folds', type=int, default=5, help='Number of CV folds')

    cv_args, remaining = parser.parse_known_args()

    # parse training args from remaining
    sys.argv = ['train.py'] + remaining
    args = parse_args()

    results = run_cross_validation(args, num_folds=cv_args.num_folds)
