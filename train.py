import time
import os
import random
import logging
import argparse

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

from utils import (load_data, normalize, pos_neg_split, undersample,
                   test_sage, test_care, seed_everything, get_device,
                   save_checkpoint, EarlyStopping)
from model import OneLayerCARE, MultiLayerCARE, CAPNOneLayerCARE
from layers import InterAgg, IntraAgg
from graphsage import GraphSage, MeanAggregator, Encoder
from capn import (EnhancedLabelPredictor, StateConstructor, PolicyNetwork,
                  ShapedRewardComputer, LLMPriorLoader, MultiViewDistance)
from config import CareConfig

"""
    Training CARE-GNN
    Paper: Enhancing Graph Neural Network-based Fraud Detectors against Camouflaged Fraudsters
    Source: https://github.com/YingtongDou/CARE-GNN
"""

logger = logging.getLogger(__name__)


def setup_logging(log_dir='logs', level=logging.INFO):
    """Configure logging to both file and console."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'train_{time.strftime("%Y%m%d_%H%M%S")}.log')

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # file handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(level)
    fh.setFormatter(formatter)

    # console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(fh)
    root_logger.addHandler(ch)

    return log_file


def parse_args():
    parser = argparse.ArgumentParser()

    # dataset and model dependent args
    parser.add_argument('--data', type=str, default='yelp', help='The dataset name. [yelp, amazon]')
    parser.add_argument('--model', type=str, default='CARE', help='The model name. [CARE, SAGE, MULTI_CARE]')
    parser.add_argument('--inter', type=str, default='GNN', help='The inter-relation aggregator type. [Att, Weight, Mean, GNN]')
    parser.add_argument('--batch-size', type=int, default=1024, help='Batch size 1024 for yelp, 256 for amazon.')

    # hyper-parameters
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
    parser.add_argument('--lambda_1', type=float, default=2, help='Simi loss weight.')
    parser.add_argument('--lambda_2', type=float, default=1e-3, help='Weight decay (L2 loss weight).')
    parser.add_argument('--emb-size', type=int, default=64, help='Node embedding size at the last layer.')
    parser.add_argument('--num-epochs', type=int, default=31, help='Number of epochs.')
    parser.add_argument('--test-epochs', type=int, default=3, help='Epoch interval to run test set.')
    parser.add_argument('--under-sample', type=int, default=1, help='Under-sampling scale.')
    parser.add_argument('--step-size', type=float, default=2e-2, help='RL action step size')
    parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate.')

    # device and reproducibility
    parser.add_argument('--device', type=str, default='auto', help='Device: auto, cpu, cuda, cuda:0, etc.')
    parser.add_argument('--seed', type=int, default=72, help='Random seed.')

    # training improvements
    parser.add_argument('--grad-clip', type=float, default=1.0, help='Gradient clipping max norm.')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience.')
    parser.add_argument('--use-lr-scheduler', action='store_true', default=True, help='Use learning rate scheduler.')
    parser.add_argument('--lr-scheduler-patience', type=int, default=5, help='LR scheduler patience.')
    parser.add_argument('--lr-scheduler-factor', type=float, default=0.5, help='LR scheduler reduction factor.')

    # validation
    parser.add_argument('--val-size', type=float, default=0.15, help='Validation set size (fraction of total).')
    parser.add_argument('--use-validation', action='store_true', default=True, help='Use validation set.')

    # checkpointing
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='Directory for saving checkpoints.')
    parser.add_argument('--save-best', action='store_true', default=True, help='Save best model checkpoint.')

    # multi-layer
    parser.add_argument('--num-layers', type=int, default=1, help='Number of GNN layers (for MULTI_CARE).')

    # loss function
    parser.add_argument('--loss', type=str, default='ce', help='Loss function: ce, focal, weighted_ce')

    # GraphSAGE
    parser.add_argument('--sage-num-samples', type=int, default=5, help='Number of neighbor samples for GraphSAGE.')

    # RL terminal condition
    parser.add_argument('--rl-patience', type=int, default=5, help='RL convergence patience.')
    parser.add_argument('--rl-epsilon', type=float, default=1e-4, help='RL convergence epsilon.')

    # CAPN: Camouflage-Aware Policy Network
    parser.add_argument('--use-capn', action='store_true', default=False, help='Use CAPN policy network for adaptive thresholds.')
    parser.add_argument('--policy-lr', type=float, default=1e-3, help='Policy network learning rate.')
    parser.add_argument('--policy-hidden', type=int, default=64, help='Policy network hidden dimension.')
    parser.add_argument('--lambda-policy', type=float, default=0.1, help='Weight for policy gradient loss.')
    parser.add_argument('--reward-w1', type=float, default=0.5, help='Shaped reward: distance improvement weight.')
    parser.add_argument('--reward-w2', type=float, default=0.3, help='Shaped reward: accuracy signal weight.')
    parser.add_argument('--reward-w3', type=float, default=0.2, help='Shaped reward: regularization weight.')
    parser.add_argument('--gamma-init', type=float, default=0.7, help='Initial gamma for multi-view distance.')
    parser.add_argument('--llm-priors-file', type=str, default='', help='Path to LLM-generated priors JSON file.')

    return parser.parse_args()


def get_loss_fn(loss_type, labels=None):
    """Get loss function by name."""
    if loss_type == 'ce':
        return nn.CrossEntropyLoss()
    elif loss_type == 'focal':
        from losses import FocalLoss
        return FocalLoss(gamma=2.0)
    elif loss_type == 'weighted_ce':
        from losses import WeightedCrossEntropy
        return WeightedCrossEntropy(labels=labels)
    else:
        raise ValueError(f'Unknown loss function: {loss_type}')


def train(args):
    """Main training function."""
    device = get_device(args.device)
    logger.info(f'Using device: {device}')

    # set seeds for reproducibility
    seed_everything(args.seed)

    # load graph, feature, and label
    [homo, relation1, relation2, relation3], feat_data, labels = load_data(args.data)

    # train/val/test split
    if args.data == 'yelp':
        index = list(range(len(labels)))
        all_labels = labels
    elif args.data == 'amazon':
        # 0-3304 are unlabeled nodes
        index = list(range(3305, len(labels)))
        all_labels = labels[3305:]

    if args.use_validation:
        # first split: train vs (val + test)
        idx_train, idx_temp, y_train, y_temp = train_test_split(
            index, all_labels, stratify=all_labels,
            test_size=args.val_size + 0.60, random_state=2, shuffle=True)
        # second split: val vs test
        val_fraction = args.val_size / (args.val_size + 0.60)
        idx_val, idx_test, y_val, y_test = train_test_split(
            idx_temp, y_temp, stratify=y_temp,
            test_size=1 - val_fraction, random_state=2, shuffle=True)
        logger.info(f'Split: train={len(idx_train)}, val={len(idx_val)}, test={len(idx_test)}')
    else:
        idx_train, idx_test, y_train, y_test = train_test_split(
            index, all_labels, stratify=all_labels,
            test_size=0.60, random_state=2, shuffle=True)
        idx_val, y_val = None, None
        logger.info(f'Split: train={len(idx_train)}, test={len(idx_test)}')

    # split pos neg sets for under-sampling
    train_pos, train_neg = pos_neg_split(idx_train, y_train)

    # get loss function
    loss_fn = get_loss_fn(args.loss, labels=y_train)

    # initialize model input
    features = nn.Embedding(feat_data.shape[0], feat_data.shape[1])
    feat_data = normalize(feat_data)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
    features = features.to(device)

    # set input graph
    if args.model == 'SAGE':
        adj_lists = homo
    else:
        adj_lists = [relation1, relation2, relation3]

    logger.info(f'Model: {args.model}, Inter-AGG: {args.inter}, emb_size: {args.emb_size}')

    # CAPN components (initialized if --use-capn is set)
    policy_network = None
    state_constructor = None
    enhanced_label_clf = None
    multi_view_distance = None
    reward_computer = None

    if args.use_capn and args.model in ('CARE',):
        # load LLM priors if available
        llm_loader = LLMPriorLoader(args.llm_priors_file if args.llm_priors_file else None)
        relation_biases = llm_loader.get_relation_biases()
        gamma_inits = llm_loader.get_gamma_init()

        # enhanced label predictor
        enhanced_label_clf = EnhancedLabelPredictor(feat_data.shape[1]).to(device)

        # state constructor
        state_constructor = StateConstructor(adj_lists, homo, features, device=device)

        # policy network
        state_dim = feat_data.shape[1] + 5
        policy_network = PolicyNetwork(
            state_dim, hidden_dim=args.policy_hidden,
            num_relations=len(adj_lists),
            relation_biases=relation_biases).to(device)

        # multi-view distance
        gamma_init = gamma_inits if gamma_inits else args.gamma_init
        multi_view_distance = MultiViewDistance(len(adj_lists), gamma_init=gamma_init, adj_lists=adj_lists)

        # shaped reward computer
        reward_computer = ShapedRewardComputer(w1=args.reward_w1, w2=args.reward_w2, w3=args.reward_w3)

        logger.info(f'CAPN mode enabled: policy_hidden={args.policy_hidden}, policy_lr={args.policy_lr}, '
                    f'lambda_policy={args.lambda_policy}')

    # build models
    if args.model == 'CARE':
        intra_aggs = [IntraAgg(features, feat_data.shape[1], device=device) for _ in range(len(adj_lists))]
        inter1 = InterAgg(features, feat_data.shape[1], args.emb_size, adj_lists, intra_aggs,
                          inter=args.inter, step_size=args.step_size, device=device,
                          dropout=args.dropout, rl_patience=args.rl_patience, rl_epsilon=args.rl_epsilon,
                          policy_network=policy_network, state_constructor=state_constructor,
                          enhanced_label_clf=enhanced_label_clf, multi_view_distance=multi_view_distance)
        if args.use_capn:
            gnn_model = CAPNOneLayerCARE(2, inter1, args.lambda_1,
                                          lambda_policy=args.lambda_policy,
                                          reward_computer=reward_computer, loss_fn=loss_fn)
        else:
            gnn_model = OneLayerCARE(2, inter1, args.lambda_1, loss_fn=loss_fn)

    elif args.model == 'MULTI_CARE':
        inter_layers = []
        for layer_idx in range(args.num_layers):
            feat_dim = feat_data.shape[1] if layer_idx == 0 else args.emb_size
            intra_aggs = [IntraAgg(features, feat_dim, device=device) for _ in range(len(adj_lists))]
            inter_layer = InterAgg(features, feat_dim, args.emb_size, adj_lists, intra_aggs,
                                   inter=args.inter, step_size=args.step_size, device=device,
                                   dropout=args.dropout, rl_patience=args.rl_patience, rl_epsilon=args.rl_epsilon)
            inter_layers.append(inter_layer)
        gnn_model = MultiLayerCARE(2, inter_layers, args.lambda_1, loss_fn=loss_fn, dropout=args.dropout)

    elif args.model == 'SAGE':
        agg1 = MeanAggregator(features, device=device)
        enc1 = Encoder(features, feat_data.shape[1], args.emb_size, adj_lists, agg1, gcn=True, device=device)
        enc1.num_samples = args.sage_num_samples
        gnn_model = GraphSage(2, enc1)

    gnn_model = gnn_model.to(device)

    # set up optimizers
    if args.use_capn and policy_network is not None:
        # dual optimizer: separate LR for policy network
        gnn_params = [p for n, p in gnn_model.named_parameters()
                      if p.requires_grad and 'policy_network' not in n]
        policy_params = list(policy_network.parameters())
        optimizer = torch.optim.Adam(gnn_params, lr=args.lr, weight_decay=args.lambda_2)
        policy_optimizer = torch.optim.Adam(policy_params, lr=args.policy_lr)
        logger.info(f'Dual optimizer: GNN params={sum(p.numel() for p in gnn_params)}, '
                    f'Policy params={sum(p.numel() for p in policy_params)}')
    else:
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, gnn_model.parameters()),
            lr=args.lr, weight_decay=args.lambda_2)
        policy_optimizer = None

    # learning rate scheduler
    scheduler = None
    if args.use_lr_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', patience=args.lr_scheduler_patience,
            factor=args.lr_scheduler_factor, verbose=True)

    # early stopping
    early_stopping = EarlyStopping(patience=args.patience)

    performance_log = []
    best_val_auc = 0.0

    # train the model
    for epoch in range(args.num_epochs):
        gnn_model.train()

        # randomly under-sampling negative nodes for each epoch
        sampled_idx_train = undersample(train_pos, train_neg, scale=args.under_sample)
        random.shuffle(sampled_idx_train)

        # send number of batches to model to let the RLModule know the training progress
        num_batches = int(len(sampled_idx_train) / args.batch_size) + 1
        if args.model in ('CARE', 'MULTI_CARE'):
            if args.model == 'CARE':
                gnn_model.inter1.batch_num = num_batches
            elif args.model == 'MULTI_CARE':
                for inter in gnn_model.inter_layers:
                    inter.batch_num = num_batches

        epoch_loss = 0.0
        epoch_time = 0

        # mini-batch training
        for batch in range(num_batches):
            start_time = time.time()
            i_start = batch * args.batch_size
            i_end = min((batch + 1) * args.batch_size, len(sampled_idx_train))
            batch_nodes = sampled_idx_train[i_start:i_end]
            batch_label = labels[np.array(batch_nodes)]
            optimizer.zero_grad()
            if policy_optimizer is not None:
                policy_optimizer.zero_grad()
            loss = gnn_model.loss(batch_nodes, torch.LongTensor(batch_label).to(device))
            loss.backward()

            # gradient clipping
            torch.nn.utils.clip_grad_norm_(gnn_model.parameters(), args.grad_clip)

            optimizer.step()
            if policy_optimizer is not None:
                policy_optimizer.step()
            end_time = time.time()
            epoch_time += end_time - start_time
            epoch_loss += loss.item()

        logger.info(f'Epoch: {epoch}, loss: {epoch_loss / num_batches:.4f}, '
                    f'lr: {optimizer.param_groups[0]["lr"]:.6f}, time: {epoch_time:.2f}s')

        # evaluation
        if epoch % args.test_epochs == 0:
            gnn_model.eval()
            with torch.no_grad():
                if args.model == 'SAGE':
                    metrics = test_sage(idx_test, y_test, gnn_model, args.batch_size, device=device)
                    val_metric = metrics['auc']
                else:
                    metrics = test_care(idx_test, y_test, gnn_model, args.batch_size, device=device)
                    val_metric = metrics['gnn_auc']
                    performance_log.append(metrics)

                # validation set evaluation
                if args.use_validation and idx_val is not None:
                    if args.model == 'SAGE':
                        val_metrics = test_sage(idx_val, y_val, gnn_model, args.batch_size, device=device)
                        val_metric = val_metrics['auc']
                    else:
                        val_metrics = test_care(idx_val, y_val, gnn_model, args.batch_size, device=device)
                        val_metric = val_metrics['gnn_auc']
                    logger.info(f'Validation AUC: {val_metric:.4f}')

                # learning rate scheduling
                if scheduler is not None:
                    scheduler.step(val_metric)

                # save best model
                if args.save_best and val_metric > best_val_auc:
                    best_val_auc = val_metric
                    save_checkpoint(gnn_model, optimizer, epoch, metrics,
                                    os.path.join(args.checkpoint_dir, f'best_{args.model}_{args.data}.pt'))
                    logger.info(f'New best model saved (AUC: {val_metric:.4f})')

                # early stopping
                if early_stopping.step(val_metric):
                    logger.info(f'Early stopping at epoch {epoch}')
                    break

    logger.info('Training complete.')
    return gnn_model, performance_log


if __name__ == '__main__':
    args = parse_args()
    log_file = setup_logging()
    logger.info(f'Logging to {log_file}')
    logger.info(f'Arguments: {vars(args)}')
    train(args)
