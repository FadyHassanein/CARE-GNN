import os
import pickle
import logging
import random as rd
import numpy as np
import scipy.sparse as sp
from scipy.io import loadmat
from sklearn.metrics import (f1_score, accuracy_score, recall_score,
                             roc_auc_score, average_precision_score,
                             precision_score, confusion_matrix)
from collections import defaultdict

import torch

logger = logging.getLogger(__name__)

"""
    Utility functions to handle data and evaluate model.
"""


def seed_everything(seed):
    """Set all random seeds for reproducibility."""
    rd.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f'Random seed set to {seed}')


def get_device(device_str='auto'):
    """Get torch device from string specification."""
    if device_str == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_str)


def load_data(data, prefix='data/'):
    """
    Load graph, feature, and label given dataset name
    :returns: home and single-relation graphs, feature, label
    """
    if data == 'yelp':
        data_file = loadmat(prefix + 'YelpChi.mat')
        labels = data_file['label'].flatten()
        feat_data = data_file['features'].todense().A
        with open(prefix + 'yelp_homo_adjlists.pickle', 'rb') as file:
            homo = pickle.load(file)
        with open(prefix + 'yelp_rur_adjlists.pickle', 'rb') as file:
            relation1 = pickle.load(file)
        with open(prefix + 'yelp_rtr_adjlists.pickle', 'rb') as file:
            relation2 = pickle.load(file)
        with open(prefix + 'yelp_rsr_adjlists.pickle', 'rb') as file:
            relation3 = pickle.load(file)
    elif data == 'amazon':
        data_file = loadmat(prefix + 'Amazon.mat')
        labels = data_file['label'].flatten()
        feat_data = data_file['features'].todense().A
        with open(prefix + 'amz_homo_adjlists.pickle', 'rb') as file:
            homo = pickle.load(file)
        with open(prefix + 'amz_upu_adjlists.pickle', 'rb') as file:
            relation1 = pickle.load(file)
        with open(prefix + 'amz_usu_adjlists.pickle', 'rb') as file:
            relation2 = pickle.load(file)
        with open(prefix + 'amz_uvu_adjlists.pickle', 'rb') as file:
            relation3 = pickle.load(file)
    else:
        raise ValueError(f'Unknown dataset: {data}. Supported: yelp, amazon')

    logger.info(f'Loaded {data} dataset: {len(labels)} nodes, {feat_data.shape[1]} features')
    return [homo, relation1, relation2, relation3], feat_data, labels


def normalize(mx):
    """
    Row-normalize sparse matrix
    Code from https://github.com/williamleif/graphsage-simple/
    """
    rowsum = np.array(mx.sum(1)) + 0.01
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_to_adjlist(sp_matrix, filename):
    """
    Transfer sparse matrix to adjacency list
    :param sp_matrix: the sparse matrix
    :param filename: the filename of adjlist
    """
    # add self loop
    homo_adj = sp_matrix + sp.eye(sp_matrix.shape[0])
    # create adj_list
    adj_lists = defaultdict(set)
    edges = homo_adj.nonzero()
    for index, node in enumerate(edges[0]):
        adj_lists[node].add(edges[1][index])
        adj_lists[edges[1][index]].add(node)
    with open(filename, 'wb') as file:
        pickle.dump(adj_lists, file)


def pos_neg_split(nodes, labels):
    """
    Find positive and negative nodes given a list of nodes and their labels.
    Uses single-pass approach instead of O(n^2) list.remove().
    """
    pos_nodes = []
    neg_nodes = []
    for node, label in zip(nodes, labels):
        if label == 1:
            pos_nodes.append(node)
        else:
            neg_nodes.append(node)
    return pos_nodes, neg_nodes


def undersample(pos_nodes, neg_nodes, scale=1):
    """
    Under-sample the negative nodes
    :param pos_nodes: a list of positive nodes
    :param neg_nodes: a list negative nodes
    :param scale: the under-sampling scale
    :return: a list of under-sampled batch nodes
    """
    k = min(int(len(pos_nodes) * scale), len(neg_nodes))
    sampled_neg = rd.sample(neg_nodes, k=k)
    batch_nodes = pos_nodes + sampled_neg
    return batch_nodes


def test_sage(test_cases, labels, model, batch_size, device=None):
    """
    Test the performance of GraphSAGE
    :returns: dictionary of all metrics
    """
    test_batch_num = int(len(test_cases) / batch_size) + 1
    f1_gnn = 0.0
    acc_gnn = 0.0
    recall_gnn = 0.0
    precision_gnn = 0.0
    gnn_list = []

    for iteration in range(test_batch_num):
        i_start = iteration * batch_size
        i_end = min((iteration + 1) * batch_size, len(test_cases))
        batch_nodes = test_cases[i_start:i_end]
        batch_label = labels[i_start:i_end]
        gnn_prob = model.to_prob(batch_nodes)
        preds = gnn_prob.data.cpu().numpy().argmax(axis=1)
        f1_gnn += f1_score(batch_label, preds, average="macro")
        acc_gnn += accuracy_score(batch_label, preds)
        recall_gnn += recall_score(batch_label, preds, average="macro")
        precision_gnn += precision_score(batch_label, preds, average="macro", zero_division=0)
        gnn_list.extend(gnn_prob.data.cpu().numpy()[:, 1].tolist())

    auc_gnn = roc_auc_score(labels, np.array(gnn_list))
    ap_gnn = average_precision_score(labels, np.array(gnn_list))

    metrics = {
        'f1': f1_gnn / test_batch_num,
        'accuracy': acc_gnn / test_batch_num,
        'recall': recall_gnn / test_batch_num,
        'precision': precision_gnn / test_batch_num,
        'auc': auc_gnn,
        'ap': ap_gnn,
    }

    logger.info(f"GNN F1: {metrics['f1']:.4f}, Acc: {metrics['accuracy']:.4f}, "
                f"Recall: {metrics['recall']:.4f}, AUC: {metrics['auc']:.4f}, AP: {metrics['ap']:.4f}")
    return metrics


def test_care(test_cases, labels, model, batch_size, device=None):
    """
    Test the performance of CARE-GNN and its variants
    :returns: dictionary of all metrics for both GNN and label modules
    """
    test_batch_num = int(len(test_cases) / batch_size) + 1
    f1_gnn = 0.0
    acc_gnn = 0.0
    recall_gnn = 0.0
    precision_gnn = 0.0
    f1_label = 0.0
    acc_label = 0.0
    recall_label = 0.0
    precision_label = 0.0
    gnn_list = []
    label_list = []
    all_preds_gnn = []
    all_preds_label = []
    all_labels = []

    for iteration in range(test_batch_num):
        i_start = iteration * batch_size
        i_end = min((iteration + 1) * batch_size, len(test_cases))
        batch_nodes = test_cases[i_start:i_end]
        batch_label = labels[i_start:i_end]
        gnn_prob, label_prob = model.to_prob(batch_nodes, batch_label, train_flag=False)

        gnn_preds = gnn_prob.data.cpu().numpy().argmax(axis=1)
        label_preds = label_prob.data.cpu().numpy().argmax(axis=1)

        f1_gnn += f1_score(batch_label, gnn_preds, average="macro")
        acc_gnn += accuracy_score(batch_label, gnn_preds)
        recall_gnn += recall_score(batch_label, gnn_preds, average="macro")
        precision_gnn += precision_score(batch_label, gnn_preds, average="macro", zero_division=0)

        f1_label += f1_score(batch_label, label_preds, average="macro")
        acc_label += accuracy_score(batch_label, label_preds)
        recall_label += recall_score(batch_label, label_preds, average="macro")
        precision_label += precision_score(batch_label, label_preds, average="macro", zero_division=0)

        gnn_list.extend(gnn_prob.data.cpu().numpy()[:, 1].tolist())
        label_list.extend(label_prob.data.cpu().numpy()[:, 1].tolist())
        all_preds_gnn.extend(gnn_preds.tolist())
        all_preds_label.extend(label_preds.tolist())
        all_labels.extend(batch_label.tolist() if hasattr(batch_label, 'tolist') else list(batch_label))

    auc_gnn = roc_auc_score(labels, np.array(gnn_list))
    ap_gnn = average_precision_score(labels, np.array(gnn_list))
    auc_label = roc_auc_score(labels, np.array(label_list))
    ap_label = average_precision_score(labels, np.array(label_list))

    metrics = {
        'gnn_f1': f1_gnn / test_batch_num,
        'gnn_accuracy': acc_gnn / test_batch_num,
        'gnn_recall': recall_gnn / test_batch_num,
        'gnn_precision': precision_gnn / test_batch_num,
        'gnn_auc': auc_gnn,
        'gnn_ap': ap_gnn,
        'gnn_confusion_matrix': confusion_matrix(all_labels, all_preds_gnn),
        'label_f1': f1_label / test_batch_num,
        'label_accuracy': acc_label / test_batch_num,
        'label_recall': recall_label / test_batch_num,
        'label_precision': precision_label / test_batch_num,
        'label_auc': auc_label,
        'label_ap': ap_label,
        'label_confusion_matrix': confusion_matrix(all_labels, all_preds_label),
    }

    logger.info(f"GNN F1: {metrics['gnn_f1']:.4f}, Acc: {metrics['gnn_accuracy']:.4f}, "
                f"Recall: {metrics['gnn_recall']:.4f}, AUC: {metrics['gnn_auc']:.4f}, AP: {metrics['gnn_ap']:.4f}")
    logger.info(f"Label F1: {metrics['label_f1']:.4f}, Acc: {metrics['label_accuracy']:.4f}, "
                f"Recall: {metrics['label_recall']:.4f}, AUC: {metrics['label_auc']:.4f}, AP: {metrics['label_ap']:.4f}")

    return metrics


def save_checkpoint(model, optimizer, epoch, metrics, filepath):
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }
    torch.save(checkpoint, filepath)
    logger.info(f'Checkpoint saved to {filepath}')


def load_checkpoint(filepath, model, optimizer=None):
    """Load model checkpoint."""
    checkpoint = torch.load(filepath, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    logger.info(f'Checkpoint loaded from {filepath} (epoch {checkpoint["epoch"]})')
    return checkpoint['epoch'], checkpoint['metrics']


class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def step(self, score):
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(f'Early stopping triggered after {self.counter} epochs without improvement')
        else:
            self.best_score = score
            self.counter = 0
        return self.should_stop
