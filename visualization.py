import os
import numpy as np
import logging

logger = logging.getLogger(__name__)

"""
    Visualization toolkit for CARE-GNN thesis figures.
    Generates publication-ready plots for threshold convergence,
    training curves, ROC curves, confusion matrices, and t-SNE embeddings.
"""

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    from sklearn.metrics import roc_curve, auc
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.warning('matplotlib not installed. Visualization functions will not work.')


def _check_matplotlib():
    if not HAS_MATPLOTLIB:
        raise ImportError('matplotlib is required for visualization. Install with: pip install matplotlib')


def plot_threshold_convergence(thresholds_log, relation_names=None, save_path='figures/threshold_convergence.pdf'):
    """
    Plot threshold convergence curves over training batches.
    :param thresholds_log: list of threshold lists from InterAgg.thresholds_log
    :param relation_names: optional list of relation names
    :param save_path: path to save the figure
    """
    _check_matplotlib()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    thresholds_array = np.array(thresholds_log)
    num_relations = thresholds_array.shape[1]

    if relation_names is None:
        relation_names = [f'Relation {i+1}' for i in range(num_relations)]

    fig, ax = plt.subplots(figsize=(8, 5))
    for r in range(num_relations):
        ax.plot(thresholds_array[:, r], label=relation_names[r], linewidth=2)

    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Threshold', fontsize=12)
    ax.set_title('Neighbor Filtering Threshold Convergence', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f'Threshold convergence plot saved to {save_path}')


def plot_training_curves(metrics_log, save_path='figures/training_curves.pdf'):
    """
    Plot training curves (AUC, F1, loss) over epochs.
    :param metrics_log: list of metric dictionaries from test_care()
    :param save_path: path to save the figure
    """
    _check_matplotlib()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    epochs = range(len(metrics_log))
    metric_keys = ['gnn_auc', 'gnn_f1', 'gnn_recall', 'gnn_ap']
    metric_labels = ['AUC-ROC', 'F1 (macro)', 'Recall (macro)', 'AP']

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, (key, label) in enumerate(zip(metric_keys, metric_labels)):
        values = [m[key] for m in metrics_log if key in m]
        if values:
            axes[i].plot(epochs[:len(values)], values, 'b-o', linewidth=2, markersize=4)
            axes[i].set_xlabel('Evaluation Step', fontsize=11)
            axes[i].set_ylabel(label, fontsize=11)
            axes[i].set_title(label, fontsize=13)
            axes[i].grid(True, alpha=0.3)

    fig.suptitle('CARE-GNN Training Progress', fontsize=15)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f'Training curves plot saved to {save_path}')


def plot_roc_curve(labels, scores, title='ROC Curve', save_path='figures/roc_curve.pdf'):
    """
    Plot ROC curve with AUC.
    :param labels: true labels
    :param scores: predicted probabilities for the positive class
    :param title: plot title
    :param save_path: path to save the figure
    """
    _check_matplotlib()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.4f}')
    ax.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='lower right', fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f'ROC curve saved to {save_path}')


def plot_confusion_matrix(cm, class_names=None, title='Confusion Matrix',
                          save_path='figures/confusion_matrix.pdf'):
    """
    Plot confusion matrix heatmap.
    :param cm: confusion matrix (numpy array)
    :param class_names: list of class names
    :param title: plot title
    :param save_path: path to save the figure
    """
    _check_matplotlib()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if class_names is None:
        class_names = ['Benign', 'Fraud']

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           ylabel='True Label', xlabel='Predicted Label',
           title=title)

    # add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=14)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f'Confusion matrix saved to {save_path}')


def plot_tsne_embeddings(embeddings, labels, title='t-SNE Visualization',
                         save_path='figures/tsne.pdf', perplexity=30):
    """
    Plot t-SNE visualization of learned node embeddings.
    :param embeddings: node embeddings (numpy array, shape: [n_nodes, embed_dim])
    :param labels: node labels
    :param title: plot title
    :param save_path: path to save the figure
    :param perplexity: t-SNE perplexity parameter
    """
    _check_matplotlib()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(8, 7))
    scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                         c=labels, cmap='coolwarm', alpha=0.6, s=10)
    ax.legend(*scatter.legend_elements(), title='Class', fontsize=11)
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.set_title(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f't-SNE visualization saved to {save_path}')


def plot_ablation_comparison(results_dict, metric='gnn_auc', save_path='figures/ablation.pdf'):
    """
    Plot bar chart comparing ablation study results.
    :param results_dict: dict mapping variant names to metric dicts
    :param metric: which metric to compare
    :param save_path: path to save the figure
    """
    _check_matplotlib()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    names = list(results_dict.keys())
    values = [results_dict[n].get(metric, 0) for n in names]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(names)), values, color='steelblue', edgecolor='black')

    # add value labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.005,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel(metric.replace('_', ' ').upper(), fontsize=12)
    ax.set_title('Ablation Study Results', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f'Ablation comparison plot saved to {save_path}')


# ===================== CAPN-Specific Visualizations =====================

def plot_capn_threshold_distribution(thresholds_per_relation, labels, relation_names=None,
                                     save_path='figures/capn_threshold_dist.pdf'):
    """
    Plot per-node threshold distributions for fraud vs benign nodes.
    Key CAPN insight: fraudulent nodes should get lower thresholds (more aggressive filtering).

    :param thresholds_per_relation: list of threshold tensors [num_nodes] per relation
    :param labels: node labels (0=benign, 1=fraud)
    :param relation_names: optional relation names
    :param save_path: path to save figure
    """
    _check_matplotlib()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    num_relations = len(thresholds_per_relation)
    if relation_names is None:
        relation_names = [f'Relation {i+1}' for i in range(num_relations)]

    labels = np.array(labels)
    fig, axes = plt.subplots(1, num_relations, figsize=(5 * num_relations, 4))
    if num_relations == 1:
        axes = [axes]

    for r, (thresholds, name) in enumerate(zip(thresholds_per_relation, relation_names)):
        if hasattr(thresholds, 'cpu'):
            thresholds = thresholds.cpu().numpy()
        else:
            thresholds = np.array(thresholds)

        benign_t = thresholds[labels == 0]
        fraud_t = thresholds[labels == 1]

        axes[r].hist(benign_t, bins=30, alpha=0.6, color='steelblue', label=f'Benign (μ={benign_t.mean():.3f})', density=True)
        axes[r].hist(fraud_t, bins=30, alpha=0.6, color='coral', label=f'Fraud (μ={fraud_t.mean():.3f})', density=True)
        axes[r].set_xlabel('Threshold', fontsize=11)
        axes[r].set_ylabel('Density', fontsize=11)
        axes[r].set_title(name, fontsize=13)
        axes[r].legend(fontsize=9)
        axes[r].grid(True, alpha=0.3)

    fig.suptitle('CAPN Per-Node Threshold Distributions', fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f'CAPN threshold distribution plot saved to {save_path}')


def plot_capn_reward_curves(reward_log, save_path='figures/capn_rewards.pdf'):
    """
    Plot CAPN reward component curves over training batches.

    :param reward_log: list of reward dicts from ShapedRewardComputer.reward_log
    :param save_path: path to save figure
    """
    _check_matplotlib()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    steps = range(len(reward_log))
    rewards = [r['reward'] for r in reward_log]
    dist_rewards = [r['dist_reward'] for r in reward_log]
    acc_rewards = [r['acc_reward'] for r in reward_log]
    reg_penalties = [r['reg_penalty'] for r in reward_log]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(steps, rewards, 'b-', linewidth=1, alpha=0.7)
    axes[0, 0].set_title('Total Shaped Reward', fontsize=12)
    axes[0, 0].set_ylabel('Reward', fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(steps, dist_rewards, 'g-', linewidth=1, alpha=0.7)
    axes[0, 1].set_title('Distance Improvement', fontsize=12)
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(steps, acc_rewards, 'r-', linewidth=1, alpha=0.7)
    axes[1, 0].set_title('Accuracy Signal', fontsize=12)
    axes[1, 0].set_xlabel('Training Step', fontsize=11)
    axes[1, 0].set_ylabel('Reward', fontsize=11)
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(steps, reg_penalties, 'm-', linewidth=1, alpha=0.7)
    axes[1, 1].set_title('Regularization Penalty', fontsize=12)
    axes[1, 1].set_xlabel('Training Step', fontsize=11)
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle('CAPN Shaped Reward Components', fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f'CAPN reward curves saved to {save_path}')


def plot_capn_policy_convergence(threshold_means_log, relation_names=None,
                                  save_path='figures/capn_policy_convergence.pdf'):
    """
    Plot mean threshold convergence for CAPN policy network.

    :param threshold_means_log: list of [mean_t_r1, mean_t_r2, ...] per evaluation step
    :param relation_names: optional relation names
    :param save_path: path to save figure
    """
    _check_matplotlib()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    data = np.array(threshold_means_log)
    num_relations = data.shape[1] if data.ndim > 1 else 1

    if relation_names is None:
        relation_names = [f'Relation {i+1}' for i in range(num_relations)]

    fig, ax = plt.subplots(figsize=(8, 5))
    for r in range(num_relations):
        vals = data[:, r] if data.ndim > 1 else data
        ax.plot(vals, label=relation_names[r], linewidth=2)

    ax.set_xlabel('Evaluation Step', fontsize=12)
    ax.set_ylabel('Mean Threshold', fontsize=12)
    ax.set_title('CAPN Policy Network: Mean Threshold Convergence', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f'CAPN policy convergence plot saved to {save_path}')
