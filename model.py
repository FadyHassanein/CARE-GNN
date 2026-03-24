import logging

import torch
import torch.nn as nn
from torch.nn import init

logger = logging.getLogger(__name__)

"""
    CARE-GNN Models
    Paper: Enhancing Graph Neural Network-based Fraud Detectors against Camouflaged Fraudsters
    Source: https://github.com/YingtongDou/CARE-GNN
"""


class OneLayerCARE(nn.Module):
    """
    The CARE-GNN model in one layer
    """

    def __init__(self, num_classes, inter1, lambda_1, loss_fn=None):
        """
        Initialize the CARE-GNN model
        :param num_classes: number of classes (2 in our paper)
        :param inter1: the inter-relation aggregator that output the final embedding
        :param lambda_1: weight for the similarity loss
        :param loss_fn: optional custom loss function (default: CrossEntropyLoss)
        """
        super(OneLayerCARE, self).__init__()
        self.inter1 = inter1
        self.xent = loss_fn if loss_fn is not None else nn.CrossEntropyLoss()

        # the parameter to transform the final embedding
        self.weight = nn.Parameter(torch.FloatTensor(inter1.embed_dim, num_classes))
        init.xavier_uniform_(self.weight)
        self.lambda_1 = lambda_1

    def forward(self, nodes, labels, train_flag=True):
        embeds1, label_scores = self.inter1(nodes, labels, train_flag)
        scores = torch.mm(embeds1, self.weight)
        return scores, label_scores

    def to_prob(self, nodes, labels, train_flag=True):
        gnn_scores, label_scores = self.forward(nodes, labels, train_flag)
        gnn_prob = nn.functional.softmax(gnn_scores, dim=1)
        label_prob = nn.functional.softmax(label_scores, dim=1)
        return gnn_prob, label_prob

    def loss(self, nodes, labels, train_flag=True):
        gnn_scores, label_scores = self.forward(nodes, labels, train_flag)
        # Simi loss, Eq. (4) in the paper
        label_loss = self.xent(label_scores, labels.squeeze())
        # GNN loss, Eq. (10) in the paper
        gnn_loss = self.xent(gnn_scores, labels.squeeze())
        # the loss function of CARE-GNN, Eq. (11) in the paper
        final_loss = gnn_loss + self.lambda_1 * label_loss
        return final_loss


class MultiLayerCARE(nn.Module):
    """
    Multi-layer CARE-GNN model with residual connections.
    Extends the original single-layer model as noted in the paper's future work.
    """

    def __init__(self, num_classes, inter_layers, lambda_1, loss_fn=None, dropout=0.5):
        """
        :param num_classes: number of classes
        :param inter_layers: list of InterAgg layers (one per GNN layer)
        :param lambda_1: weight for the similarity loss
        :param loss_fn: optional custom loss function
        :param dropout: dropout between layers
        """
        super(MultiLayerCARE, self).__init__()
        self.inter_layers = nn.ModuleList(inter_layers)
        self.num_layers = len(inter_layers)
        self.xent = loss_fn if loss_fn is not None else nn.CrossEntropyLoss()
        self.lambda_1 = lambda_1
        self.dropout = nn.Dropout(dropout)

        # projection weights for each layer (for residual connections when dims differ)
        self.layer_weights = nn.ParameterList()
        for i, inter in enumerate(inter_layers):
            if i == 0:
                w = nn.Parameter(torch.FloatTensor(inter.feat_dim, inter.embed_dim))
            else:
                w = nn.Parameter(torch.FloatTensor(inter_layers[i-1].embed_dim, inter.embed_dim))
            init.xavier_uniform_(w)
            self.layer_weights.append(w)

        # final classification weight
        last_embed_dim = inter_layers[-1].embed_dim
        self.weight = nn.Parameter(torch.FloatTensor(last_embed_dim, num_classes))
        init.xavier_uniform_(self.weight)

    def forward(self, nodes, labels, train_flag=True):
        all_label_scores = []
        embeds = None

        for i, (inter, w) in enumerate(zip(self.inter_layers, self.layer_weights)):
            layer_embeds, label_scores = inter(nodes, labels, train_flag)
            all_label_scores.append(label_scores)

            if embeds is not None:
                # residual connection
                residual = torch.mm(embeds, w)
                layer_embeds = layer_embeds + residual

            if i < self.num_layers - 1:
                layer_embeds = self.dropout(layer_embeds)

            embeds = layer_embeds

        scores = torch.mm(embeds, self.weight)
        # use label scores from the last layer
        return scores, all_label_scores[-1]

    def to_prob(self, nodes, labels, train_flag=True):
        gnn_scores, label_scores = self.forward(nodes, labels, train_flag)
        gnn_prob = nn.functional.softmax(gnn_scores, dim=1)
        label_prob = nn.functional.softmax(label_scores, dim=1)
        return gnn_prob, label_prob

    def loss(self, nodes, labels, train_flag=True):
        gnn_scores, label_scores = self.forward(nodes, labels, train_flag)
        label_loss = self.xent(label_scores, labels.squeeze())
        gnn_loss = self.xent(gnn_scores, labels.squeeze())
        final_loss = gnn_loss + self.lambda_1 * label_loss
        return final_loss


class CAPNOneLayerCARE(OneLayerCARE):
    """
    CAPN-enhanced CARE-GNN model.

    Extends OneLayerCARE with policy gradient loss from the CAPN policy network.
    The total loss becomes:
        L = L_gnn + λ₁ * L_label + λ_policy * L_policy

    Where L_policy is the REINFORCE policy gradient loss for adaptive thresholds.
    """

    def __init__(self, num_classes, inter1, lambda_1, lambda_policy=0.1,
                 reward_computer=None, loss_fn=None):
        """
        :param num_classes: number of output classes
        :param inter1: InterAgg layer (with CAPN components attached)
        :param lambda_1: weight for label similarity loss
        :param lambda_policy: weight for policy gradient loss
        :param reward_computer: ShapedRewardComputer instance
        :param loss_fn: optional custom loss function
        """
        super(CAPNOneLayerCARE, self).__init__(num_classes, inter1, lambda_1, loss_fn)
        self.lambda_policy = lambda_policy
        self.reward_computer = reward_computer

    def loss(self, nodes, labels, train_flag=True):
        gnn_scores, label_scores = self.forward(nodes, labels, train_flag)

        # standard CARE-GNN losses
        label_loss = self.xent(label_scores, labels.squeeze())
        gnn_loss = self.xent(gnn_scores, labels.squeeze())
        supervised_loss = gnn_loss + self.lambda_1 * label_loss

        # CAPN policy gradient loss
        policy_loss = torch.tensor(0.0, device=gnn_scores.device)
        if train_flag and self.inter1.use_capn and self.reward_computer is not None:
            # compute batch accuracy for reward
            with torch.no_grad():
                preds = gnn_scores.argmax(dim=1)
                batch_acc = (preds == labels.squeeze()).float().mean().item()

            # compute shaped reward
            avg_dist = self.inter1.get_capn_avg_dist()
            capn_thresholds = self.inter1.get_capn_thresholds()
            reward = self.reward_computer.compute_reward(avg_dist, batch_acc, capn_thresholds)

            # compute policy gradient loss
            policy_loss = self.inter1.get_capn_policy_loss(reward)

            logger.debug(f'CAPN reward: {reward:.4f}, policy_loss: {policy_loss.item():.4f}, '
                         f'batch_acc: {batch_acc:.4f}')

        final_loss = supervised_loss + self.lambda_policy * policy_loss
        return final_loss
