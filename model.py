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

    Extends OneLayerCARE with policy-gradient learning on top of the CAPN
    policy network. Two RL variants are supported:

    * REINFORCE  (default): uses the normalised advantage from
      :class:`ShapedRewardComputer` directly in the policy loss.
    * Actor-Critic:        uses a :class:`ValueNetwork` critic to estimate
      V(s); advantage = R - V(s), and a TD(0) MSE loss trains the critic.

    Total loss:
        L = L_gnn + λ_1 * L_label + λ_policy * L_policy + λ_critic * L_critic

    Extra controls:
    * ``gnn_warmup_epochs`` - disable policy loss for the first N epochs so the
      GNN can stabilise before the RL signal is added.
    * ``lambda_policy_ramp_epochs`` - linearly ramp the effective
      ``lambda_policy`` from 0 up to ``lambda_policy`` over the given epochs
      after warmup (set to 0 to disable ramping).
    """

    def __init__(self, num_classes, inter1, lambda_1, lambda_policy=0.1,
                 reward_computer=None, loss_fn=None,
                 value_network=None, lambda_critic=0.1,
                 use_actor_critic=False,
                 gnn_warmup_epochs=0, lambda_policy_ramp_epochs=0):
        """
        :param num_classes: number of output classes
        :param inter1: InterAgg layer (with CAPN components attached)
        :param lambda_1: weight for label similarity loss
        :param lambda_policy: final weight for policy gradient loss
        :param reward_computer: ShapedRewardComputer instance
        :param loss_fn: optional custom loss function
        :param value_network: optional :class:`ValueNetwork` for Actor-Critic
        :param lambda_critic: weight for critic MSE loss
        :param use_actor_critic: if True, use critic-based advantage; otherwise
                                 fall back to REINFORCE using the reward
                                 computer's normalised advantage
        :param gnn_warmup_epochs: number of initial epochs in which the policy
                                  loss (and critic loss) is disabled
        :param lambda_policy_ramp_epochs: number of epochs after warmup to
                                          linearly ramp lambda_policy from 0
                                          up to its target value
        """
        super(CAPNOneLayerCARE, self).__init__(num_classes, inter1, lambda_1, loss_fn)
        self.lambda_policy = lambda_policy
        self.reward_computer = reward_computer
        self.value_network = value_network
        self.lambda_critic = lambda_critic
        self.use_actor_critic = use_actor_critic and value_network is not None
        self.gnn_warmup_epochs = gnn_warmup_epochs
        self.lambda_policy_ramp_epochs = lambda_policy_ramp_epochs
        self._current_epoch = 0

    def set_epoch(self, epoch):
        """Called from the training loop to schedule warmup/ramp."""
        self._current_epoch = int(epoch)

    def _effective_lambda_policy(self):
        """Return the current effective lambda_policy (respecting warmup + ramp)."""
        epoch = self._current_epoch
        if epoch < self.gnn_warmup_epochs:
            return 0.0
        if self.lambda_policy_ramp_epochs <= 0:
            return self.lambda_policy
        progress = min(1.0, (epoch - self.gnn_warmup_epochs) /
                       max(1, self.lambda_policy_ramp_epochs))
        return self.lambda_policy * progress

    def loss(self, nodes, labels, train_flag=True):
        gnn_scores, label_scores = self.forward(nodes, labels, train_flag)

        # standard CARE-GNN losses
        label_loss = self.xent(label_scores, labels.squeeze())
        gnn_loss = self.xent(gnn_scores, labels.squeeze())
        supervised_loss = gnn_loss + self.lambda_1 * label_loss

        eff_lambda = self._effective_lambda_policy() if train_flag else 0.0
        policy_loss = torch.tensor(0.0, device=gnn_scores.device)
        critic_loss = torch.tensor(0.0, device=gnn_scores.device)

        # Policy learning is only active once past warmup AND we have a
        # reward computer AND this is a training pass.
        if (train_flag and self.inter1.use_capn and self.reward_computer is not None
                and eff_lambda > 0):
            with torch.no_grad():
                preds = gnn_scores.argmax(dim=1)
                batch_acc = (preds == labels.squeeze()).float().mean().item()

            avg_dist = self.inter1.get_capn_avg_dist()
            capn_thresholds = self.inter1.get_capn_thresholds()

            if self.use_actor_critic:
                # Actor-Critic: critic estimates V(s); advantage = R - V(s)
                raw_reward, _ = self.reward_computer.compute_reward(
                    avg_dist, batch_acc, capn_thresholds, return_raw=True)

                states = self.inter1.get_capn_states()
                if states:
                    # Per-node, per-relation V(s): keep [B, 1] from each relation's
                    # ValueNetwork call and concat across relations -> [B, R].
                    # Letting V(s) vary per element is what gives the policy its
                    # B*R effective gradient samples (vs. 1 in the legacy scalar form).
                    values = torch.cat([self.value_network(s) for s in states], dim=1)
                    advantage = raw_reward - values.detach()  # [B, R]
                    policy_loss = self.inter1.get_capn_policy_loss(advantage)
                    # Critic regresses each per-element V(s) toward the observed
                    # scalar reward (broadcast over [B, R]).
                    critic_loss = ((values - raw_reward) ** 2).mean()
                else:
                    # states missing (e.g. first batch setup) — fall back to raw reward
                    policy_loss = self.inter1.get_capn_policy_loss(raw_reward)
            else:
                # REINFORCE path: normalised advantage from ShapedRewardComputer
                reward = self.reward_computer.compute_reward(
                    avg_dist, batch_acc, capn_thresholds)
                policy_loss = self.inter1.get_capn_policy_loss(reward)

            logger.debug(f'CAPN policy_loss: {policy_loss.item():.4f}, '
                         f'critic_loss: {critic_loss.item():.4f}, '
                         f'batch_acc: {batch_acc:.4f}, eff_lambda: {eff_lambda:.4f}')

        final_loss = supervised_loss
        if eff_lambda > 0:
            final_loss = (final_loss
                          + eff_lambda * policy_loss
                          + self.lambda_critic * critic_loss)
        return final_loss
