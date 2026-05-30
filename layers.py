import logging
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from operator import itemgetter
import math

logger = logging.getLogger(__name__)

"""
    CARE-GNN Layers
    Paper: Enhancing Graph Neural Network-based Fraud Detectors against Camouflaged Fraudsters
    Source: https://github.com/YingtongDou/CARE-GNN
"""


class FeatureEncoder(nn.Module):
    """Learnable nonlinear node-feature encoder (MLP).

    Replaces the single linear ``weight`` transform in the inter-relation
    aggregator. The original CARE-GNN backbone applies exactly one linear map
    (feat_dim -> embed_dim) + one ReLU before the classifier, so it cannot
    represent nonlinear feature interactions. A probe on the frozen YelpChi
    split shows a plain MLP / gradient-boosting classifier on the same node
    features beats the single-layer GNN by +4 to +8.6 AUC points — i.e. the
    backbone underfits. This module gives the backbone genuine nonlinear
    capacity while keeping the multi-relation aggregation, CAPN, and LLM gate
    intact. Applied identically to the centre node and the aggregated neighbour
    features so the combination stays consistent.
    """

    def __init__(self, feat_dim, embed_dim, hidden_dim=None, dropout=0.5, num_layers=2):
        super(FeatureEncoder, self).__init__()
        hidden_dim = hidden_dim or embed_dim
        layers = []
        d = feat_dim
        for _ in range(max(0, num_layers - 1)):
            layers += [nn.Linear(d, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            d = hidden_dim
        layers += [nn.Linear(d, embed_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class InterAgg(nn.Module):

    def __init__(self, features, feature_dim,
                 embed_dim, adj_lists, intraggs,
                 inter='GNN', step_size=0.02, device=None,
                 dropout=0.6, initial_thresholds=None,
                 threshold_min=0.001, threshold_max=0.999,
                 rl_patience=5, rl_epsilon=1e-4,
                 policy_network=None, state_constructor=None,
                 enhanced_label_clf=None, multi_view_distance=None,
                 soft_attn=False, feat_encoder=None, ego_separation=False):
        """
        Initialize the inter-relation aggregator.
        Generalized to support N relations (not hardcoded to 3).
        Optionally supports CAPN mode with per-node adaptive thresholds.
        :param features: the input node features or embeddings for all nodes
        :param feature_dim: the input dimension
        :param embed_dim: the output dimension
        :param adj_lists: a list of adjacency lists for each single-relation graph
        :param intraggs: the intra-relation aggregators used by each single-relation graph
        :param inter: the aggregator type: 'Att', 'Weight', 'Mean', 'GNN'
        :param step_size: the RL action step size
        :param device: torch device (cpu or cuda)
        :param dropout: dropout rate for attention aggregator
        :param initial_thresholds: initial filtering thresholds per relation
        :param threshold_min: minimum threshold value
        :param threshold_max: maximum threshold value
        :param rl_patience: number of epochs without improvement before RL stops
        :param rl_epsilon: minimum change in scores to be considered an improvement
        :param policy_network: optional CAPN PolicyNetwork for per-node thresholds
        :param state_constructor: optional CAPN StateConstructor
        :param enhanced_label_clf: optional CAPN EnhancedLabelPredictor
        :param multi_view_distance: optional CAPN MultiViewDistance
        """
        super(InterAgg, self).__init__()

        self.features = features
        self.dropout = dropout
        self.adj_lists = adj_lists
        self.num_relations = len(adj_lists)
        self.intra_aggs = nn.ModuleList(intraggs)
        self.embed_dim = embed_dim
        self.feat_dim = feature_dim
        self.inter = inter
        self.step_size = step_size
        self.device = device or torch.device('cpu')
        self.threshold_min = threshold_min
        self.threshold_max = threshold_max
        self.rl_patience = rl_patience
        self.rl_epsilon = rl_epsilon

        # CAPN components (None = use original CARE-GNN behavior)
        self.policy_network = policy_network
        self.state_constructor = state_constructor
        self.multi_view_distance = multi_view_distance
        self.use_capn = policy_network is not None
        self.soft_attn = soft_attn

        # optional nonlinear feature encoder (None = original single-linear transform)
        self.feat_encoder = feat_encoder

        # optional ego/neighbour separation: concat-project [self, neighbours]
        # instead of summing them, so the strong self-signal is not diluted by
        # heterophilous (camouflaged) neighbours. embed_dim stays unchanged.
        self.ego_separation = ego_separation
        self.combine = nn.Linear(2 * embed_dim, embed_dim) if ego_separation else None

        # set device on all intra-aggregators
        for agg in self.intra_aggs:
            agg.device = self.device

        # RL condition flag (only used in non-CAPN mode)
        self.RL = True

        # number of batches for current epoch, assigned during training
        self.batch_num = 0

        # initial filtering thresholds (used in non-CAPN mode)
        if initial_thresholds is not None:
            self.thresholds = list(initial_thresholds)
        else:
            self.thresholds = [0.5] * self.num_relations

        # the activation function used by attention mechanism
        self.leakyrelu = nn.LeakyReLU(0.2)

        # parameter used to transform node embeddings before inter-relation aggregation
        self.weight = nn.Parameter(torch.FloatTensor(self.feat_dim, self.embed_dim))
        init.xavier_uniform_(self.weight)

        # weight parameter for each relation used by CARE-Weight
        self.alpha = nn.Parameter(torch.FloatTensor(self.embed_dim, self.num_relations))
        init.xavier_uniform_(self.alpha)

        # parameters used by attention layer
        self.a = nn.Parameter(torch.FloatTensor(2 * self.embed_dim, 1))
        init.xavier_uniform_(self.a)

        # label predictor for similarity measure
        # use enhanced version if provided (CAPN mode), otherwise default linear
        if enhanced_label_clf is not None:
            self.label_clf = enhanced_label_clf
        else:
            self.label_clf = nn.Linear(self.feat_dim, 2)

        # initialize the parameter logs
        self.weights_log = []
        self.thresholds_log = [list(self.thresholds)]
        self.relation_score_log = []

        # CAPN: store per-batch policy outputs for loss computation
        self._capn_log_probs = []
        self._capn_thresholds = []
        self._capn_avg_dist = 0.0
        self._capn_states = []

    def forward(self, nodes, labels, train_flag=True):
        """
        :param nodes: a list of batch node ids
        :param labels: a list of batch node labels, only used by the RLModule
        :param train_flag: indicates whether in training or testing mode
        :return combined: the embeddings of a batch of input node features
        :return center_scores: the label-aware scores of batch nodes
        """

        # extract 1-hop neighbor ids from adj lists of each single-relation graph
        to_neighs = []
        for adj_list in self.adj_lists:
            to_neighs.append([set(adj_list[int(node)]) for node in nodes])

        # find unique nodes and their neighbors used in current batch
        all_sets = [set(nodes)]
        for rel_neighs in to_neighs:
            all_sets.append(set.union(*rel_neighs))
        unique_nodes = set.union(*all_sets)

        # calculate label-aware scores
        batch_features = self.features(torch.LongTensor(list(unique_nodes)).to(self.device))
        batch_scores = self.label_clf(batch_features)
        id_mapping = {node_id: index for index, node_id in enumerate(unique_nodes)}

        # the label-aware scores for current batch of nodes
        center_scores = batch_scores[itemgetter(*nodes)(id_mapping), :]

        # get neighbor node id list and scores for each relation
        r_lists = []
        r_scores = []
        r_sample_num_lists = []

        if self.use_capn:
            # CAPN mode: compute per-node thresholds via policy network
            batch_log_probs = []
            batch_thresholds = []
            batch_states = []  # keep states so the critic can estimate V(s)
            self.policy_network.reset_episode()

            for r_idx in range(self.num_relations):
                r_list = [list(to_neigh) for to_neigh in to_neighs[r_idx]]
                r_lists.append(r_list)
                r_score = [batch_scores[itemgetter(*to_neigh)(id_mapping), :].view(-1, 2) for to_neigh in r_list]
                r_scores.append(r_score)

                # compute state vectors for policy network
                state = self.state_constructor.compute_state(
                    nodes, r_idx, center_scores, r_score, r_list)
                batch_states.append(state)

                # get per-node thresholds from policy
                deterministic = not train_flag
                node_thresholds, log_probs = self.policy_network(state, r_idx, deterministic)
                batch_log_probs.append(log_probs)
                batch_thresholds.append(node_thresholds)

                # store for policy gradient
                self.policy_network.store_action(log_probs, node_thresholds)

                # CAPN: pass per-node thresholds as temperatures for soft attention
                # or as sample counts for hard filtering (backward compat)
                if self.soft_attn:
                    # threshold is used as temperature directly
                    temps = [node_thresholds[i].item() for i in range(len(r_list))]
                    r_sample_num_lists.append(temps)
                else:
                    sample_nums = [max(1, math.ceil(len(neighs) * node_thresholds[i].item()))
                                   for i, neighs in enumerate(r_list)]
                    r_sample_num_lists.append(sample_nums)

            # store for reward computation
            self._capn_log_probs = batch_log_probs
            self._capn_thresholds = batch_thresholds
            self._capn_states = batch_states
        else:
            # Original CARE-GNN mode: global thresholds per relation
            for r_idx in range(self.num_relations):
                r_list = [list(to_neigh) for to_neigh in to_neighs[r_idx]]
                r_lists.append(r_list)
                r_score = [batch_scores[itemgetter(*to_neigh)(id_mapping), :].view(-1, 2) for to_neigh in r_list]
                r_scores.append(r_score)
                r_sample_num_lists.append([math.ceil(len(neighs) * self.thresholds[r_idx]) for neighs in r_list])

        # intra-aggregation steps for each relation (Eq. 8)
        r_feats_list = []
        r_scores_out = []
        for r_idx in range(self.num_relations):
            use_soft = self.use_capn and self.soft_attn
            r_feats, r_samp_scores = self.intra_aggs[r_idx].forward(
                nodes, r_lists[r_idx], center_scores, r_scores[r_idx], r_sample_num_lists[r_idx],
                multi_view_distance=self.multi_view_distance if self.use_capn else None,
                relation_idx=r_idx, soft_attn=use_soft)
            r_feats_list.append(r_feats)
            r_scores_out.append(r_samp_scores)

        # compute average distance for CAPN reward
        if self.use_capn:
            total_dist = 0.0
            total_count = 0
            for scores_list in r_scores_out:
                for s in scores_list:
                    if isinstance(s, list):
                        total_dist += sum(s)
                        total_count += len(s)
                    elif isinstance(s, float):
                        total_dist += s
                        total_count += 1
            self._capn_avg_dist = total_dist / max(total_count, 1)

        # concat the intra-aggregated embeddings from each relation
        neigh_feats = torch.cat(r_feats_list, dim=0)

        # get features or embeddings for batch nodes
        self_feats = self.features(torch.LongTensor(nodes).to(self.device))

        # number of nodes in a batch
        n = len(nodes)

        # inter-relation aggregation steps (Eq. 9)
        if self.use_capn and self.inter == 'GNN':
            # CAPN mode: use mean of per-node thresholds as inter-agg weights
            mean_thresholds = [t.mean().item() for t in self._capn_thresholds]
            combined = threshold_inter_agg(
                self.num_relations, self_feats, neigh_feats, self.embed_dim,
                self.weight, mean_thresholds, n, self.device,
                encoder=self.feat_encoder, combine=self.combine)
        elif self.inter == 'Att':
            combined, attention = att_inter_agg(
                self.num_relations, self.leakyrelu, self_feats, neigh_feats,
                self.embed_dim, self.weight, self.a, n, self.dropout, self.training, self.device)
        elif self.inter == 'Weight':
            combined = weight_inter_agg(
                self.num_relations, self_feats, neigh_feats, self.embed_dim,
                self.weight, self.alpha, n, self.device)
            gem_weights = F.softmax(torch.sum(self.alpha, dim=0), dim=0).tolist()
            if train_flag:
                logger.debug(f'Weights: {gem_weights}')
        elif self.inter == 'Mean':
            combined = mean_inter_agg(
                self.num_relations, self_feats, neigh_feats, self.embed_dim,
                self.weight, n, self.device)
        elif self.inter == 'GNN':
            combined = threshold_inter_agg(
                self.num_relations, self_feats, neigh_feats, self.embed_dim,
                self.weight, self.thresholds, n, self.device,
                encoder=self.feat_encoder, combine=self.combine)

        # the reinforcement learning module (only in non-CAPN mode)
        if not self.use_capn and self.RL and train_flag:
            relation_scores, rewards, thresholds, stop_flag = RLModule(
                r_scores_out, self.relation_score_log, labels, self.thresholds,
                self.batch_num, self.step_size, self.threshold_min, self.threshold_max,
                self.rl_patience, self.rl_epsilon)
            self.thresholds = thresholds
            self.RL = stop_flag
            self.relation_score_log.append(relation_scores)
            self.thresholds_log.append(list(self.thresholds))

        return combined, center_scores

    def get_capn_policy_loss(self, reward):
        """
        Get the CAPN policy gradient loss for the current batch.
        Called by CAPNOneLayerCARE.loss() after computing shaped reward.
        :param reward: scalar reward from ShapedRewardComputer
        :return: policy gradient loss tensor
        """
        if not self.use_capn or self.policy_network is None:
            return torch.tensor(0.0, device=self.device)
        return self.policy_network.get_policy_loss(reward)

    def get_capn_avg_dist(self):
        """Get the latest average distance for reward computation."""
        return self._capn_avg_dist

    def get_capn_thresholds(self):
        """Get the latest per-node thresholds for logging."""
        if self._capn_thresholds:
            return [t.detach() for t in self._capn_thresholds]
        return []

    def get_capn_states(self):
        """Get the latest per-relation state tensors used by the policy network.

        Returns a list of tensors (one per relation), each [batch_size, state_dim].
        Used by the Actor-Critic ValueNetwork to estimate V(s).
        """
        return list(self._capn_states) if self._capn_states else []


class IntraAgg(nn.Module):

    def __init__(self, features, feat_dim, device=None, cuda=False):
        """
        Initialize the intra-relation aggregator
        :param features: the input node features or embeddings for all nodes
        :param feat_dim: the input dimension
        :param device: torch device
        :param cuda: whether to use GPU (legacy, prefer device)
        """
        super(IntraAgg, self).__init__()

        self.features = features
        self.feat_dim = feat_dim
        if device is not None:
            self.device = device
        else:
            self.device = torch.device('cuda' if cuda else 'cpu')

    def forward(self, nodes, to_neighs_list, batch_scores, neigh_scores, sample_list,
                multi_view_distance=None, relation_idx=None, soft_attn=False):
        """
        Code partially from https://github.com/williamleif/graphsage-simple/
        :param nodes: list of nodes in a batch
        :param to_neighs_list: neighbor node id list for each batch node in one relation
        :param batch_scores: the label-aware scores of batch nodes
        :param neigh_scores: the label-aware scores 1-hop neighbors each batch node in one relation
        :param sample_list: the number of neighbors kept for each batch node in one relation
                           In soft_attn mode, these are temperature values (floats) per node.
        :param multi_view_distance: optional MultiViewDistance for CAPN mode
        :param relation_idx: relation index for multi-view distance
        :param soft_attn: if True, use soft attention weighting instead of hard top-K
        :return to_feats: the aggregated embeddings of batch nodes neighbors in one relation
        :return samp_scores: the average neighbor distances for each relation after filtering
        """

        if soft_attn:
            return self._forward_soft_attn(
                nodes, to_neighs_list, batch_scores, neigh_scores, sample_list,
                multi_view_distance=multi_view_distance, relation_idx=relation_idx)

        # filter neighbors under given relation
        samp_neighs, samp_scores = filter_neighs_ada_threshold(
            batch_scores, neigh_scores, to_neighs_list, sample_list,
            multi_view_distance=multi_view_distance, center_nodes=nodes, relation_idx=relation_idx)

        # find the unique nodes among batch nodes and the filtered neighbors
        unique_nodes_list = list(set.union(*samp_neighs))
        unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}

        # intra-relation aggregation only with sampled neighbors
        mask = torch.zeros(len(samp_neighs), len(unique_nodes))
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
        row_indices = [i for i in range(len(samp_neighs)) for _ in range(len(samp_neighs[i]))]
        mask[row_indices, column_indices] = 1
        mask = mask.to(self.device)
        num_neigh = mask.sum(1, keepdim=True)
        num_neigh = num_neigh.clamp(min=1)
        mask = mask.div(num_neigh)
        embed_matrix = self.features(torch.LongTensor(unique_nodes_list).to(self.device))
        to_feats = mask.mm(embed_matrix)
        to_feats = F.relu(to_feats)
        return to_feats, samp_scores

    def _forward_soft_attn(self, nodes, to_neighs_list, batch_scores, neigh_scores, temperature_list,
                           multi_view_distance=None, relation_idx=None):
        """
        Soft attention aggregation: use ALL neighbors with distance-based attention weights.
        Temperature controls sharpness: low temp = focus on closest, high temp = uniform.
        Vectorized implementation for performance.
        """
        batch_size = len(nodes)

        # Collect all neighbor node ids across the batch
        all_neighs_flat = []
        for neighs in to_neighs_list:
            all_neighs_flat.extend(neighs)
        if not all_neighs_flat:
            zero_feats = torch.zeros(batch_size, self.feat_dim, device=self.device)
            return zero_feats, [[] for _ in range(batch_size)]

        unique_nodes_list = list(set(all_neighs_flat) | set(nodes))
        unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}
        embed_matrix = self.features(torch.LongTensor(unique_nodes_list).to(self.device))

        # Pre-compute all distances and build sparse index arrays
        row_ids = []
        col_ids = []
        logit_vals = []
        samp_scores = []

        for idx in range(batch_size):
            neighs_indices = to_neighs_list[idx]
            if len(neighs_indices) == 0:
                samp_scores.append([])
                continue

            center_score = batch_scores[idx][0]
            neigh_score = neigh_scores[idx][:, 0].view(-1, 1)

            # Compute distances
            if multi_view_distance is not None:
                score_diff = multi_view_distance.compute_distance(
                    center_score, neigh_score, neighs_indices,
                    nodes[idx], relation_idx)
            else:
                cs = center_score.repeat(neigh_score.size()[0], 1)
                score_diff = torch.abs(cs - neigh_score).squeeze()

            if score_diff.dim() == 0:
                score_diff = score_diff.unsqueeze(0)

            samp_scores.append(score_diff.detach().tolist())

            # Temperature from policy
            temp = max(temperature_list[idx], 0.01)
            attn_logits = -score_diff / temp

            # Build index arrays for sparse → dense conversion
            cols = [unique_nodes[n] for n in neighs_indices]
            row_ids.extend([idx] * len(cols))
            col_ids.extend(cols)
            logit_vals.extend(attn_logits.detach().cpu().tolist() if attn_logits.dim() > 0 else [attn_logits.item()])

        # Build dense mask from sparse indices
        mask = torch.full((batch_size, len(unique_nodes_list)), float('-inf'), device=self.device)
        if row_ids:
            mask[row_ids, col_ids] = torch.tensor(logit_vals, device=self.device)

        # Softmax over neighbors for each node
        attn_weights = F.softmax(mask, dim=1)
        # Zero out nodes with no neighbors
        has_neighs = (mask > float('-inf')).any(dim=1, keepdim=True).float()
        attn_weights = attn_weights * has_neighs

        to_feats = attn_weights.mm(embed_matrix)
        to_feats = F.relu(to_feats)
        return to_feats, samp_scores


def RLModule(scores, scores_log, labels, thresholds, batch_num, step_size,
             threshold_min=0.001, threshold_max=0.999, rl_patience=5, rl_epsilon=1e-4):
    """
    The reinforcement learning module with terminal condition.
    It updates the neighbor filtering threshold for each relation based
    on the average neighbor distances between two consecutive epochs.
    :param scores: the neighbor nodes label-aware scores for each relation
    :param scores_log: a list stores the relation average distances for each batch
    :param labels: the batch node labels used to select positive nodes
    :param thresholds: the current neighbor filtering thresholds for each relation
    :param batch_num: numbers batches in an epoch
    :param step_size: the RL action step size
    :param threshold_min: minimum allowed threshold value
    :param threshold_max: maximum allowed threshold value
    :param rl_patience: epochs without improvement before stopping RL updates
    :param rl_epsilon: minimum score change to count as improvement
    :return relation_scores: the relation average distances for current batch
    :return rewards: the reward for given thresholds in current epoch
    :return new_thresholds: the new filtering thresholds updated according to the rewards
    :return stop_flag: the RL terminal condition flag (False = stop updating)
    """

    relation_scores = []
    stop_flag = True
    num_relations = len(scores)

    # only compute the average neighbor distances for positive nodes
    pos_index = (labels == 1).nonzero().tolist()
    pos_index = [i[0] for i in pos_index]

    # compute average neighbor distances for each relation
    for score in scores:
        pos_scores = itemgetter(*pos_index)(score)
        neigh_count = sum([1 if isinstance(i, float) else len(i) for i in pos_scores])
        pos_sum = [i if isinstance(i, float) else sum(i) for i in pos_scores]
        relation_scores.append(sum(pos_sum) / max(neigh_count, 1))

    if len(scores_log) % batch_num != 0 or len(scores_log) < 2 * batch_num:
        # do not call RL module within the epoch or within the first two epochs
        rewards = [0] * num_relations
        new_thresholds = thresholds
    else:
        # update thresholds according to average scores in last epoch
        # Eq.(5) in the paper
        previous_epoch_scores = [sum(s) / batch_num for s in zip(*scores_log[-2 * batch_num:-batch_num])]
        current_epoch_scores = [sum(s) / batch_num for s in zip(*scores_log[-batch_num:])]

        # compute reward for each relation and update the thresholds according to reward
        # Eq. (6) in the paper
        rewards = [1 if previous_epoch_scores[i] - s >= 0 else -1 for i, s in enumerate(current_epoch_scores)]
        new_thresholds = [thresholds[i] + step_size if r == 1 else thresholds[i] - step_size for i, r in enumerate(rewards)]

        # clamp thresholds to valid range
        new_thresholds = [max(threshold_min, min(threshold_max, t)) for t in new_thresholds]

        logger.info(f'RL update - scores: {[f"{s:.4f}" for s in current_epoch_scores]}, '
                    f'rewards: {rewards}, thresholds: {[f"{t:.4f}" for t in new_thresholds]}')

        # Terminal condition: stop if score changes are below epsilon for rl_patience epochs
        if len(scores_log) >= (rl_patience + 2) * batch_num:
            recent_epochs = rl_patience + 1
            epoch_scores_history = []
            for ep in range(recent_epochs):
                start_idx = -(recent_epochs - ep) * batch_num
                end_idx = start_idx + batch_num if start_idx + batch_num != 0 else None
                epoch_avg = [sum(s) / batch_num for s in zip(*scores_log[start_idx:end_idx])]
                epoch_scores_history.append(epoch_avg)

            # check if all relations have converged
            all_converged = True
            for r in range(num_relations):
                recent_changes = [abs(epoch_scores_history[e+1][r] - epoch_scores_history[e][r])
                                  for e in range(len(epoch_scores_history) - 1)]
                if max(recent_changes) > rl_epsilon:
                    all_converged = False
                    break

            if all_converged:
                stop_flag = False
                logger.info(f'RL terminal condition reached: scores converged (epsilon={rl_epsilon}, '
                            f'patience={rl_patience})')

    return relation_scores, rewards, new_thresholds, stop_flag


def filter_neighs_ada_threshold(center_scores, neigh_scores, neighs_list, sample_list,
                                multi_view_distance=None, center_nodes=None, relation_idx=None):
    """
    Filter neighbors according label predictor result with adaptive thresholds
    :param center_scores: the label-aware scores of batch nodes
    :param neigh_scores: the label-aware scores 1-hop neighbors each batch node in one relation
    :param neighs_list: neighbor node id list for each batch node in one relation
    :param sample_list: the number of neighbors kept for each batch node in one relation
    :param multi_view_distance: optional MultiViewDistance for combined L1+structural distance
    :param center_nodes: batch node ids (required when multi_view_distance is set)
    :param relation_idx: relation index (required when multi_view_distance is set)
    :return samp_neighs: the neighbor indices and neighbor simi scores
    :return samp_scores: the average neighbor distances for each relation after filtering
    """

    samp_neighs = []
    samp_scores = []
    for idx, center_score in enumerate(center_scores):
        center_score = center_scores[idx][0]
        neigh_score = neigh_scores[idx][:, 0].view(-1, 1)
        neighs_indices = neighs_list[idx]
        num_sample = sample_list[idx]

        if multi_view_distance is not None and center_nodes is not None:
            # multi-view distance: L1 label distance + structural Jaccard
            score_diff = multi_view_distance.compute_distance(
                center_score, neigh_score, neighs_indices,
                center_nodes[idx], relation_idx)
        else:
            # compute the L1-distance of batch nodes and their neighbors
            # Eq. (2) in paper
            center_score = center_score.repeat(neigh_score.size()[0], 1)
            score_diff = torch.abs(center_score - neigh_score).squeeze()
        sorted_scores, sorted_indices = torch.sort(score_diff, dim=0, descending=False)
        selected_indices = sorted_indices.tolist()

        # top-p sampling according to distance ranking and thresholds
        # Section 3.3.1 in paper
        if len(neigh_scores[idx]) > num_sample + 1:
            selected_neighs = [neighs_indices[n] for n in selected_indices[:num_sample]]
            selected_scores = sorted_scores.tolist()[:num_sample]
        else:
            selected_neighs = neighs_indices
            selected_scores = score_diff.tolist()
            if isinstance(selected_scores, float):
                selected_scores = [selected_scores]

        samp_neighs.append(set(selected_neighs))
        samp_scores.append(selected_scores)

    return samp_neighs, samp_scores


def mean_inter_agg(num_relations, self_feats, neigh_feats, embed_dim, weight, n, device):
    """
    Mean inter-relation aggregator
    :param num_relations: number of relations in the graph
    :param self_feats: batch nodes features or embeddings
    :param neigh_feats: intra-relation aggregated neighbor embeddings for each relation
    :param embed_dim: the dimension of output embedding
    :param weight: parameter used to transform node embeddings before inter-relation aggregation
    :param n: number of nodes in a batch
    :param device: torch device
    :return: inter-relation aggregated node embeddings
    """

    # transform batch node embedding and neighbor embedding in each relation with weight parameter
    center_h = torch.mm(self_feats, weight)
    neigh_h = torch.mm(neigh_feats, weight)

    aggregated = torch.zeros(size=(n, embed_dim), device=device)

    # sum neighbor embeddings together
    for r in range(num_relations):
        aggregated += neigh_h[r * n:(r + 1) * n, :]

    # take the average and apply activation
    combined = F.relu((center_h + aggregated) / (num_relations + 1.0))

    return combined


def weight_inter_agg(num_relations, self_feats, neigh_feats, embed_dim, weight, alpha, n, device):
    """
    Weight inter-relation aggregator
    Reference: https://arxiv.org/abs/2002.12307
    """

    center_h = torch.mm(self_feats, weight)
    neigh_h = torch.mm(neigh_feats, weight)

    # compute relation weights using softmax
    w = F.softmax(alpha, dim=1)

    aggregated = torch.zeros(size=(n, embed_dim), device=device)

    # add weighted neighbor embeddings in each relation together
    for r in range(num_relations):
        aggregated += neigh_h[r * n:(r + 1) * n, :] * w[:, r]

    combined = F.relu(center_h + aggregated)

    return combined


def att_inter_agg(num_relations, att_layer, self_feats, neigh_feats, embed_dim, weight, a, n, dropout, training, device):
    """
    Attention-based inter-relation aggregator
    Reference: https://github.com/Diego999/pyGAT
    """

    center_h = torch.mm(self_feats, weight)
    neigh_h = torch.mm(neigh_feats, weight)

    # compute attention weights
    combined = torch.cat((center_h.repeat(num_relations, 1), neigh_h), dim=1)
    e = att_layer(combined.mm(a))
    attention_parts = [e[r * n:(r + 1) * n, :] for r in range(num_relations)]
    attention = torch.cat(attention_parts, dim=1)
    ori_attention = F.softmax(attention, dim=1)
    attention = F.dropout(ori_attention, dropout, training=training)

    aggregated = torch.zeros(size=(n, embed_dim), device=device)

    # add neighbor embeddings in each relation together with attention weights
    for r in range(num_relations):
        aggregated += torch.mul(attention[:, r].unsqueeze(1).repeat(1, embed_dim), neigh_h[r * n:(r + 1) * n, :])

    combined = F.relu(center_h + aggregated)

    # extract the attention weights
    att = F.softmax(torch.sum(ori_attention, dim=0), dim=0)

    return combined, att


def threshold_inter_agg(num_relations, self_feats, neigh_feats, embed_dim, weight, threshold, n, device, encoder=None, combine=None):
    """
    CARE-GNN inter-relation aggregator
    Eq. (9) in the paper

    When ``encoder`` is provided, the single linear ``weight`` transform is
    replaced by the nonlinear MLP encoder (applied identically to centre and
    neighbour features) to give the backbone nonlinear capacity.

    When ``combine`` (a Linear[2*embed_dim, embed_dim]) is provided, the centre
    and aggregated-neighbour representations are concatenated and projected
    instead of summed, keeping the ego signal in its own channels — important on
    heterophilous fraud graphs where neighbours are camouflaged.
    """

    if encoder is not None:
        center_h = encoder(self_feats)
        neigh_h = encoder(neigh_feats)
    else:
        center_h = torch.mm(self_feats, weight)
        neigh_h = torch.mm(neigh_feats, weight)

    aggregated = torch.zeros(size=(n, embed_dim), device=device)

    # add weighted neighbor embeddings in each relation together
    for r in range(num_relations):
        aggregated += neigh_h[r * n:(r + 1) * n, :] * threshold[r]

    if combine is not None:
        combined = F.relu(combine(torch.cat([center_h, aggregated], dim=1)))
    else:
        combined = F.relu(center_h + aggregated)

    return combined
