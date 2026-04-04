"""
CAPN: Camouflage-Aware Policy Network for Adaptive Neighbor Selection

A novel framework extension for CARE-GNN that replaces the heuristic RL module
with a learned policy network producing per-node adaptive filtering thresholds.

Key components:
- EnhancedLabelPredictor: Non-linear MLP replacing the linear label classifier
- StateConstructor: Builds state vectors from node features + structural statistics
- PolicyNetwork: Beta-distribution policy outputting per-node thresholds
- ShapedRewardComputer: Magnitude-aware reward combining distance + accuracy signals
- LLMPriorLoader: Loads pre-generated LLM domain priors for warm-starting
"""

import json
import logging
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta

logger = logging.getLogger(__name__)


class EnhancedLabelPredictor(nn.Module):
    """
    Non-linear label predictor replacing nn.Linear(feat_dim, 2).

    Uses a 2-layer MLP with ReLU and dropout to capture non-linear
    feature-label relationships that the original linear predictor misses.
    Also computes prediction confidence via entropy of softmax output.
    """

    def __init__(self, feat_dim, num_classes=2, hidden_dim=None, dropout=0.3):
        super(EnhancedLabelPredictor, self).__init__()
        if hidden_dim is None:
            hidden_dim = max(feat_dim // 2, 32)

        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, features):
        """
        :param features: node features [batch_size, feat_dim]
        :return scores: label-aware scores [batch_size, num_classes]
        """
        return self.mlp(features)

class StateConstructor:
    """
    Constructs state vectors for the policy network.

    For each node v in relation r, the state is:
        s_v^r = [x_v, conf_v, deg_r(v), mean_dist_r(v), overlap_r(v), feat_var_r(v)]
    or with LLM enrichment:
        s_v^r = [x_v, conf_v, deg_r(v), mean_dist_r(v), overlap_r(v), feat_var_r(v), e_LLM(v)]

    Where:
    - x_v: node features (feat_dim)
    - conf_v: label predictor confidence (1)
    - deg_r(v): normalized degree in relation r (1)
    - mean_dist_r(v): average L1 distance to neighbors (1)
    - overlap_r(v): Jaccard overlap with homogeneous graph (1)
    - feat_var_r(v): mean feature variance of neighbors (1)
    - e_LLM(v): projected LLM semantic embedding (projection_dim, optional)
    """

    def __init__(self, adj_lists, homo_adj, features, device=None,
                 llm_embeddings=None, llm_projector=None,
                 enrichment_tensor=None, enrichment_projector=None):
        """
        :param adj_lists: list of adjacency lists for each relation
        :param homo_adj: homogeneous graph adjacency list (for structural overlap)
        :param features: nn.Embedding for node features
        :param device: torch device
        :param llm_embeddings: precomputed LLM semantic embeddings [N, 384] (optional, v1)
        :param llm_projector: LLMProjector module (optional, v1)
        :param enrichment_tensor: precomputed graph/reasoning features [N, D] (optional, v2)
        :param enrichment_projector: LLMProjector module for v2 features (optional)
        """
        self.adj_lists = adj_lists
        self.homo_adj = homo_adj
        self.features = features
        self.device = device or torch.device('cpu')
        self.num_relations = len(adj_lists)
        self.llm_embeddings = llm_embeddings
        self.llm_projector = llm_projector
        self.enrichment_tensor = enrichment_tensor
        self.enrichment_projector = enrichment_projector

        # precompute max degree per relation for normalization
        self.max_degrees = []
        for adj_list in adj_lists:
            max_deg = max((len(neighbors) for neighbors in adj_list.values()), default=1)
            self.max_degrees.append(max(max_deg, 1))

    def compute_state(self, nodes, relation_idx, label_scores, neigh_scores_list, neighs_list):
        """
        Compute state vectors for a batch of nodes in a given relation.

        :param nodes: list of node ids
        :param relation_idx: which relation (0, 1, 2, ...)
        :param label_scores: label-aware scores for batch nodes [batch, 2]
        :param neigh_scores_list: list of neighbor score tensors per node
        :param neighs_list: list of neighbor id lists per node
        :return states: state tensor [batch_size, state_dim]
        """
        batch_size = len(nodes)
        adj_list = self.adj_lists[relation_idx]

        # get node features (detached — policy gradients should not flow into features)
        node_features = self.features(torch.LongTensor(nodes).to(self.device)).detach()

        # confidence from label predictor (requires EnhancedLabelPredictor)
        # clamp scores to prevent extreme softmax values that cause 0 * -inf = NaN
        clamped_scores = label_scores.detach().clamp(-20, 20)
        probs = F.softmax(clamped_scores, dim=1)
        log_probs = F.log_softmax(clamped_scores, dim=1)
        entropy = -(probs * log_probs).sum(dim=1, keepdim=True)
        max_entropy = math.log(label_scores.size(1))
        confidence = 1.0 - entropy / max_entropy  # [batch, 1]
        confidence = confidence.clamp(0.0, 1.0)  # ensure valid range

        # per-node structural statistics
        degrees = torch.zeros(batch_size, 1, device=self.device)
        mean_dists = torch.zeros(batch_size, 1, device=self.device)
        overlaps = torch.zeros(batch_size, 1, device=self.device)
        feat_vars = torch.zeros(batch_size, 1, device=self.device)

        for i, node in enumerate(nodes):
            neighs = neighs_list[i]
            num_neighs = len(neighs)

            # normalized degree
            degrees[i] = num_neighs / self.max_degrees[relation_idx]

            # mean L1 distance to neighbors
            if num_neighs > 0 and i < len(neigh_scores_list):
                neigh_score = neigh_scores_list[i]
                center_score = label_scores[i][0].detach().expand(neigh_score.size(0))
                dists = torch.abs(center_score - neigh_score[:, 0].detach())
                dist_mean = dists.mean()
                mean_dists[i] = dist_mean if not torch.isnan(dist_mean) else 0.0

            # Jaccard overlap with homogeneous graph
            node_int = int(node)
            rel_neighs = set(adj_list.get(node_int, set()))
            homo_neighs = set(self.homo_adj.get(node_int, set()))
            if len(rel_neighs) > 0 or len(homo_neighs) > 0:
                intersection = len(rel_neighs & homo_neighs)
                union = len(rel_neighs | homo_neighs)
                overlaps[i] = intersection / max(union, 1)

            # mean feature variance of neighbors (need >= 2 for meaningful variance)
            if num_neighs > 1:
                neigh_feats = self.features(torch.LongTensor(neighs).to(self.device))
                feat_vars[i] = neigh_feats.var(dim=0).mean()

        # concatenate all state components: [x_v, conf, deg, mean_dist, overlap, feat_var]
        components = [
            node_features,     # [batch, feat_dim]
            confidence,        # [batch, 1]
            degrees,           # [batch, 1]
            mean_dists,        # [batch, 1]
            overlaps,          # [batch, 1]
            feat_vars,         # [batch, 1]
        ]

        # LLM semantic embedding projection (optional, v1)
        if self.llm_embeddings is not None and self.llm_projector is not None:
            node_indices = torch.LongTensor(nodes).to(self.llm_embeddings.device)
            llm_raw = self.llm_embeddings[node_indices]          # [batch, 384]
            llm_proj = self.llm_projector(llm_raw.to(self.device))  # [batch, projection_dim]
            components.append(llm_proj)

        # v2 enrichment: graph structural features + Claude reasoning scores
        if self.enrichment_tensor is not None and self.enrichment_projector is not None:
            node_indices = torch.LongTensor(nodes).to(self.enrichment_tensor.device)
            enrich_raw = self.enrichment_tensor[node_indices]
            enrich_proj = self.enrichment_projector(enrich_raw.to(self.device))
            components.append(enrich_proj)

        state = torch.cat(components, dim=1)

        # replace any remaining NaN with 0 to prevent downstream crashes
        state = torch.nan_to_num(state, nan=0.0)

        return state


class SafeLgamma(torch.autograd.Function):
    """lgamma computed on CPU to avoid CUDA NVRTC JIT compilation errors.

    Forward: lgamma(x) on CPU
    Backward: digamma(x) on CPU (since d/dx lgamma(x) = digamma(x))
    Results are placed back on the original device for seamless GPU integration.
    """

    @staticmethod
    def forward(ctx, x):
        x_cpu = x.detach().cpu()
        result = torch.lgamma(x_cpu)
        ctx.save_for_backward(x_cpu)
        ctx.orig_device = x.device
        return result.to(x.device)

    @staticmethod
    def backward(ctx, grad_output):
        x_cpu, = ctx.saved_tensors
        return grad_output * torch.digamma(x_cpu).to(ctx.orig_device)


def _beta_log_prob(alpha, beta, t):
    """Manual Beta log_prob with gradients through alpha/beta via SafeLgamma.

    log p(t | α, β) = (α-1)·log(t) + (β-1)·log(1-t) - lgamma(α) - lgamma(β) + lgamma(α+β)

    :param alpha: Beta distribution alpha parameter (has grad)
    :param beta: Beta distribution beta parameter (has grad)
    :param t: sampled threshold values (detached, no grad needed for REINFORCE)
    :return: log probability with gradients flowing through alpha and beta
    """
    safe_lgamma = SafeLgamma.apply
    log_norm = safe_lgamma(alpha) + safe_lgamma(beta) - safe_lgamma(alpha + beta)
    return (alpha - 1) * torch.log(t) + (beta - 1) * torch.log(1 - t) - log_norm


class PolicyNetwork(nn.Module):
    """
    Policy network that outputs per-node, per-relation filtering thresholds.

    During training, thresholds are sampled from a Beta distribution for exploration.
    During inference, the deterministic mean α/(α+β) is used.

    Architecture: MLP → (α, β) parameters of Beta distribution → threshold ∈ (0, 1)
    """

    def __init__(self, state_dim, hidden_dim=64, num_relations=3,
                 relation_biases=None):
        """
        :param state_dim: dimension of state vector (feat_dim + 5)
        :param hidden_dim: hidden layer size
        :param num_relations: number of relations
        :param relation_biases: optional per-relation bias initialization from LLM priors
        """
        super(PolicyNetwork, self).__init__()
        self.num_relations = num_relations

        # shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )

        # per-relation heads outputting (alpha, beta) for Beta distribution
        self.relation_heads = nn.ModuleList([
            nn.Linear(hidden_dim // 2, 2) for _ in range(num_relations)
        ])

        # initialize biases from LLM priors if available
        if relation_biases is not None:
            for r, bias_val in enumerate(relation_biases):
                if r < num_relations:
                    # higher bias → lower default threshold → more aggressive filtering
                    # map camouflage_risk to output space
                    with torch.no_grad():
                        # bias alpha lower for high-risk relations
                        self.relation_heads[r].bias[0] = 1.0 - bias_val  # alpha
                        self.relation_heads[r].bias[1] = 1.0 + bias_val  # beta

        # stored log probs for policy gradient
        self._log_probs = []
        self._thresholds = []

    def forward(self, state, relation_idx, deterministic=False):
        """
        :param state: state tensor [batch_size, state_dim]
        :param relation_idx: which relation to compute thresholds for
        :param deterministic: if True, return mean instead of sampling
        :return thresholds: per-node thresholds [batch_size]
        :return log_probs: log probabilities of sampled thresholds [batch_size]
        """
        h = self.shared(state)
        params = self.relation_heads[relation_idx](h)

        # softplus to ensure positive alpha, beta (add small constant for stability)
        alpha = F.softplus(params[:, 0]) + 0.5
        beta = F.softplus(params[:, 1]) + 0.5

        # guard against NaN from upstream — replace with safe default (uniform Beta(1,1))
        alpha = torch.where(torch.isnan(alpha) | torch.isinf(alpha),
                            torch.ones_like(alpha), alpha)
        beta = torch.where(torch.isnan(beta) | torch.isinf(beta),
                           torch.ones_like(beta), beta)

        if deterministic or not self.training:
            thresholds = alpha / (alpha + beta)  # mean of Beta
            t_clamped = thresholds.detach().clamp(1e-6, 1 - 1e-6)
            log_probs = _beta_log_prob(alpha, beta, t_clamped)
        else:
            # sample on CPU (REINFORCE doesn't need grad through samples)
            with torch.no_grad():
                dist = Beta(alpha.detach().cpu(), beta.detach().cpu())
                t_sampled = dist.rsample().clamp(1e-3, 0.999).to(alpha.device)

            # log_prob via SafeLgamma — gradients flow through alpha/beta
            log_probs = _beta_log_prob(alpha, beta, t_sampled)

            thresholds = t_sampled

        return thresholds, log_probs

    def reset_episode(self):
        """Reset stored log probs and thresholds for a new batch."""
        self._log_probs = []
        self._thresholds = []

    def store_action(self, log_probs, thresholds):
        """Store log probs and thresholds for policy gradient computation."""
        self._log_probs.append(log_probs)
        self._thresholds.append(thresholds)

    def get_policy_loss(self, reward):
        """
        Compute REINFORCE policy gradient loss.

        loss = -E[R * sum_r(log π(t_r | s))]

        :param reward: scalar reward for this batch
        :return loss: policy gradient loss
        """
        if not self._log_probs:
            return torch.tensor(0.0, device=next(self.parameters()).device)

        # sum log probs across relations
        total_log_prob = torch.stack([lp.mean() for lp in self._log_probs]).sum()
        loss = -reward * total_log_prob
        return loss


class ShapedRewardComputer:
    """
    Computes shaped rewards for the policy network.

    R = w1 * clamp(Δdist / (prev_dist + ε), -1, 1)    # distance improvement
      + w2 * (batch_acc - baseline_acc)                  # accuracy signal
      - w3 * mean(|threshold - 0.5|)                     # regularization

    Uses exponential moving average for baseline accuracy.
    """

    def __init__(self, w1=0.5, w2=0.3, w3=0.2, ema_decay=0.95):
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.ema_decay = ema_decay
        self.baseline_acc = None
        self.prev_avg_dist = None
        self.reward_log = []

    def compute_reward(self, avg_dist, batch_acc, thresholds):
        """
        :param avg_dist: average neighbor distance for this batch (float)
        :param batch_acc: classification accuracy for this batch (float)
        :param thresholds: list of threshold tensors per relation
        :return reward: scalar reward value
        """
        # distance improvement component
        if self.prev_avg_dist is not None:
            delta_dist = self.prev_avg_dist - avg_dist
            dist_reward = max(-1.0, min(1.0, delta_dist / (abs(self.prev_avg_dist) + 1e-8)))
        else:
            dist_reward = 0.0
        self.prev_avg_dist = avg_dist

        # accuracy improvement component
        if self.baseline_acc is not None:
            acc_reward = batch_acc - self.baseline_acc
            self.baseline_acc = self.ema_decay * self.baseline_acc + (1 - self.ema_decay) * batch_acc
        else:
            self.baseline_acc = batch_acc
            acc_reward = 0.0

        # regularization: penalize extreme thresholds
        if thresholds:
            all_t = torch.cat([t.detach() for t in thresholds])
            reg_penalty = (all_t - 0.5).abs().mean().item()
        else:
            reg_penalty = 0.0

        reward = self.w1 * dist_reward + self.w2 * acc_reward - self.w3 * reg_penalty

        self.reward_log.append({
            'reward': reward,
            'dist_reward': dist_reward,
            'acc_reward': acc_reward,
            'reg_penalty': reg_penalty,
            'avg_dist': avg_dist,
            'batch_acc': batch_acc,
        })

        return reward

    def reset(self):
        """Reset for new training run."""
        self.baseline_acc = None
        self.prev_avg_dist = None
        self.reward_log = []

    def reset_epoch(self):
        """Reset distance tracking at epoch boundary (preserves EMA baseline)."""
        self.prev_avg_dist = None


class LLMPriorLoader:
    """
    Loads pre-generated LLM domain priors for warm-starting the policy network.

    Expected JSON format:
    {
        "relations": [
            {"name": "R-U-R", "importance": 0.8, "camouflage_risk": 0.6},
            {"name": "R-T-R", "importance": 0.5, "camouflage_risk": 0.3},
            ...
        ]
    }
    """

    def __init__(self, priors_file=None):
        self.priors = None
        if priors_file and os.path.exists(priors_file):
            with open(priors_file, 'r') as f:
                self.priors = json.load(f)
            logger.info(f'Loaded LLM priors from {priors_file}')
        elif priors_file:
            logger.warning(f'LLM priors file not found: {priors_file}')

    def get_relation_biases(self):
        """Get per-relation bias values for policy network initialization."""
        if self.priors is None:
            return None
        return [r['camouflage_risk'] for r in self.priors['relations']]

    def get_gamma_init(self):
        """Get per-relation gamma initialization for multi-view distance."""
        if self.priors is None:
            return None
        return [r['importance'] for r in self.priors['relations']]


class MultiViewDistance:
    """
    Computes multi-view neighbor distance combining label-aware and structural signals.

    d(v, u, r) = γ_r * |s_v - s_u|_1  +  (1 - γ_r) * (1 - Jaccard(v, u, r))

    Where γ_r is a learnable per-relation mixing parameter.
    """

    def __init__(self, num_relations, gamma_init=0.7, adj_lists=None):
        """
        :param num_relations: number of relations
        :param gamma_init: initial gamma value (or list per relation)
        :param adj_lists: adjacency lists for structural overlap computation
        """
        self.adj_lists = adj_lists
        if isinstance(gamma_init, list):
            self.gammas = gamma_init
        else:
            self.gammas = [gamma_init] * num_relations
        # Cache structural Jaccard distances (graph is static)
        self._jaccard_cache = {}

    def _get_struct_dist(self, center_node, neigh_id, relation_idx):
        """Cached structural Jaccard distance for a node pair."""
        key = (center_node, neigh_id, relation_idx)
        if key in self._jaccard_cache:
            return self._jaccard_cache[key]
        adj_list = self.adj_lists[relation_idx]
        center_neighs = adj_list.get(center_node, set())
        neigh_neighs = adj_list.get(neigh_id, set())
        if len(center_neighs) > 0 or len(neigh_neighs) > 0:
            jaccard = len(center_neighs & neigh_neighs) / max(len(center_neighs | neigh_neighs), 1)
        else:
            jaccard = 0.0
        dist = 1.0 - jaccard
        self._jaccard_cache[key] = dist
        return dist

    def compute_distance(self, center_score, neigh_scores, neighs_indices,
                         center_node, relation_idx):
        """
        Compute multi-view distance between center node and its neighbors.

        :param center_score: label score of center node [1]
        :param neigh_scores: label scores of neighbors [num_neigh, 1]
        :param neighs_indices: neighbor node ids
        :param center_node: center node id
        :param relation_idx: which relation
        :return distances: multi-view distances [num_neigh]
        """
        gamma = self.gammas[relation_idx]

        # L1 label-aware distance (existing)
        label_dist = torch.abs(center_score - neigh_scores).squeeze()
        if label_dist.dim() == 0:
            label_dist = label_dist.unsqueeze(0)

        # structural Jaccard distance (cached)
        if self.adj_lists is not None:
            center_node_int = int(center_node)
            struct_dists = [self._get_struct_dist(center_node_int, int(n), relation_idx)
                           for n in neighs_indices]
            struct_dist = torch.tensor(struct_dists, device=label_dist.device, dtype=label_dist.dtype)
        else:
            struct_dist = torch.zeros_like(label_dist)

        # combine
        multi_dist = gamma * label_dist + (1 - gamma) * struct_dist
        return multi_dist
