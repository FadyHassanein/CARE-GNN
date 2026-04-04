"""
Embedding/Feature Projector for CAPN state enrichment.

Projects input features to a lower-dimensional space for concatenation
into the CAPN policy state vector. Supports:
  - v1: 384-dim sentence embeddings (legacy)
  - v2: 26-dim graph structural features, 6-dim risk scores, or 32-dim combined
"""

import torch
import torch.nn as nn


class LLMProjector(nn.Module):
    """Projects input features to a compact representation for the policy state."""

    def __init__(self, input_dim=384, projection_dim=16, dropout=0.1):
        super(LLMProjector, self).__init__()
        hidden_dim = min(max(32, input_dim * 2), 128)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, projection_dim),
        )

    def forward(self, x):
        """
        :param x: input features [batch_size, input_dim]
        :return: projected features [batch_size, projection_dim]
        """
        return self.mlp(x)
