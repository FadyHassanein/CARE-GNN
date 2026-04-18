"""
Embedding/Feature Projector for CAPN state enrichment.

Projects input features to a lower-dimensional space for concatenation
into the CAPN policy state vector. Supports:
  - v1: 384-dim sentence embeddings (legacy)
  - v2: 26-dim graph structural features, 6-dim risk scores, or 32-dim combined

Projector modes:
  - 'mlp': 2-layer MLP (original, best for high-dim inputs like 384)
  - 'small_mlp': smaller MLP (for medium inputs 8-32 dim)
  - 'linear': single linear layer (for small inputs <= 8 dim)
  - 'auto': selects mode based on input_dim
"""

import torch
import torch.nn as nn


class LLMProjector(nn.Module):
    """Projects input features to a compact representation for the policy state."""

    def __init__(self, input_dim=384, projection_dim=16, dropout=0.1, mode='auto'):
        super(LLMProjector, self).__init__()

        if mode == 'auto':
            if input_dim <= 8:
                mode = 'linear'
            elif input_dim <= 32:
                mode = 'small_mlp'
            elif input_dim >= 128:
                mode = 'deep_mlp'
            else:
                mode = 'mlp'

        self.mode = mode

        if mode == 'linear':
            self.mlp = nn.Linear(input_dim, projection_dim)
        elif mode == 'small_mlp':
            hidden_dim = max(12, input_dim * 2)
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, projection_dim),
            )
        elif mode == 'deep_mlp':
            # 3-layer projector for high-dim inputs (e.g. 384-dim sentence embeddings)
            h1 = min(input_dim, 256)
            h2 = min(max(projection_dim * 2, 64), 128)
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, h1),
                nn.LayerNorm(h1),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(h1, h2),
                nn.LayerNorm(h2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(h2, projection_dim),
                nn.LayerNorm(projection_dim),
            )
        else:  # 'mlp'
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
