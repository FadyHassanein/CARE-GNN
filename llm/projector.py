"""
Deliverable 4A: LLM Embedding Projector.

Projects 384-dim sentence embeddings to a lower-dimensional space (default 16)
for concatenation into CAPN's state vector.
"""

import torch
import torch.nn as nn


class LLMProjector(nn.Module):
    """Projects sentence embeddings to a compact representation for the policy state."""

    def __init__(self, input_dim=384, projection_dim=16, dropout=0.1):
        super(LLMProjector, self).__init__()
        hidden_dim = 64
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, projection_dim),
        )

    def forward(self, x):
        """
        :param x: sentence embeddings [batch_size, input_dim]
        :return: projected embeddings [batch_size, projection_dim]
        """
        return self.mlp(x)
