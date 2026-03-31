"""
Graph Attention Network for spatial xG estimation.

Architecture:
  - 2x GATConv layers with multi-head attention
  - Global mean + max pooling to get graph-level embedding
  - MLP head → P(goal)

The attention weights on edges are the interpretability hook:
high attention = the model considers that player relationship
important for estimating goal probability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool


class SpatialXGModel(nn.Module):
    """
    GAT-based spatial xG model.

    Args:
        node_features: Number of input node features (default 7).
        hidden_dim:    Hidden dimension per attention head.
        heads:         Number of attention heads in each GAT layer.
        dropout:       Dropout rate applied to node embeddings.
    """

    def __init__(
        self,
        node_features: int = 7,
        hidden_dim: int = 32,
        heads: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.dropout = dropout

        # Layer 1: node_features → hidden_dim * heads
        self.gat1 = GATConv(
            node_features,
            hidden_dim,
            heads=heads,
            dropout=dropout,
            concat=True,
        )

        # Layer 2: hidden_dim * heads → hidden_dim (single head, concat=False averages)
        self.gat2 = GATConv(
            hidden_dim * heads,
            hidden_dim,
            heads=1,
            dropout=dropout,
            concat=False,
        )

        # Graph-level readout: concat mean + max pooling → 2 * hidden_dim
        readout_dim = 2 * hidden_dim

        # MLP head
        self.mlp = nn.Sequential(
            nn.Linear(readout_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x, edge_index, batch, return_attention: bool = False):
        # GAT layer 1
        if return_attention:
            x, (edge_idx1, attn1) = self.gat1(x, edge_index, return_attention_weights=True)
        else:
            x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # GAT layer 2
        if return_attention:
            x, (edge_idx2, attn2) = self.gat2(x, edge_index, return_attention_weights=True)
        else:
            x = self.gat2(x, edge_index)
        x = F.elu(x)

        # Graph-level pooling
        mean_pool = global_mean_pool(x, batch)
        max_pool = global_max_pool(x, batch)
        graph_emb = torch.cat([mean_pool, max_pool], dim=-1)

        # MLP → logit
        logit = self.mlp(graph_emb).squeeze(-1)

        if return_attention:
            return logit, {"layer1": (edge_idx1, attn1), "layer2": (edge_idx2, attn2)}
        return logit

    def predict_proba(self, x, edge_index, batch):
        """Return P(goal) for each graph in batch."""
        with torch.no_grad():
            logits = self.forward(x, edge_index, batch)
            return torch.sigmoid(logits)
