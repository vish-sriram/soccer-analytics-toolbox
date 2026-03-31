"""
Spatial-Temporal Value Network (STVN)

Architecture
------------
For each frame in a possession chain:
  1. Frame Encoder (shared weights):
       GATConv × 2  →  per-node embeddings
       global mean + max pool  →  frame embedding  [d_frame]

  2. Temporal Encoder:
       GRU over the sequence of frame embeddings  [d_hidden]
       Hidden state h_t encodes "accumulated value of chain up to frame t"

  3. Value Head:
       MLP(h_T)  →  scalar logit  →  P(chain produces a shot)

Per-frame value:
  The GRU hidden state at each step h_t is a learned representation of
  the chain value up to that point.  Δ(h_t) = h_t - h_{t-1} quantifies
  the marginal spatial value added by action t — a deep-learning analogue
  of VAEP that is sensitive to player geometry and chain history.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Batch, Data


NODE_FEATURES = 11   # x, y, vx, vy, ax, ay, is_actor, is_teammate, is_gk, dist, angle


class FrameEncoder(nn.Module):
    """
    Encodes a single freeze-frame graph into a fixed-size embedding.
    Weights are shared across all frames in a chain.
    """

    def __init__(self, node_features: int = NODE_FEATURES,
                 hidden_dim: int = 48, heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.dropout = dropout

        self.gat1 = GATConv(node_features,  hidden_dim,
                             heads=heads, dropout=dropout, concat=True)
        self.gat2 = GATConv(hidden_dim * heads, hidden_dim,
                             heads=1,     dropout=dropout, concat=False)

        self.frame_dim = hidden_dim * 2   # mean + max pool

    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor,
                batch: torch.Tensor) -> torch.Tensor:
        x = F.elu(self.gat1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.gat2(x, edge_index))
        return torch.cat([global_mean_pool(x, batch),
                          global_max_pool(x, batch)], dim=-1)  # [B, frame_dim]


class STVN(nn.Module):
    """
    Spatial-Temporal Value Network.

    Takes a variable-length possession chain (list of PyG Data objects)
    and returns:
      - logit:  scalar prediction (P chain produces a shot)
      - h_seq:  GRU hidden states at each step [T, d_hidden]
                (used for per-frame value attribution)
    """

    def __init__(self, node_features: int = NODE_FEATURES,
                 hidden_dim: int = 48, gru_dim: int = 64,
                 heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.frame_enc  = FrameEncoder(node_features, hidden_dim, heads, dropout)
        self.gru        = nn.GRU(self.frame_enc.frame_dim, gru_dim,
                                  num_layers=2, batch_first=True,
                                  dropout=dropout)
        self.dropout    = nn.Dropout(dropout)
        self.head       = nn.Sequential(
            nn.Linear(gru_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )
        self.gru_dim = gru_dim

    def encode_chains_batched(self, chains: list[list[Data]]) -> list[torch.Tensor]:
        """
        Encode all frames across all chains in a single batched GATConv pass.

        Flattens every frame from every chain into one big PyG Batch,
        runs the frame encoder once, then splits results back into
        per-chain sequences.  This is the key speedup: O(1) GATConv
        calls instead of O(sum of chain lengths).

        Returns list of [T_i, frame_dim] tensors, one per chain.
        """
        lengths = [len(c) for c in chains]
        all_graphs = [g for chain in chains for g in chain]

        big_batch = Batch.from_data_list(all_graphs)
        frame_embs = self.frame_enc(big_batch.x,
                                    big_batch.edge_index,
                                    big_batch.batch)   # [sum(T_i), frame_dim]

        # Split back into per-chain sequences
        return torch.split(frame_embs, lengths)        # tuple of [T_i, frame_dim]

    def forward(self, chains: list[list[Data]],
                return_hidden: bool = False):
        """
        chains: list of possession chains, each a list of PyG Data objects.

        Encodes all frames in one batched GATConv pass, then runs GRU
        per-chain (variable length — no padding needed).
        """
        chain_seqs = self.encode_chains_batched(chains)  # list of [T_i, frame_dim]

        logits, hidden_seqs = [], []
        for seq in chain_seqs:
            out, _ = self.gru(seq.unsqueeze(0))          # [1, T, gru_dim]
            h_seq  = out.squeeze(0)                       # [T, gru_dim]
            logit  = self.head(self.dropout(h_seq[-1]))   # [1]
            logits.append(logit)
            if return_hidden:
                hidden_seqs.append(h_seq.detach())

        logits = torch.cat(logits, dim=0)                 # [B]

        if return_hidden:
            return logits, hidden_seqs
        return logits

    def predict_proba(self, chains: list[list[Data]]) -> torch.Tensor:
        with torch.no_grad():
            return torch.sigmoid(self.forward(chains))

    def frame_values(self, chain: list[Data]) -> torch.Tensor:
        """
        Per-frame value for a single chain: [T] tensor.
        Δvalue[t] = value[t] - value[t-1] = marginal spatial value of action t.
        """
        self.eval()
        with torch.no_grad():
            seqs = self.encode_chains_batched([chain])
            out, _ = self.gru(seqs[0].unsqueeze(0))
            h_seq  = out.squeeze(0)
            return torch.sigmoid(self.head(h_seq).squeeze(-1))
