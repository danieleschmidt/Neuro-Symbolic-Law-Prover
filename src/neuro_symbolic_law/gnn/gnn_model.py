"""
GNN model for legal entity graph processing.

Architecture: simple 2-layer message-passing GNN (mean aggregation).
No exotic dependencies — pure PyTorch, no PyG required.

Input:  node feature matrix X  [N, in_features]
        edge_index              [2, E]
Output: node embeddings         [N, out_features]

The GNN learns (or in the untrained/demo case, applies random weights) to
produce dense embeddings for each entity node. These embeddings are then
passed to the GNNPropertyExtractor which projects them back to scalar
property scores in [0, 1].
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GNNConfig:
    in_features: int = 14        # 5 type one-hot + 9 boolean properties
    hidden_features: int = 32
    out_features: int = 16
    num_layers: int = 2
    dropout: float = 0.0
    aggregation: str = "mean"    # 'mean' | 'sum' | 'max'


class MessagePassingLayer(nn.Module):
    """
    Single message-passing layer.

    h_i^(l+1) = σ( W_self · h_i^(l)  +  W_neigh · AGG_{j∈N(i)} h_j^(l) )
    """

    def __init__(self, in_dim: int, out_dim: int, aggregation: str = "mean") -> None:
        super().__init__()
        self.aggregation = aggregation
        self.linear_self = nn.Linear(in_dim, out_dim)
        self.linear_neigh = nn.Linear(in_dim, out_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        for lin in (self.linear_self, self.linear_neigh):
            nn.init.kaiming_uniform_(lin.weight, a=math.sqrt(5))
            nn.init.zeros_(lin.bias)

    def forward(
        self,
        x: torch.Tensor,           # [N, in_dim]
        edge_index: torch.Tensor,  # [2, E]
    ) -> torch.Tensor:             # [N, out_dim]
        N = x.size(0)

        if edge_index.size(1) == 0:
            # No edges: identity pass-through
            return F.relu(self.linear_self(x))

        src, dst = edge_index[0], edge_index[1]  # each [E]

        # Gather source features
        src_feats = x[src]  # [E, in_dim]

        # Aggregate neighbor features into each destination node
        agg = torch.zeros(N, x.size(1), device=x.device, dtype=x.dtype)

        if self.aggregation == "mean":
            # Scatter-add then divide by degree
            agg.scatter_add_(0, dst.unsqueeze(1).expand_as(src_feats), src_feats)
            degree = torch.zeros(N, device=x.device, dtype=x.dtype)
            degree.scatter_add_(0, dst, torch.ones(dst.size(0), device=x.device, dtype=x.dtype))
            degree = degree.clamp(min=1).unsqueeze(1)
            agg = agg / degree

        elif self.aggregation == "sum":
            agg.scatter_add_(0, dst.unsqueeze(1).expand_as(src_feats), src_feats)

        elif self.aggregation == "max":
            agg.fill_(-1e9)
            agg.scatter_reduce_(0, dst.unsqueeze(1).expand_as(src_feats),
                                src_feats, reduce="amax", include_self=True)
            agg = agg.clamp(min=0)

        # Combine self + neighbor
        out = self.linear_self(x) + self.linear_neigh(agg)
        return F.relu(out)


class LegalGNN(nn.Module):
    """
    Multi-layer GNN for legal entity graphs.

    Usage (inference only, no training needed for demo)::

        gnn = LegalGNN(GNNConfig())
        node_emb = gnn(node_features, edge_index)   # [N, out_features]
    """

    def __init__(self, config: Optional[GNNConfig] = None) -> None:
        super().__init__()
        self.config = config or GNNConfig()
        cfg = self.config

        dims = (
            [cfg.in_features]
            + [cfg.hidden_features] * (cfg.num_layers - 1)
            + [cfg.out_features]
        )

        self.layers = nn.ModuleList([
            MessagePassingLayer(dims[i], dims[i + 1], cfg.aggregation)
            for i in range(cfg.num_layers)
        ])
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,            # [N, in_features]
        edge_index: torch.Tensor,   # [2, E]
    ) -> torch.Tensor:              # [N, out_features]
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h, edge_index)
            if i < len(self.layers) - 1:
                h = self.dropout(h)
        return h
