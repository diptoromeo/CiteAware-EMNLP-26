"""
models/gcn.py
─────────────────────────────────────────────────────────────────────────────
Graph Convolutional Network for citation graph node classification.
Supports heterogeneous node feature matrices (paper + word/author nodes).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    """
    Single GCN layer: H' = σ(Ã H W + b)
    Supports feature-less mode (uses only W as lookup table) for
    one-hot initialised word/author nodes.
    """

    def __init__(
        self,
        in_features:  int,
        out_features: int,
        dropout:      float = 0.0,
        activation:   nn.Module | None = None,
        bias:         bool = True,
    ):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.activation   = activation
        self.dropout      = nn.Dropout(dropout)

        self.weight = Parameter(torch.empty(in_features, out_features))
        self.bias   = Parameter(torch.zeros(1, out_features)) if bias else None
        self._reset()

    def _reset(self):
        stdv = np.sqrt(6.0 / (self.in_features + self.out_features))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(
        self,
        x:            torch.Tensor,
        adj:          torch.Tensor,
        feature_less: bool = False,
    ) -> torch.Tensor:
        if feature_less:
            support = self.dropout(self.weight)
        else:
            support = torch.mm(self.dropout(x), self.weight)

        out = torch.spmm(adj, support)

        if self.bias is not None:
            out = out + self.bias
        if self.activation is not None:
            out = self.activation(out)
        return out

    def __repr__(self) -> str:
        return f"GraphConvolution({self.in_features} → {self.out_features})"


class GCN(nn.Module):
    """
    Multi-layer GCN.

    Args:
        nfeat    : input feature dimension
        nhid     : hidden dimension
        nclass   : number of output labels (K)
        dropout  : dropout rate
        n_layers : total number of GCN layers (≥2)
    """

    def __init__(
        self,
        nfeat:    int,
        nhid:     int,
        nclass:   int,
        dropout:  float = 0.3,
        n_layers: int   = 2,
    ):
        super().__init__()
        self.n_layers = max(n_layers, 1)
        act           = nn.ReLU()

        layers = []
        if self.n_layers == 1:
            layers.append(GraphConvolution(nfeat, nclass, dropout=dropout))
        else:
            layers.append(GraphConvolution(nfeat, nhid, dropout=dropout, activation=act))
            for _ in range(self.n_layers - 2):
                layers.append(GraphConvolution(nhid, nhid, dropout=dropout, activation=act))
            # final layer — no activation (logits)
            layers.append(GraphConvolution(nhid, nclass, dropout=dropout))

        self.layers = nn.ModuleList(layers)

    def forward(
        self,
        x:   torch.Tensor,
        adj: torch.Tensor,
        feature_less: bool = False,
    ) -> torch.Tensor:
        """
        x   : (N, d) node feature matrix
        adj : (N, N) sparse normalised adjacency
        Returns: (N, K) logits
        """
        for i, layer in enumerate(self.layers):
            fl = feature_less and (i == 0)
            x  = layer(x, adj, feature_less=fl)
        return x
