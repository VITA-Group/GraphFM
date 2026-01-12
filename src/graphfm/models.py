from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, sizes: List[int], dropout: float = 0.0) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeepSets(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.phi = MLP([in_dim, hidden, hidden], dropout=dropout)
        self.rho = MLP([hidden, hidden, out_dim], dropout=dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        h = self.phi(tokens)
        pooled = h.mean(dim=0, keepdim=True)
        return self.rho(pooled).squeeze(0)


class DegreeHistMLP(nn.Module):
    def __init__(self, bins: int, hidden: int, out_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.mlp = MLP([bins, hidden, out_dim], dropout=dropout)

    def forward(self, degree_hist: torch.Tensor) -> torch.Tensor:
        return self.mlp(degree_hist)


class GINLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, eps: float = 0.0) -> None:
        super().__init__()
        self.eps = nn.Parameter(torch.tensor(eps))
        self.mlp = MLP([in_dim, out_dim, out_dim])

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        agg = adj @ x
        return self.mlp((1.0 + self.eps) * x + agg)


class GIN(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, layers: int = 3) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        dims = [in_dim] + [hidden] * (layers - 1) + [hidden]
        for i in range(layers):
            self.layers.append(GINLayer(dims[i], dims[i + 1]))
        self.readout = MLP([hidden, hidden, out_dim])

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = torch.relu(layer(h, adj))
        pooled = h.mean(dim=0, keepdim=True)
        return self.readout(pooled).squeeze(0)
