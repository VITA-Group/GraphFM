from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .dataset import GraphSample
from .models import DeepSets, DegreeHistMLP, GIN


class GraphDataset(Dataset):
    def __init__(self, samples: List[GraphSample]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> GraphSample:
        return self.samples[idx]


@dataclass
class TrainConfig:
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 1
    device: str = "cpu"
    model: str = "deepsets"  # "deepsets", "degree", "gin"
    hidden: int = 128
    degree_bins: int = 32


def _degree_histogram(adj: np.ndarray, bins: int) -> np.ndarray:
    deg = adj.sum(axis=1)
    hist, _ = np.histogram(deg, bins=bins, range=(deg.min(), deg.max() + 1e-6), density=True)
    return hist.astype(np.float32)


def build_model(config: TrainConfig, in_dim: int, num_classes: int) -> nn.Module:
    if config.model == "deepsets":
        return DeepSets(in_dim=in_dim, hidden=config.hidden, out_dim=num_classes)
    if config.model == "degree":
        return DegreeHistMLP(bins=config.degree_bins, hidden=config.hidden, out_dim=num_classes)
    if config.model == "gin":
        return GIN(in_dim=in_dim, hidden=config.hidden, out_dim=num_classes)
    raise ValueError(f"Unknown model: {config.model}")


def train_classifier(
    train_samples: List[GraphSample],
    val_samples: List[GraphSample],
    num_classes: int,
    config: TrainConfig,
) -> nn.Module:
    device = torch.device(config.device)
    in_dim = train_samples[0].tokens.shape[1]
    model = build_model(config, in_dim=in_dim, num_classes=num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    def run_epoch(samples: List[GraphSample], train: bool) -> float:
        if train:
            model.train()
        else:
            model.eval()
        total = 0.0
        count = 0
        for sample in samples:
            y = torch.tensor(sample.label, dtype=torch.long, device=device)
            if config.model == "degree":
                feats = _degree_histogram(sample.adjacency, config.degree_bins)
                x = torch.tensor(feats, device=device)
                logits = model(x)
            elif config.model == "gin":
                x = torch.tensor(sample.tokens, dtype=torch.float32, device=device)
                adj = torch.tensor(sample.adjacency, dtype=torch.float32, device=device)
                logits = model(x, adj)
            else:
                x = torch.tensor(sample.tokens, dtype=torch.float32, device=device)
                logits = model(x)
            loss = loss_fn(logits.unsqueeze(0), y.unsqueeze(0))
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
            total += loss.item()
            count += 1
        return total / max(count, 1)

    best_state = None
    best_val = float("inf")
    for _ in range(config.epochs):
        run_epoch(train_samples, train=True)
        val_loss = run_epoch(val_samples, train=False)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def evaluate_classifier(
    model: nn.Module,
    samples: List[GraphSample],
    config: TrainConfig,
) -> float:
    device = torch.device(config.device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for sample in samples:
            y = torch.tensor(sample.label, dtype=torch.long, device=device)
            if config.model == "degree":
                feats = _degree_histogram(sample.adjacency, config.degree_bins)
                x = torch.tensor(feats, device=device)
                logits = model(x)
            elif config.model == "gin":
                x = torch.tensor(sample.tokens, dtype=torch.float32, device=device)
                adj = torch.tensor(sample.adjacency, dtype=torch.float32, device=device)
                logits = model(x, adj)
            else:
                x = torch.tensor(sample.tokens, dtype=torch.float32, device=device)
                logits = model(x)
            pred = int(torch.argmax(logits).item())
            correct += int(pred == int(y.item()))
            total += 1
    return 1.0 - float(correct) / max(total, 1)
