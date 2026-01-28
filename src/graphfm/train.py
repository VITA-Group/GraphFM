from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm

from .dataset import GraphSample
from .models import DeepSets, DegreeHistMLP, GIN
from .pe import PEConfig, compute_pe_batch, build_learnable_pe, eigh_batch_for_learnable


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


def _group_by_size(
    samples: List[GraphSample],
    device: torch.device,
) -> Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Group samples by graph size for batch processing.

    Returns:
        Dict mapping size n -> (deltas, adjs, labels) where:
            deltas: (B, n, n) normalized shift operators
            adjs: (B, n, n) adjacency matrices
            labels: (B,) class labels
    """
    by_size: Dict[int, List[GraphSample]] = defaultdict(list)
    for s in samples:
        by_size[s.delta.shape[0]].append(s)

    result = {}
    for n, group in by_size.items():
        deltas = torch.from_numpy(np.stack([s.delta for s in group])).to(device, dtype=torch.float32)
        adjs = torch.from_numpy(np.stack([s.adjacency for s in group])).to(device, dtype=torch.float32)
        labels = torch.tensor([s.label for s in group], dtype=torch.long, device=device)
        result[n] = (deltas, adjs, labels)
    return result


def train_classifier(
    train_samples: List[GraphSample],
    val_samples: List[GraphSample],
    num_classes: int,
    config: TrainConfig,
    pe_cfg: Optional[PEConfig] = None,
) -> nn.Module:
    """Train a graph classifier.

    Args:
        train_samples: training samples
        val_samples: validation samples
        num_classes: number of classes
        config: training configuration
        pe_cfg: PE configuration for on-the-fly computation (GPU accelerated).
                If None, uses pre-computed tokens from samples.

    Returns:
        For spe_learnable: tuple of (classifier, learnable_pe)
        Otherwise: classifier model
    """
    # Handle learnable PE case - returns tuple
    if pe_cfg is not None and pe_cfg.kind == "spe_learnable":
        return _train_with_learnable_pe(
            train_samples, val_samples, num_classes, config, pe_cfg
        )

    device = torch.device(config.device)

    # Determine input dimension
    if pe_cfg is not None:
        in_dim = pe_cfg.m if pe_cfg.kind in ("proj", "spe") else pe_cfg.k
    else:
        in_dim = train_samples[0].tokens.shape[1]

    model = build_model(config, in_dim=in_dim, num_classes=num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    # Pre-group samples by size for efficient batch PE computation
    if pe_cfg is not None:
        train_grouped = _group_by_size(train_samples, device)
        val_grouped = _group_by_size(val_samples, device)

    def run_epoch_grouped(grouped: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]], train: bool) -> float:
        """Run epoch with on-the-fly GPU batch PE computation."""
        if train:
            model.train()
        else:
            model.eval()
        total_loss = 0.0
        count = 0

        for n, (deltas, adjs, labels) in grouped.items():
            # Compute PE on-the-fly (GPU batch)
            tokens = compute_pe_batch(deltas, pe_cfg)  # (B, n, k) or (B, n, m)

            for i in range(tokens.shape[0]):
                if config.model == "gin":
                    logits = model(tokens[i], adjs[i])
                else:
                    logits = model(tokens[i])
                loss = loss_fn(logits.unsqueeze(0), labels[i].unsqueeze(0))

                if train:
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

                total_loss += loss.item()
                count += 1

        return total_loss / max(count, 1)

    def run_epoch_precomputed(samples: List[GraphSample], train: bool) -> float:
        """Run epoch with pre-computed tokens (original behavior)."""
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
    epoch_pbar = tqdm(range(config.epochs), desc="Training", unit="epoch")

    for epoch in epoch_pbar:
        if pe_cfg is not None:
            train_loss = run_epoch_grouped(train_grouped, train=True)
            val_loss = run_epoch_grouped(val_grouped, train=False)
        else:
            train_loss = run_epoch_precomputed(train_samples, train=True)
            val_loss = run_epoch_precomputed(val_samples, train=False)

        epoch_pbar.set_postfix(train_loss=f"{train_loss:.4f}", val_loss=f"{val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def _train_with_learnable_pe(
    train_samples: List[GraphSample],
    val_samples: List[GraphSample],
    num_classes: int,
    config: TrainConfig,
    pe_cfg: PEConfig,
) -> Tuple[nn.Module, nn.Module]:
    """Train classifier + learnable PE end-to-end.

    Args:
        train_samples: training samples
        val_samples: validation samples
        num_classes: number of classes
        config: training configuration
        pe_cfg: PE configuration (must be spe_learnable)

    Returns:
        Tuple of (classifier, learnable_pe)
    """
    device = torch.device(config.device)

    in_dim = pe_cfg.m
    model = build_model(config, in_dim=in_dim, num_classes=num_classes).to(device)
    learnable_pe = build_learnable_pe(pe_cfg).to(device)

    all_params = list(model.parameters()) + list(learnable_pe.parameters())
    opt = torch.optim.Adam(all_params, lr=config.lr, weight_decay=config.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    # LR scheduling: warmup + cosine annealing
    # warmup_epochs = min(5, config.epochs // 10)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     opt, T_max=max(1, config.epochs - warmup_epochs), eta_min=config.lr * 0.01
    # )

    train_grouped = _group_by_size(train_samples, device)
    val_grouped = _group_by_size(val_samples, device)

    def run_epoch(
        grouped: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        train: bool,
    ) -> float:
        if train:
            model.train()
            learnable_pe.train()
        else:
            model.eval()
            learnable_pe.eval()

        total_loss = 0.0
        count = 0

        for n, (deltas, adjs, labels) in grouped.items():
            # Pre-compute eigendecomposition for the whole group (no grad required here)
            # This is efficient on GPU
            Lambda_all, V_all = eigh_batch_for_learnable(deltas, pe_cfg.k)

            # Create mini-batches
            num_samples = deltas.shape[0]
            batch_size = 32  # Use a reasonable mini-batch size for updates
            
            indices = torch.randperm(num_samples, device=device) if train else torch.arange(num_samples, device=device)

            for start_idx in range(0, num_samples, batch_size):
                end_idx = min(start_idx + batch_size, num_samples)
                batch_idx = indices[start_idx:end_idx]

                # Slice data for this mini-batch
                Lambda_batch = Lambda_all[batch_idx]
                V_batch = V_all[batch_idx]
                adjs_batch = adjs[batch_idx]
                labels_batch = labels[batch_idx]

                # Forward pass through learnable PE (Fresh computation for gradients)
                tokens_list = learnable_pe(Lambda_batch, V_batch)

                # Compute loss for the batch
                batch_loss = torch.tensor(0.0, device=device)
                curr_batch_count = 0

                for i, tokens in enumerate(tokens_list):
                    if config.model == "gin":
                        logits = model(tokens, adjs_batch[i])
                    else:
                        logits = model(tokens)
                    
                    loss = loss_fn(logits.unsqueeze(0), labels_batch[i].unsqueeze(0))
                    batch_loss = batch_loss + loss
                    curr_batch_count += 1

                if train and curr_batch_count > 0:
                    # Average loss over the mini-batch
                    batch_loss = batch_loss / curr_batch_count
                    
                    opt.zero_grad()
                    batch_loss.backward()
                    opt.step()
                    
                    # Log the *summed* loss for reporting to match other logic (or just report avg * count)
                    total_loss += batch_loss.item() * curr_batch_count
                else:
                    total_loss += batch_loss.item()
                
                count += curr_batch_count

        return total_loss / max(count, 1)

    best_model_state = None
    best_pe_state = None
    best_val = float("inf")
    epoch_pbar = tqdm(range(config.epochs), desc="Training (learnable PE)", unit="epoch")

    for epoch in epoch_pbar:
        train_loss = run_epoch(train_grouped, train=True)
        val_loss = run_epoch(val_grouped, train=False)

        epoch_pbar.set_postfix(train_loss=f"{train_loss:.4f}", val_loss=f"{val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            best_model_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_pe_state = {k: v.detach().cpu() for k, v in learnable_pe.state_dict().items()}

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        learnable_pe.load_state_dict(best_pe_state)

    return model, learnable_pe


def evaluate_classifier(
    model: nn.Module,
    samples: List[GraphSample],
    config: TrainConfig,
    pe_cfg: Optional[PEConfig] = None,
    desc: str = "Evaluating",
    learnable_pe: Optional[nn.Module] = None,
) -> float:
    """Evaluate a graph classifier.

    Args:
        model: trained model
        samples: samples to evaluate
        config: training configuration
        pe_cfg: PE configuration for on-the-fly computation (GPU accelerated).
                If None, uses pre-computed tokens from samples.
        desc: progress bar description
        learnable_pe: learnable PE module (required if pe_cfg.kind == "spe_learnable")
    """
    device = torch.device(config.device)
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        if pe_cfg is not None and pe_cfg.kind == "spe_learnable" and learnable_pe is not None:
            # Learnable PE evaluation
            learnable_pe.eval()
            grouped = _group_by_size(samples, device)
            for n, (deltas, adjs, labels) in grouped.items():
                Lambda, V = eigh_batch_for_learnable(deltas, pe_cfg.k)
                tokens_list = learnable_pe(Lambda, V)
                for i, tokens in enumerate(tokens_list):
                    if config.model == "gin":
                        logits = model(tokens, adjs[i])
                    else:
                        logits = model(tokens)
                    pred = int(torch.argmax(logits).item())
                    correct += int(pred == int(labels[i].item()))
                    total += 1
        elif pe_cfg is not None:
            # On-the-fly GPU batch PE computation
            grouped = _group_by_size(samples, device)
            for n, (deltas, adjs, labels) in grouped.items():
                tokens = compute_pe_batch(deltas, pe_cfg)  # (B, n, k)
                for i in range(tokens.shape[0]):
                    if config.model == "gin":
                        logits = model(tokens[i], adjs[i])
                    else:
                        logits = model(tokens[i])
                    pred = int(torch.argmax(logits).item())
                    correct += int(pred == int(labels[i].item()))
                    total += 1
        else:
            # Pre-computed tokens (original behavior)
            for sample in tqdm(samples, desc=desc, unit="sample", leave=False):
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


def evaluate_classifier_by_size(
    model: nn.Module,
    samples: List[GraphSample],
    config: TrainConfig,
    pe_cfg: Optional[PEConfig] = None,
    learnable_pe: Optional[nn.Module] = None,
) -> Dict[int, float]:
    """Evaluate a graph classifier and return error rate per graph size.

    Args:
        model: trained model
        samples: samples to evaluate
        config: training configuration
        pe_cfg: PE configuration for on-the-fly computation (GPU accelerated).
                If None, uses pre-computed tokens from samples.
        learnable_pe: learnable PE module (required if pe_cfg.kind == "spe_learnable")

    Returns:
        Dict mapping graph size -> error rate for that size
    """
    device = torch.device(config.device)
    model.eval()

    # Track correct/total per size
    correct_by_size: Dict[int, int] = defaultdict(int)
    total_by_size: Dict[int, int] = defaultdict(int)

    with torch.no_grad():
        if pe_cfg is not None and pe_cfg.kind == "spe_learnable" and learnable_pe is not None:
            # Learnable PE evaluation
            learnable_pe.eval()
            grouped = _group_by_size(samples, device)
            for n, (deltas, adjs, labels) in grouped.items():
                Lambda, V = eigh_batch_for_learnable(deltas, pe_cfg.k)
                tokens_list = learnable_pe(Lambda, V)
                for i, tokens in enumerate(tokens_list):
                    if config.model == "gin":
                        logits = model(tokens, adjs[i])
                    else:
                        logits = model(tokens)
                    pred = int(torch.argmax(logits).item())
                    correct_by_size[n] += int(pred == int(labels[i].item()))
                    total_by_size[n] += 1
        elif pe_cfg is not None:
            # On-the-fly GPU batch PE computation
            grouped = _group_by_size(samples, device)
            for n, (deltas, adjs, labels) in grouped.items():
                tokens = compute_pe_batch(deltas, pe_cfg)  # (B, n, k)
                for i in range(tokens.shape[0]):
                    if config.model == "gin":
                        logits = model(tokens[i], adjs[i])
                    else:
                        logits = model(tokens[i])
                    pred = int(torch.argmax(logits).item())
                    correct_by_size[n] += int(pred == int(labels[i].item()))
                    total_by_size[n] += 1
        else:
            # Pre-computed tokens (original behavior)
            for sample in samples:
                n = sample.delta.shape[0]
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
                correct_by_size[n] += int(pred == int(y.item()))
                total_by_size[n] += 1

    # Compute error rate per size
    error_by_size = {}
    for n in total_by_size:
        error_by_size[n] = 1.0 - float(correct_by_size[n]) / max(total_by_size[n], 1)
    return error_by_size
