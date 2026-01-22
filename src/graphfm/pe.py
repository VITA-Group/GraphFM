from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import torch


@dataclass(frozen=True)
class PEConfig:
    kind: str  # "eig", "proj", "spe"
    k: int
    m: int = 0  # readout dim for proj/spe
    spe_alpha: float = 10.0
    spe_tau: float = 0.0
    seed: int = 0


def _eigh_sorted(delta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    evals, evecs = np.linalg.eigh(delta)
    idx = np.argsort(evals)
    return evals[idx], evecs[:, idx]


def eig_pe(delta: np.ndarray, k: int) -> np.ndarray:
    n = delta.shape[0]
    k = min(k, n)
    _, evecs = _eigh_sorted(delta)
    evecs = evecs[:, :k] * np.sqrt(float(n))
    return evecs


def proj_pe(delta: np.ndarray, k: int, m: int, seed: int = 0) -> np.ndarray:
    if m <= 0:
        raise ValueError("proj_pe requires m > 0.")
    n = delta.shape[0]
    k = min(k, n)
    _, evecs = _eigh_sorted(delta)
    u = evecs[:, :k] * np.sqrt(float(n))
    rng = np.random.default_rng(seed)
    r = rng.normal(size=(n, m))
    r = r / (np.linalg.norm(r, axis=0, keepdims=True) + 1e-12) * np.sqrt(float(n))
    proj = (u @ (u.T @ r)) / float(n)
    return proj


def spe_pe(
    delta: np.ndarray,
    k: int,
    m: int,
    spe_alpha: float,
    spe_tau: float,
    seed: int = 0,
) -> np.ndarray:
    if m <= 0:
        raise ValueError("spe_pe requires m > 0.")
    n = delta.shape[0]
    k = min(k, n)
    evals, evecs = _eigh_sorted(delta)
    evals = evals[:k]
    evecs = evecs[:, :k]
    g = 1.0 / (1.0 + np.exp(-spe_alpha * (evals - spe_tau)))
    g = np.diag(g)
    rng = np.random.default_rng(seed)
    r = rng.normal(size=(n, m))
    r = r / (np.linalg.norm(r, axis=0, keepdims=True) + 1e-12) * np.sqrt(float(n))
    out = evecs @ g @ evecs.T @ r
    return out


def compute_pe(delta: np.ndarray, cfg: PEConfig) -> np.ndarray:
    if cfg.kind == "eig":
        return eig_pe(delta, cfg.k)
    if cfg.kind == "proj":
        return proj_pe(delta, cfg.k, cfg.m, seed=cfg.seed)
    if cfg.kind == "spe":
        return spe_pe(delta, cfg.k, cfg.m, cfg.spe_alpha, cfg.spe_tau, seed=cfg.seed)
    raise ValueError(f"Unknown PE kind: {cfg.kind}")


# ============ GPU Batch Versions ============


def eig_pe_batch(deltas: torch.Tensor, k: int) -> torch.Tensor:
    """Batch eigenvalue PE on GPU.

    Args:
        deltas: (B, n, n) batch of normalized shift operators
        k: number of eigenvectors to keep

    Returns:
        (B, n, k) batch of positional encodings
    """
    B, n, _ = deltas.shape
    k = min(k, n)
    _, evecs = torch.linalg.eigh(deltas)  # (B, n, n)
    evecs_k = evecs[:, :, :k] * (n ** 0.5)  # (B, n, k)
    return evecs_k


def proj_pe_batch(
    deltas: torch.Tensor,
    k: int,
    m: int,
    seed: int = 0,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Batch projection PE on GPU.

    Args:
        deltas: (B, n, n) batch of normalized shift operators
        k: number of eigenvectors
        m: projection dimension
        seed: random seed (used if generator is None)
        generator: optional torch Generator for reproducibility

    Returns:
        (B, n, m) batch of positional encodings
    """
    if m <= 0:
        raise ValueError("proj_pe_batch requires m > 0.")
    B, n, _ = deltas.shape
    k = min(k, n)
    device = deltas.device
    dtype = deltas.dtype

    _, evecs = torch.linalg.eigh(deltas)  # (B, n, n)
    u = evecs[:, :, :k] * (n ** 0.5)  # (B, n, k)

    if generator is None:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

    r = torch.randn(B, n, m, device=device, dtype=dtype, generator=generator)
    r = r / (torch.norm(r, dim=1, keepdim=True) + 1e-12) * (n ** 0.5)

    # proj = u @ (u.T @ r) / n  =>  (B, n, k) @ (B, k, n) @ (B, n, m) / n
    proj = torch.bmm(u, torch.bmm(u.transpose(1, 2), r)) / float(n)  # (B, n, m)
    return proj


def spe_pe_batch(
    deltas: torch.Tensor,
    k: int,
    m: int,
    spe_alpha: float = 10.0,
    spe_tau: float = 0.0,
    seed: int = 0,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Batch SPE on GPU.

    Args:
        deltas: (B, n, n) batch of normalized shift operators
        k: number of eigenvectors
        m: output dimension
        spe_alpha: sigmoid steepness
        spe_tau: sigmoid threshold
        seed: random seed (used if generator is None)
        generator: optional torch Generator for reproducibility

    Returns:
        (B, n, m) batch of positional encodings
    """
    if m <= 0:
        raise ValueError("spe_pe_batch requires m > 0.")
    B, n, _ = deltas.shape
    k = min(k, n)
    device = deltas.device
    dtype = deltas.dtype

    evals, evecs = torch.linalg.eigh(deltas)  # (B, n), (B, n, n)
    evals_k = evals[:, :k]  # (B, k)
    evecs_k = evecs[:, :, :k]  # (B, n, k)

    # g = sigmoid filter: (B, k)
    g = 1.0 / (1.0 + torch.exp(-spe_alpha * (evals_k - spe_tau)))

    if generator is None:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

    r = torch.randn(B, n, m, device=device, dtype=dtype, generator=generator)
    r = r / (torch.norm(r, dim=1, keepdim=True) + 1e-12) * (n ** 0.5)

    # out = evecs @ diag(g) @ evecs.T @ r
    # = (B, n, k) @ (B, k, k) @ (B, k, n) @ (B, n, m)
    g_diag = torch.diag_embed(g)  # (B, k, k)
    temp = torch.bmm(evecs_k, g_diag)  # (B, n, k)
    temp = torch.bmm(temp, evecs_k.transpose(1, 2))  # (B, n, n)
    out = torch.bmm(temp, r)  # (B, n, m)
    return out


def compute_pe_batch(deltas: torch.Tensor, cfg: PEConfig, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Compute PE for a batch of graphs on GPU.

    Args:
        deltas: (B, n, n) batch of normalized shift operators
        cfg: PE configuration
        generator: optional torch Generator for reproducibility

    Returns:
        (B, n, k) or (B, n, m) batch of positional encodings
    """
    if cfg.kind == "eig":
        return eig_pe_batch(deltas, cfg.k)
    if cfg.kind == "proj":
        return proj_pe_batch(deltas, cfg.k, cfg.m, seed=cfg.seed, generator=generator)
    if cfg.kind == "spe":
        return spe_pe_batch(deltas, cfg.k, cfg.m, cfg.spe_alpha, cfg.spe_tau, seed=cfg.seed, generator=generator)
    raise ValueError(f"Unknown PE kind: {cfg.kind}")
