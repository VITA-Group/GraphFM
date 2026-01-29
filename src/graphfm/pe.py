from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional, List

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class PEConfig:
    kind: str  # "eig", "proj", "spe", "spe_learnable"
    k: int
    m: int = 0  # readout dim for proj/spe/spe_learnable
    spe_alpha: float = 10.0
    spe_tau: float = 0.0
    seed: int = 0
    # Learnable SPE parameters
    spe_learnable_psi_hidden: int = 32
    spe_learnable_phi_hidden: int = 64


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


def _regularize_for_eigh(deltas: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Add small regularization to diagonal for numerical stability.

    Args:
        deltas: (B, n, n) batch of matrices
        eps: regularization value to add to diagonal

    Returns:
        Regularized matrices (B, n, n)
    """
    B, n, _ = deltas.shape
    eye = torch.eye(n, device=deltas.device, dtype=deltas.dtype).unsqueeze(0)
    return deltas + eps * eye


def eig_pe_batch(deltas: torch.Tensor, k: int) -> torch.Tensor:
    """Batch eigenvalue PE on GPU.

    Args:
        deltas: (B, n, n) batch of normalized shift operators
        k: number of eigenvectors to keep

    Returns:
        (B, n, k) batch of positional encodings (padded with zeros if n < k)
    """
    B, n, _ = deltas.shape
    k_eff = min(k, n)
    deltas_reg = _regularize_for_eigh(deltas)
    _, evecs = torch.linalg.eigh(deltas_reg)  # (B, n, n)
    evecs_k = evecs[:, :, :k_eff] * (n ** 0.5)  # (B, n, k_eff)
    # Pad with zeros if n < k
    if k_eff < k:
        pad = torch.zeros(B, n, k - k_eff, device=deltas.device, dtype=deltas.dtype)
        evecs_k = torch.cat([evecs_k, pad], dim=2)
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
    k_eff = min(k, n)
    device = deltas.device
    dtype = deltas.dtype

    deltas_reg = _regularize_for_eigh(deltas)
    _, evecs = torch.linalg.eigh(deltas_reg)  # (B, n, n)
    u = evecs[:, :, :k_eff] * (n ** 0.5)  # (B, n, k_eff)

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
    k_eff = min(k, n)
    device = deltas.device
    dtype = deltas.dtype

    deltas_reg = _regularize_for_eigh(deltas)
    evals, evecs = torch.linalg.eigh(deltas_reg)  # (B, n), (B, n, n)
    evals_k = evals[:, :k_eff]  # (B, k_eff)
    evecs_k = evecs[:, :, :k_eff]  # (B, n, k_eff)

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
    if cfg.kind == "spe_learnable":
        raise ValueError("spe_learnable requires using StableExpressivePE module directly, not compute_pe_batch")
    raise ValueError(f"Unknown PE kind: {cfg.kind}")


# ============ Learnable PE Modules ============


class PsiNetwork(nn.Module):
    """Learnable spectral filter applied to eigenvalues.

    Uses a learnable sigmoid-like filter: sigma(alpha * (lambda - tau))
    This is more effective than MLP for spectral filtering.
    """

    def __init__(self, hidden: int = 32, filter_idx: int = 0, num_filters: int = 16) -> None:
        super().__init__()
        # Initialize with diverse thresholds spread across [0, 1]
        tau_init = (filter_idx + 0.5) / num_filters  # Spread tau across [0, 1]
        alpha_init = 3.0 + 4.0 * (filter_idx % 4) / 3  # Vary alpha between 3 and 7

        self.alpha = nn.Parameter(torch.tensor(alpha_init))  # Steepness
        self.tau = nn.Parameter(torch.tensor(tau_init))      # Threshold
        self.scale = nn.Parameter(torch.tensor(1.0))         # Output scale
        self.bias = nn.Parameter(torch.tensor(0.0))          # Output bias

    def forward(self, Lambda: torch.Tensor) -> torch.Tensor:
        """Apply learned sigmoid filter to eigenvalues.

        Args:
            Lambda: [B, k, 1] normalized eigenvalues in [0, 1]

        Returns:
            [B, k, 1] filtered values
        """
        # Sigmoid filter centered at tau with steepness alpha
        x = self.alpha * (Lambda - self.tau)
        return self.scale * torch.sigmoid(x) + self.bias


class PhiAggregator(nn.Module):
    """Aggregates spectral filter outputs to node features (memory-efficient)."""

    def __init__(self, m: int, hidden: int, out_dim: int) -> None:
        super().__init__()
        # Process aggregated node features directly (not edge features)
        self.node_mlp = nn.Sequential(
            nn.Linear(m, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, node_feats_list: List[torch.Tensor]) -> List[torch.Tensor]:
        """Aggregate to node features.

        Args:
            node_feats_list: List of [N_i, M] aggregated node features

        Returns:
            List of [N_i, out_dim] node features
        """
        return [self.node_mlp(feats) for feats in node_feats_list]


class StableExpressivePE(nn.Module):
    """Learnable Stable Expressive Positional Encoding (memory-efficient).

    Computes node features from spectral filters without materializing [n,n,M] matrices.

    Mathematical insight:
        W = V @ diag(Z) @ V^T  where W is [n, n, M]
        mean_j(W[i,j,:]) = (1/n) * (V @ diag(Z) @ V^T @ 1)[i]
                        = (1/n) * V @ (Z * V.sum(dim=0))

    This reduces memory from O(n²M) to O(nkM).
    """

    def __init__(
        self,
        m: int,
        out_dim: int,
        psi_hidden: int = 32,
        phi_hidden: int = 64,
    ) -> None:
        super().__init__()
        self.m = m
        self.out_dim = out_dim
        self.psi_list = nn.ModuleList([
            PsiNetwork(hidden=psi_hidden, filter_idx=i, num_filters=m) for i in range(m)
        ])
        self.phi = PhiAggregator(m=m, hidden=phi_hidden, out_dim=out_dim)
        # Layer norm for numerical stability
        self.layer_norm = nn.LayerNorm(m)

    def forward(
        self,
        Lambda: torch.Tensor,
        V: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Compute learnable PE from eigenvalues and eigenvectors.

        Args:
            Lambda: [B, k] eigenvalues (batched by size)
            V: [B, n, k] eigenvectors (batched by size)

        Returns:
            List of [n, out_dim] node features, one per graph in batch
        """
        B, n, k = V.shape

        # Normalize eigenvalues to [0, 1] range for stable psi input
        Lambda_min = Lambda.min(dim=1, keepdim=True).values
        Lambda_max = Lambda.max(dim=1, keepdim=True).values
        Lambda_norm = (Lambda - Lambda_min) / (Lambda_max - Lambda_min + 1e-8)
        Lambda_norm = Lambda_norm.unsqueeze(2)  # [B, k, 1]

        # Apply each psi to get filter coefficients: [B, k, M]
        Z = torch.stack(
            [psi(Lambda_norm).squeeze(2) for psi in self.psi_list], dim=2
        )  # [B, k, M]

        # Simple and effective: node features = eigenvectors modulated by learned filters
        # node_feats[i, m] = sum_k V[i,k] * Z[k,m]
        node_feats_list = []
        for b in range(B):
            V_b = V[b]  # [n, k]
            Z_b = Z[b]  # [k, M]

            # Direct spectral features: V @ Z -> [n, M]
            node_feats = V_b @ Z_b  # [n, M]

            # Apply layer norm for stability
            node_feats = self.layer_norm(node_feats)
            node_feats_list.append(node_feats)

        # Apply phi MLP to get final features
        return self.phi(node_feats_list)


def eigh_batch_for_learnable(
    deltas: torch.Tensor,
    k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Batch eigendecomposition for learnable PE.

    Args:
        deltas: [B, n, n] normalized shift operators
        k: number of eigenvalues/eigenvectors to keep

    Returns:
        Lambda: [B, k] eigenvalues
        V: [B, n, k] eigenvectors (scaled by sqrt(n))
    """
    B, n, _ = deltas.shape
    k_eff = min(k, n)
    deltas_reg = _regularize_for_eigh(deltas)
    evals, evecs = torch.linalg.eigh(deltas_reg)
    Lambda = evals[:, :k_eff]  # [B, k_eff]
    V = evecs[:, :, :k_eff] * (n**0.5)  # [B, n, k_eff]
    # Pad with zeros if n < k
    if k_eff < k:
        Lambda_pad = torch.zeros(B, k - k_eff, device=deltas.device, dtype=deltas.dtype)
        Lambda = torch.cat([Lambda, Lambda_pad], dim=1)
        V_pad = torch.zeros(B, n, k - k_eff, device=deltas.device, dtype=deltas.dtype)
        V = torch.cat([V, V_pad], dim=2)
    return Lambda, V


def build_learnable_pe(cfg: PEConfig) -> StableExpressivePE:
    """Build learnable PE module from config."""
    if cfg.kind != "spe_learnable":
        raise ValueError(f"build_learnable_pe requires kind='spe_learnable', got {cfg.kind}")
    if cfg.m <= 0:
        raise ValueError("spe_learnable requires m > 0")
    return StableExpressivePE(
        m=cfg.m,
        out_dim=cfg.m,
        psi_hidden=cfg.spe_learnable_psi_hidden,
        phi_hidden=cfg.spe_learnable_phi_hidden,
    )
