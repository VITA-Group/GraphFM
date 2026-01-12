from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


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
