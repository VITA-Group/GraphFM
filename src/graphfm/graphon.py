from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np


class Graphon:
    def __call__(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        raise NotImplementedError


@dataclass(frozen=True)
class FourierGraphon(Graphon):
    rho: float
    coeffs: Sequence[float]

    def __call__(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        y = np.asarray(y)
        diff = x - y
        out = np.full_like(diff, fill_value=self.rho, dtype=np.float64)
        for m, a_m in enumerate(self.coeffs, start=1):
            out += a_m * np.cos(2.0 * np.pi * m * diff)
        return np.clip(out, 0.0, 1.0)


@dataclass(frozen=True)
class StepGraphon(Graphon):
    bins: int
    matrix: np.ndarray  # (bins, bins)

    def __post_init__(self) -> None:
        if self.matrix.shape != (self.bins, self.bins):
            raise ValueError("StepGraphon matrix shape mismatch.")

    def __call__(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        y = np.asarray(y)
        ix = np.minimum((x * self.bins).astype(int), self.bins - 1)
        iy = np.minimum((y * self.bins).astype(int), self.bins - 1)
        return self.matrix[ix, iy]


def make_fourier_graphons(
    num_classes: int,
    rho: float,
    num_terms: int,
    coeff_scale: float,
    rng: np.random.Generator,
) -> List[FourierGraphon]:
    graphons: List[FourierGraphon] = []
    for _ in range(num_classes):
        coeffs = rng.uniform(-coeff_scale, coeff_scale, size=num_terms)
        graphons.append(FourierGraphon(rho=rho, coeffs=coeffs.tolist()))
    return graphons


def perturb_graphon_coeffs(
    graphon: FourierGraphon,
    noise_scale: float,
    rng: np.random.Generator,
) -> FourierGraphon:
    coeffs = np.asarray(graphon.coeffs)
    coeffs = coeffs + rng.normal(scale=noise_scale, size=coeffs.shape)
    return FourierGraphon(rho=graphon.rho, coeffs=coeffs.tolist())
