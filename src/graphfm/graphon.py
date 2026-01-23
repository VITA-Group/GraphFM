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


@dataclass(frozen=True)
class ControlledFourierGraphon(Graphon):
    """Fourier graphon with controllable eigenvalues.

    W(x,y) = rho + sum_m (lambda_cos[m] * phi_c(x) * phi_c(y)
                        + lambda_sin[m] * phi_s(x) * phi_s(y))
    where phi_c(x) = sqrt(2) * cos(2*pi*m*x)
          phi_s(x) = sqrt(2) * sin(2*pi*m*x)

    Constraint: sum(|lambda_cos| + |lambda_sin|) <= 0.5 * min(rho, 1-rho)
    """

    rho: float
    lambda_cos: Sequence[float]  # coefficients for cos basis (m=1,2,...)
    lambda_sin: Sequence[float]  # coefficients for sin basis (m=1,2,...)

    def __call__(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        y = np.asarray(y)
        out = np.full_like(x, fill_value=self.rho, dtype=np.float64)
        sqrt2 = np.sqrt(2.0)

        for m, lam_c in enumerate(self.lambda_cos, start=1):
            phi_x = sqrt2 * np.cos(2.0 * np.pi * m * x)
            phi_y = sqrt2 * np.cos(2.0 * np.pi * m * y)
            out += lam_c * phi_x * phi_y

        for m, lam_s in enumerate(self.lambda_sin, start=1):
            phi_x = sqrt2 * np.sin(2.0 * np.pi * m * x)
            phi_y = sqrt2 * np.sin(2.0 * np.pi * m * y)
            out += lam_s * phi_x * phi_y

        return out  # No clip needed if constraints satisfied


def make_controlled_fourier_graphons(
    num_classes: int,
    rho: float,
    num_terms: int,
    rng: np.random.Generator,
) -> List[ControlledFourierGraphon]:
    """Create controlled Fourier graphons with valid eigenvalue constraints.

    Default: rho=0.5, so sum(|lambdas|) <= 0.25
    """
    max_sum = 0.5 * min(rho, 1.0 - rho)  # = 0.25 when rho=0.5
    graphons: List[ControlledFourierGraphon] = []

    for _ in range(num_classes):
        # Generate random coefficients and scale to satisfy constraint
        raw_cos = rng.uniform(-1, 1, size=num_terms)
        raw_sin = rng.uniform(-1, 1, size=num_terms)
        total = np.sum(np.abs(raw_cos)) + np.sum(np.abs(raw_sin))

        # Scale to use ~80% of budget for safety margin
        scale = (0.8 * max_sum) / max(total, 1e-8)
        lambda_cos = (raw_cos * scale).tolist()
        lambda_sin = (raw_sin * scale).tolist()

        graphons.append(
            ControlledFourierGraphon(
                rho=rho,
                lambda_cos=lambda_cos,
                lambda_sin=lambda_sin,
            )
        )

    return graphons
