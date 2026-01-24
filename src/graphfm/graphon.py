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


def perturb_controlled_graphon(
    graphon: ControlledFourierGraphon,
    perturbation_level: float,  # in [0, 1]
    max_l2_distance: float,     # maximum L2 distance at level=1
    rng: np.random.Generator,
    perturb_rho: bool = False,
) -> ControlledFourierGraphon:
    """
    Perturb a ControlledFourierGraphon with controllable L2 distance.
    
    L2 distance = perturbation_level * max_l2_distance
    
    Args:
        graphon: Original graphon
        perturbation_level: Value in [0, 1], controls perturbation strength
        max_l2_distance: L2 distance when perturbation_level = 1
        rng: Random generator
        perturb_rho: Whether to also perturb rho (default: False for simplicity)
    
    Returns:
        New graphon with ||W_new - W_old||_L2 = perturbation_level * max_l2_distance
    """
    if not 0 <= perturbation_level <= 1:
        raise ValueError("perturbation_level must be in [0, 1]")
    
    if perturbation_level == 0:
        return graphon
    
    target_l2 = perturbation_level * max_l2_distance
    
    # Generate random direction for perturbation
    n_cos = len(graphon.lambda_cos)
    n_sin = len(graphon.lambda_sin)
    
    if perturb_rho:
        # Include rho in perturbation
        direction = rng.standard_normal(1 + n_cos + n_sin)
    else:
        direction = rng.standard_normal(n_cos + n_sin)
    
    # Normalize to unit vector, then scale to target L2 distance
    direction = direction / np.linalg.norm(direction)
    delta = direction * target_l2
    
    # Apply perturbation
    if perturb_rho:
        new_rho = graphon.rho + delta[0]
        delta_cos = delta[1:1+n_cos]
        delta_sin = delta[1+n_cos:]
    else:
        new_rho = graphon.rho
        delta_cos = delta[:n_cos]
        delta_sin = delta[n_cos:]
    
    new_lambda_cos = (np.array(graphon.lambda_cos) + delta_cos).tolist()
    new_lambda_sin = (np.array(graphon.lambda_sin) + delta_sin).tolist()
    
    # Check constraint: sum(|lambdas|) <= 0.5 * min(rho, 1-rho)
    new_rho_clipped = np.clip(new_rho, 0.01, 0.99)
    max_sum = 0.5 * min(new_rho_clipped, 1.0 - new_rho_clipped)
    current_sum = np.sum(np.abs(new_lambda_cos)) + np.sum(np.abs(new_lambda_sin))
    
    if current_sum > max_sum:
        # Scale down lambdas to satisfy constraint
        scale = 0.95 * max_sum / current_sum
        new_lambda_cos = (np.array(new_lambda_cos) * scale).tolist()
        new_lambda_sin = (np.array(new_lambda_sin) * scale).tolist()
    
    return ControlledFourierGraphon(
        rho=new_rho_clipped,
        lambda_cos=new_lambda_cos,
        lambda_sin=new_lambda_sin,
    )

def perturb_controlled_graphon_monotonic(
    graphon: ControlledFourierGraphon,
    perturbation_level: float,
    max_l2_distance: float,
    rng: np.random.Generator = np.random.default_rng(42),
    direction_seed: int = 42,  # 固定方向的种子
) -> ControlledFourierGraphon:
    """
    保证单调性的扰动：对同一 graphon 使用固定方向。
    """
    if not 0 <= perturbation_level <= 1:
        raise ValueError("perturbation_level must be in [0, 1]")
    
    if perturbation_level == 0:
        return graphon
    
    n_cos = len(graphon.lambda_cos)
    n_sin = len(graphon.lambda_sin)
    
    # 用固定种子生成固定方向
    rng_dir = np.random.default_rng(direction_seed)
    direction = rng_dir.standard_normal(n_cos + n_sin)
    direction = direction / np.linalg.norm(direction)
    
    lambda_cos_arr = np.array(graphon.lambda_cos)
    lambda_sin_arr = np.array(graphon.lambda_sin)
    
    # 计算约束
    max_sum = 0.5 * min(graphon.rho, 1.0 - graphon.rho)
    current_sum = np.sum(np.abs(lambda_cos_arr)) + np.sum(np.abs(lambda_sin_arr))
    remaining_budget = max_sum - current_sum
    
    # 计算这个方向能走的最大距离
    delta_cos_unit = direction[:n_cos]
    delta_sin_unit = direction[n_cos:]
    
    # 精确计算：找到使 sum(|lambda + t*delta|) = max_sum 的最大 t
    # 这里用保守估计
    max_increase_rate = np.sum(np.abs(delta_cos_unit)) + np.sum(np.abs(delta_sin_unit))
    
    if max_increase_rate > 1e-10:
        max_feasible_step = remaining_budget / max_increase_rate
    else:
        max_feasible_step = float('inf')
    
    # 目标步长
    target_step = perturbation_level * max_l2_distance
    
    # 实际步长（受约束限制）
    actual_step = min(target_step, max_feasible_step)
    
    new_lambda_cos = (lambda_cos_arr + delta_cos_unit * actual_step).tolist()
    new_lambda_sin = (lambda_sin_arr + delta_sin_unit * actual_step).tolist()
    
    return ControlledFourierGraphon(
        rho=graphon.rho,
        lambda_cos=new_lambda_cos,
        lambda_sin=new_lambda_sin,
    )

def graphon_l2_distance(
    g1: ControlledFourierGraphon,
    g2: ControlledFourierGraphon,
) -> float:
    """Compute exact L2 distance between two ControlledFourierGraphons."""
    d_rho = g1.rho - g2.rho
    
    # Pad to same length
    n_cos = max(len(g1.lambda_cos), len(g2.lambda_cos))
    n_sin = max(len(g1.lambda_sin), len(g2.lambda_sin))
    
    cos1 = np.array(list(g1.lambda_cos) + [0] * (n_cos - len(g1.lambda_cos)))
    cos2 = np.array(list(g2.lambda_cos) + [0] * (n_cos - len(g2.lambda_cos)))
    sin1 = np.array(list(g1.lambda_sin) + [0] * (n_sin - len(g1.lambda_sin)))
    sin2 = np.array(list(g2.lambda_sin) + [0] * (n_sin - len(g2.lambda_sin)))
    
    l2_sq = d_rho**2 + np.sum((cos1 - cos2)**2) + np.sum((sin1 - sin2)**2)
    return np.sqrt(l2_sq)

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

def check_graphon_constraint(g: ControlledFourierGraphon) -> dict:
    """Check if graphon satisfies the constraint."""
    max_sum = 0.5 * min(g.rho, 1.0 - g.rho)
    current_sum = np.sum(np.abs(g.lambda_cos)) + np.sum(np.abs(g.lambda_sin))
    return {
        "max_allowed": max_sum,
        "current_sum": current_sum,
        "satisfied": current_sum <= max_sum,
        "margin": max_sum - current_sum,
    }


if __name__ == "__main__":
    from functools import partial
    perturb_controlled_graphon = partial(perturb_controlled_graphon_monotonic, direction_seed=42)
    print("=" * 60)
    print("Testing Controlled Fourier Graphon Perturbation")
    print("=" * 60)
    
    rng = np.random.default_rng(42)
    
    # Create original graphon
    g_original = ControlledFourierGraphon(
        rho=0.5,
        lambda_cos=[0.08, 0.04, 0.02],
        lambda_sin=[0.06, 0.03, 0.01],
    )
    
    print("\n[Original Graphon]")
    print(f"  rho = {g_original.rho}")
    print(f"  lambda_cos = {g_original.lambda_cos}")
    print(f"  lambda_sin = {g_original.lambda_sin}")
    constraint_info = check_graphon_constraint(g_original)
    print(f"  Constraint: sum(|λ|) = {constraint_info['current_sum']:.4f} <= {constraint_info['max_allowed']:.4f}")
    print(f"  Satisfied: {constraint_info['satisfied']}")
    
    # Test different perturbation levels
    print("\n" + "-" * 60)
    print("Testing perturbation levels")
    print("-" * 60)
    
    levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    max_dist = 0.1  # Maximum L2 distance at level=1
    
    print(f"\nmax_l2_distance = {max_dist}")
    print(f"{'Level':<8} {'Target L2':<12} {'Actual L2':<12} {'Constraint':<12} {'Monotonic':<10}")
    print("-" * 54)
    
    prev_dist = -1.0
    all_monotonic = True
    
    for level in levels:
        # Use fresh rng state for reproducibility within each level
        rng_level = np.random.default_rng(42 + int(level * 1000))
        
        g_perturbed = perturb_controlled_graphon(
            g_original,
            perturbation_level=level,
            max_l2_distance=max_dist,
            rng=rng_level,
        )
        
        actual_dist = graphon_l2_distance(g_original, g_perturbed)
        target_dist = level * max_dist
        constraint = check_graphon_constraint(g_perturbed)
        
        is_monotonic = actual_dist >= prev_dist - 1e-10
        if not is_monotonic:
            all_monotonic = False
        
        print(f"{level:<8.1f} {target_dist:<12.4f} {actual_dist:<12.4f} {'✓' if constraint['satisfied'] else '✗':<12} {'✓' if is_monotonic else '✗':<10}")
        
        prev_dist = actual_dist
    
    print("-" * 54)
    print(f"All monotonic increasing: {'✓ Yes' if all_monotonic else '✗ No'}")
    
    # Test with multiple random seeds to verify consistency
    print("\n" + "-" * 60)
    print("Testing consistency across random seeds (level=0.5)")
    print("-" * 60)
    
    level = 0.5
    target = level * max_dist
    distances = []
    
    for seed in range(10):
        rng_test = np.random.default_rng(seed)
        g_perturbed = perturb_controlled_graphon(
            g_original,
            perturbation_level=level,
            max_l2_distance=max_dist,
            rng=rng_test,
        )
        dist = graphon_l2_distance(g_original, g_perturbed)
        distances.append(dist)
    
    print(f"Target L2 distance: {target:.4f}")
    print(f"Actual distances across 10 seeds:")
    print(f"  Mean:   {np.mean(distances):.4f}")
    print(f"  Std:    {np.std(distances):.4f}")
    print(f"  Min:    {np.min(distances):.4f}")
    print(f"  Max:    {np.max(distances):.4f}")
    
    # Visualize graphon values
    print("\n" + "-" * 60)
    print("Comparing graphon values at sample points")
    print("-" * 60)
    
    x_samples = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    y_samples = np.array([0.2, 0.4, 0.5, 0.6, 0.8])
    
    print(f"\n{'(x, y)':<12} {'Original':<12} {'Level=0.3':<12} {'Level=0.7':<12} {'Level=1.0':<12}")
    print("-" * 60)
    
    g_03 = perturb_controlled_graphon(g_original, 0.3, max_dist, np.random.default_rng(42))
    g_07 = perturb_controlled_graphon(g_original, 0.7, max_dist, np.random.default_rng(42))
    g_10 = perturb_controlled_graphon(g_original, 1.0, max_dist, np.random.default_rng(42))
    
    for x, y in zip(x_samples, y_samples):
        v_orig = g_original(np.array([x]), np.array([y]))[0]
        v_03 = g_03(np.array([x]), np.array([y]))[0]
        v_07 = g_07(np.array([x]), np.array([y]))[0]
        v_10 = g_10(np.array([x]), np.array([y]))[0]
        print(f"({x:.1f}, {y:.1f}){'':<4} {v_orig:<12.4f} {v_03:<12.4f} {v_07:<12.4f} {v_10:<12.4f}")
    
    # Test eigenvalue preservation
    print("\n" + "-" * 60)
    print("Eigenvalue comparison (coefficients)")
    print("-" * 60)
    
    print("\nOriginal eigenvalues (excluding rho):")
    print(f"  cos: {[f'{x:.4f}' for x in g_original.lambda_cos]}")
    print(f"  sin: {[f'{x:.4f}' for x in g_original.lambda_sin]}")
    
    print("\nPerturbed (level=0.5) eigenvalues:")
    g_05 = perturb_controlled_graphon(g_original, 0.5, max_dist, np.random.default_rng(42))
    print(f"  cos: {[f'{x:.4f}' for x in g_05.lambda_cos]}")
    print(f"  sin: {[f'{x:.4f}' for x in g_05.lambda_sin]}")
    
    print("\nPerturbed (level=1.0) eigenvalues:")
    print(f"  cos: {[f'{x:.4f}' for x in g_10.lambda_cos]}")
    print(f"  sin: {[f'{x:.4f}' for x in g_10.lambda_sin]}")
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)