#!/usr/bin/env python3
"""Count parameters for different PE configurations."""

import torch
from graphfm.models import DeepSets, GIN
from graphfm.pe import PEConfig, build_learnable_pe


def count_params(model: torch.nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_params(model: torch.nn.Module, name: str) -> int:
    """Print parameter breakdown for a model."""
    total = 0
    print(f"\n{name}:")
    for n, p in model.named_parameters():
        if p.requires_grad:
            print(f"  {n}: {p.numel():,}")
            total += p.numel()
    print(f"  Total: {total:,}")
    return total


def main():
    # Config from spe_learnable.sh
    hidden = 256
    num_classes = 8

    # k and m values from the sweep
    k_values = [8, 16, 32]
    m_values = [8, 16]

    print("=" * 60)
    print("Parameter Count Comparison: eig PE vs spe_learnable PE")
    print("=" * 60)

    # ============ EIG PE ============
    print("\n" + "=" * 60)
    print("EIG PE (non-learnable)")
    print("=" * 60)

    for k in k_values:
        in_dim = k  # eig PE output dim = k
        model = DeepSets(in_dim=in_dim, hidden=hidden, out_dim=num_classes)
        total = print_model_params(model, f"DeepSets (k={k}, hidden={hidden})")
        print(f"\n>>> EIG PE k={k}: Total = {total:,} params")

    # ============ SPE LEARNABLE ============
    print("\n" + "=" * 60)
    print("SPE LEARNABLE (learnable PE + DeepSets)")
    print("=" * 60)

    for k in k_values:
        for m in m_values:
            pe_cfg = PEConfig(
                kind="spe_learnable",
                k=k,
                m=m,
                spe_learnable_psi_hidden=32,
                spe_learnable_phi_hidden=64,
            )

            # Build learnable PE module
            learnable_pe = build_learnable_pe(pe_cfg)
            pe_params = print_model_params(learnable_pe, f"StableExpressivePE (k={k}, m={m})")

            # Build classifier (input dim = m, output from learnable PE)
            in_dim = m
            model = DeepSets(in_dim=in_dim, hidden=hidden, out_dim=num_classes)
            model_params = print_model_params(model, f"DeepSets (in_dim={m}, hidden={hidden})")

            total = pe_params + model_params
            print(f"\n>>> SPE_LEARNABLE k={k}, m={m}: Total = {total:,} params")
            print(f"    (PE: {pe_params:,} + Model: {model_params:,})")

    # ============ Summary Table ============
    print("\n" + "=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)

    print("\nEIG PE:")
    print(f"{'k':<6} {'DeepSets':<15} {'Total':<15}")
    print("-" * 40)
    for k in k_values:
        model = DeepSets(in_dim=k, hidden=hidden, out_dim=num_classes)
        total = count_params(model)
        print(f"{k:<6} {total:<15,} {total:<15,}")

    print("\nSPE LEARNABLE:")
    print(f"{'k':<6} {'m':<6} {'PE':<12} {'DeepSets':<12} {'Total':<12}")
    print("-" * 50)
    for k in k_values:
        for m in m_values:
            pe_cfg = PEConfig(kind="spe_learnable", k=k, m=m)
            learnable_pe = build_learnable_pe(pe_cfg)
            pe_params = count_params(learnable_pe)

            model = DeepSets(in_dim=m, hidden=hidden, out_dim=num_classes)
            model_params = count_params(model)

            total = pe_params + model_params
            print(f"{k:<6} {m:<6} {pe_params:<12,} {model_params:<12,} {total:<12,}")


if __name__ == "__main__":
    main()
