from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml

from .experiments import DatasetConfig
from .pe import PEConfig
from .train import TrainConfig


def format_number(n: int) -> str:
    """Format number with suffix: 50000 -> '50k', 1000000 -> '1m'."""
    if n >= 1_000_000_000:
        return f"{n // 1_000_000_000}b"
    if n >= 1_000_000:
        return f"{n // 1_000_000}m"
    if n >= 1_000:
        return f"{n // 1_000}k"
    return str(n)


def estimate_model_params(
    train_cfg: TrainConfig,
    in_dim: int,
    num_classes: int,
) -> int:
    """Estimate model parameter count based on config."""
    h = train_cfg.hidden

    if train_cfg.model == "deepsets":
        # phi: in->h->h, rho: h->h->out
        phi_params = in_dim * h + h + h * h + h
        rho_params = h * h + h + h * num_classes + num_classes
        return phi_params + rho_params

    if train_cfg.model == "degree":
        # bins->h->out
        return train_cfg.degree_bins * h + h + h * num_classes + num_classes

    if train_cfg.model == "gin":
        # 3 GIN layers + readout
        # Each GINLayer: MLP(in->out->out) + eps
        layers = 3
        dims = [in_dim] + [h] * layers
        gin_params = 0
        for i in range(layers):
            # MLP: d_in->d_out->d_out
            gin_params += dims[i] * dims[i + 1] + dims[i + 1]
            gin_params += dims[i + 1] * dims[i + 1] + dims[i + 1]
            gin_params += 1  # eps
        # readout: h->h->out
        readout_params = h * h + h + h * num_classes + num_classes
        return gin_params + readout_params

    return 0


def generate_config_filename(
    dataset_cfg: DatasetConfig,
    train_cfg: TrainConfig,
    pe_cfg: PEConfig,
    in_dim: int | None = None,
) -> str:
    """Generate descriptive config filename with all key info."""
    if in_dim is None:
        in_dim = pe_cfg.k

    params = estimate_model_params(train_cfg, in_dim, dataset_cfg.num_classes)
    budget_str = format_number(dataset_cfg.total_budget)
    params_str = format_number(params)

    pe_str = f"{pe_cfg.kind}_k{pe_cfg.k}"
    if pe_cfg.kind in ("proj", "spe") and pe_cfg.m > 0:
        pe_str += f"_m{pe_cfg.m}"

    return (
        f"budget{budget_str}_{train_cfg.model}_h{train_cfg.hidden}_"
        f"ep{train_cfg.epochs}_params{params_str}_{pe_str}.yaml"
    )


def _dataclass_to_dict(obj: Any) -> Dict[str, Any]:
    """Convert dataclass to dict, handling sequences."""
    result = {}
    for f in fields(obj):
        val = getattr(obj, f.name)
        if isinstance(val, (list, tuple)):
            result[f.name] = list(val)
        else:
            result[f.name] = val
    return result


def save_config(
    path: Path,
    dataset_cfg: DatasetConfig,
    train_cfg: TrainConfig,
    pe_cfg: PEConfig,
) -> None:
    """Save configuration to YAML file."""
    config = {
        "dataset": _dataclass_to_dict(dataset_cfg),
        "train": _dataclass_to_dict(train_cfg),
        "pe": _dataclass_to_dict(pe_cfg),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def load_config(path: Path) -> Tuple[DatasetConfig, TrainConfig, PEConfig]:
    """Load configuration from YAML file."""
    with open(path) as f:
        config = yaml.safe_load(f)

    # Support both "dataset" (new) and "experiment" (legacy) keys
    dataset_dict = config.get("dataset", config.get("experiment", {}))
    if "train_sizes" in dataset_dict:
        dataset_dict["train_sizes"] = tuple(dataset_dict["train_sizes"])
    if "test_sizes" in dataset_dict:
        dataset_dict["test_sizes"] = tuple(dataset_dict["test_sizes"])

    dataset_cfg = DatasetConfig(**dataset_dict)
    train_cfg = TrainConfig(**config.get("train", {}))
    pe_cfg = PEConfig(**config.get("pe", {}))

    return dataset_cfg, train_cfg, pe_cfg


def merge_config_with_args(
    dataset_cfg: DatasetConfig,
    train_cfg: TrainConfig,
    pe_cfg: PEConfig,
    args: Any,
) -> Tuple[DatasetConfig, TrainConfig, PEConfig]:
    """Merge loaded config with command-line argument overrides."""
    dataset_dict = _dataclass_to_dict(dataset_cfg)
    train_dict = _dataclass_to_dict(train_cfg)
    pe_dict = _dataclass_to_dict(pe_cfg)

    # Override with non-None command-line args
    if hasattr(args, "lambda_mix") and args.lambda_mix is not None:
        dataset_dict["lambda_mix"] = args.lambda_mix
    if hasattr(args, "device") and args.device is not None:
        train_dict["device"] = args.device
    if hasattr(args, "model") and args.model is not None:
        train_dict["model"] = args.model
    if hasattr(args, "pe_kind") and args.pe_kind is not None:
        pe_dict["kind"] = args.pe_kind
    if hasattr(args, "k") and args.k is not None:
        pe_dict["k"] = args.k
    if hasattr(args, "m") and args.m is not None:
        pe_dict["m"] = args.m

    return (
        DatasetConfig(**dataset_dict),
        TrainConfig(**train_dict),
        PEConfig(**pe_dict),
    )
