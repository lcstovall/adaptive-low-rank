"""Generate synthetic datasets described by experiment YAML files."""

import argparse
from pathlib import Path
import re

import numpy as np
import yaml

from adaptive_low_rank.datasets import generate_synthetic_dataset

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"
DATA_DIR = ROOT / "data"
SYNTHETIC_NAME = re.compile(r"^(exp|poly).+$")


def _generate(config_path, force=False):
    name = config_path.stem
    if not SYNTHETIC_NAME.fullmatch(name):
        return False

    with config_path.open() as file:
        config = yaml.safe_load(file) or {}

    missing = [key for key in ("decay_type", "decay_param") if key not in config]
    if missing:
        raise ValueError(
            f"{config_path.name} is synthetic but is missing: {', '.join(missing)}"
        )

    output_path = DATA_DIR / f"{name}.npz"
    if output_path.exists() and not force:
        print(f"Skipping {name}: {output_path.name} already exists")
        return True

    decay_type = str(config["decay_type"]).lower()
    if decay_type not in {"exp", "poly"}:
        raise ValueError(f"Unsupported decay_type {decay_type!r} in {config_path.name}")
    if not name.startswith(decay_type):
        raise ValueError(
            f"{config_path.name} starts with a different decay type than "
            f"decay_type={decay_type!r}"
        )

    n = int(config.get("n", 2000))
    d = int(config.get("d", 2000))
    random_state = config.get("random_state", 0)
    if n < 1 or d < 1:
        raise ValueError(f"n and d must be positive in {config_path.name}")

    X = generate_synthetic_dataset(
        decay_type=decay_type,
        decay_param=float(config["decay_param"]),
        n=n,
        d=d,
        random_state=random_state,
    )
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, X=X)
    print(f"Generated {output_path} with shape {X.shape}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Generate datasets for configs named exp* or poly*."
    )
    parser.add_argument(
        "--pattern",
        default=None,
        help="Only process matching config filenames, such as exp*.yml.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate datasets even when the NPZ already exists.",
    )
    args = parser.parse_args()

    if args.pattern is None:
        configs = sorted(set(CONFIG_DIR.glob("*.yml")) | set(CONFIG_DIR.glob("*.yaml")))
    else:
        configs = sorted(CONFIG_DIR.glob(args.pattern))
    generated = [_generate(path, force=args.force) for path in configs]
    count = sum(generated)
    print(f"Processed {count} synthetic dataset(s)")


if __name__ == "__main__":
    main()
