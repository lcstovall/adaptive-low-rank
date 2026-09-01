"""Plot normalized optimal low-rank residual curves for configured datasets."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from adaptive_low_rank.datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_CONFIGS = {"interactions_alpha", "yearprediction_alpha"}


def optimal_residuals(X):
    """Return Frobenius residuals for the optimal ranks 0 through r_max."""
    X = np.asarray(X, dtype=float)

    singular_values_squared = np.linalg.svdvals(X) ** 2
    residual_squared = np.concatenate(
        (
            [singular_values_squared.sum()],
            np.cumsum(singular_values_squared[::-1])[::-1][1:],
            [0.0],
        )
    )
    return np.sqrt(residual_squared)


def residuals_from_svals(svals):
    singular_values_squared = svals**2
    residual_squared = np.concatenate(
        (
            [singular_values_squared.sum()],
            np.cumsum(singular_values_squared[::-1])[::-1][1:],
            [0.0],
        )
    )
    return np.sqrt(residual_squared)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "figures" / "optimal_residuals_normalized.png",
        help="Output figure path.",
    )
    parser.add_argument(
        "--yscale",
        choices=("linear", "log"),
        default="log",
        help="Scale for normalized residuals (default: linear).",
    )
    args = parser.parse_args()

    curves = {}
    for config_path in sorted((ROOT / "configs").glob("*.yml")):
        experiment_name = config_path.stem
        if experiment_name in EXCLUDED_CONFIGS:
            continue

        with config_path.open() as config_file:
            experiment = yaml.safe_load(config_file)

        dataset_name = experiment["dataset"]
        print(f"Computing optimal curve for {experiment_name} ({dataset_name})")

        # if dataset_name != "interactions": continue
        residuals = optimal_residuals(load_dataset(dataset_name))
        curves[experiment_name] = residuals / residuals[0]

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    for name, residuals in curves.items():
        progress = np.linspace(0.0, 1.0, len(residuals))
        values = residuals.copy()
        if args.yscale == "log":
            values[values <= 0] = np.nan
        ax.plot(progress, values, linewidth=2.0, label=name)

    ax.set_xlabel("Normalized rank progress, k / r_max")
    ax.set_ylabel("Normalized optimal residual, E_k / E_0")
    ax.set_yscale(args.yscale)
    ax.set_xlim(0.0, 1.0)
    ax.grid(True, which="both", axis="y", alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)
    fig.subplots_adjust(right=0.78, left=0.11, bottom=0.12, top=0.97)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
