"""Run and plot two orthogonal-tubes parameter grids.

Both grids use ``configs/orthogonal_tubes.yml`` as their base configuration
and vary dataset parameters per cell. Datasets are generated in memory, so
``data/`` is not modified. Results are saved to ``results/<grid name>/`` and
figures to ``figures/<grid name>_residuals.png``.
"""

import argparse
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import yaml

from adaptive_low_rank.benchmark import benchmark
from adaptive_low_rank.plotting import plot_residual_grid
from adaptive_low_rank.run_generator import generate_runs
from adaptive_low_rank.save_results import save_results
from adaptive_low_rank.synthetic import generate_multiscale_dataset

ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "configs" / "orthogonal_tubes.yml"

COLUMNS_PER_CLUSTER = [15, 30, 45]

GRIDS = {
    "orthogonal_tubes_eta_columns": ("eta", [0.025, 0.05, 0.1]),
    "orthogonal_tubes_condition_columns": ("condition_number", [1e2, 1e4, 1e6]),
}

PARAM_LABELS = {
    "eta": r"$\eta$",
    "condition_number": r"$\kappa$",
    "columns_per_cluster": "Columns per cluster",
}

DATASET_KEYS = (
    "model",
    "n_clusters",
    "columns_per_cluster",
    "condition_number",
    "tube_dim",
    "eta",
    "seed",
)


def run_grid(base, runs, row_param, row_values):
    results = []
    for row_value in row_values:
        for columns in COLUMNS_PER_CLUSTER:
            cell = {row_param: row_value, "columns_per_cluster": columns}
            settings = {**base, **cell}
            X = generate_multiscale_dataset(
                **{key: settings[key] for key in DATASET_KEYS}
            ).X

            print(f"Running {cell}", flush=True)
            for run in benchmark(X=X, runs=runs):
                run["parameters"].update(cell)
                results.append(run)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--replot",
        action="store_true",
        help="Plot from the saved results.pkl files without rerunning.",
    )
    parser.add_argument("--font-size", type=float, default=15)
    args = parser.parse_args()

    if not args.replot:
        with open(BASE_CONFIG) as f:
            base = yaml.safe_load(f)
        base["condition_number"] = float(base["condition_number"])
        runs = generate_runs(base)

    for name, (row_param, row_values) in GRIDS.items():
        results_dir = ROOT / "results" / name
        if args.replot:
            with open(results_dir / "results.pkl", "rb") as f:
                results = pickle.load(f)
        else:
            results = run_grid(base, runs, row_param, row_values)
            save_results(results, BASE_CONFIG, results_dir)
        plot_residual_grid(
            results,
            row_param=row_param,
            col_param="columns_per_cluster",
            output_dir=ROOT / "figures",
            name=name,
            font_size=args.font_size,
            param_labels=PARAM_LABELS,
        )
        plt.close("all")


if __name__ == "__main__":
    main()
