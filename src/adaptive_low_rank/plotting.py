import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from collections import defaultdict
from pathlib import Path


def plot_residuals(results, output_dir):
    """
    Plot normalized residual curves.

    Parameters
    ----------
    results : list
        Output of benchmark().

    output_dir : str or Path
        Directory in which to save the figures.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    grouped = defaultdict(list)

    # Group runs that differ only by random_state
    for run in results:
        params = run["parameters"].copy()
        params.pop("random_state", None)
        key = (
            run["algorithm"],
            tuple(sorted(params.items())),
        )
        grouped[key].append(run["result"])

    # Assign one base color per algorithm
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    algorithm_names = sorted({alg for alg, _ in grouped.keys()})
    base_colors = {
        alg: color_cycle[i % len(color_cycle)]
        for i, alg in enumerate(algorithm_names)
    }

    # Count how many parameter settings each algorithm has
    algorithm_counts = defaultdict(int)
    for algorithm, _ in grouped.keys():
        algorithm_counts[algorithm] += 1

    # Sort so shades correspond to increasing parameter values
    grouped = dict(sorted(grouped.items()))
    algorithm_indices = defaultdict(int)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    for (algorithm, params), runs in grouped.items():
        residuals = np.array([r.residuals for r in runs])

        # Normalize every trial
        residuals = residuals / residuals[:, [0]]
        mean = residuals.mean(axis=0)
        x = np.arange(1, len(mean) + 1)

        # Choose shade of algorithm color
        base = np.array(mcolors.to_rgb(base_colors[algorithm]))
        n = algorithm_counts[algorithm]
        i = algorithm_indices[algorithm]
        algorithm_indices[algorithm] += 1

        # Dark -> light shading
        t = 0.15 + 0.55 * i / max(n - 1, 1)
        color = (1 - t) * base + t * np.ones(3)

        # Plot
        params = dict(params)
        label = algorithm.replace("_", " ").title()
        extras = [f"{k}={v}" for k, v in params.items()]
        if extras:
            label += " (" + ", ".join(extras) + ")"
        if len(runs) == 1:
            ax.plot(
                x,
                mean,
                color=color,
                label=label,
            )
        else:
            std = residuals.std(axis=0)
            ax.plot(
                x,
                mean,
                color=color,
                label=label,
            )
            ax.fill_between(
                x,
                mean - std,
                mean + std,
                color=color,
                alpha=0.2,
            )

    ax.set_yscale("log")
    ax.set_xlabel("Selected Rows")
    ax.set_ylabel("Normalized Residual")
    ax.legend()
    fig.tight_layout()

    fig.savefig(
        output_dir / "residuals.pdf",
        bbox_inches="tight",
    )
    fig.savefig(
        output_dir / "residuals.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_alphas(results, output_dir):
    """
    Plot alpha values over the iterations.

    Parameters
    ----------
    results : list
        Output of benchmark().

    output_dir : str or Path
        Directory in which to save the figures.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    grouped = defaultdict(list)

    # Group runs that differ only by random_state
    for run in results:
        if run["result"].alphas is None:
            continue
        if np.all(np.isnan(run["result"].alphas)):
            continue
        params = run["parameters"].copy()
        params.pop("random_state", None)
        key = (
            run["algorithm"],
            tuple(sorted(params.items())),
        )
        grouped[key].append(run["result"])

    if not grouped:
        return

    # Assign one base color per algorithm
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    algorithm_names = sorted({alg for alg, _ in grouped.keys()})
    base_colors = {
        alg: color_cycle[i % len(color_cycle)]
        for i, alg in enumerate(algorithm_names)
    }

    algorithm_counts = defaultdict(int)
    for algorithm, _ in grouped.keys():
        algorithm_counts[algorithm] += 1

    grouped = dict(sorted(grouped.items()))
    algorithm_indices = defaultdict(int)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    for (algorithm, params), runs in grouped.items():

        alphas = np.array([r.alphas for r in runs], dtype=float)

        mean = np.nanmean(alphas, axis=0)
        x = np.arange(1, len(mean) + 1)

        base = np.array(mcolors.to_rgb(base_colors[algorithm]))
        n = algorithm_counts[algorithm]
        i = algorithm_indices[algorithm]
        algorithm_indices[algorithm] += 1

        t = 0.15 + 0.55 * i / max(n - 1, 1)
        color = (1 - t) * base + t * np.ones(3)

        # Plot
        params = dict(params)
        label = algorithm.replace("_", " ").title()
        extras = [f"{k}={v}" for k, v in params.items()]
        if extras:
            label += " (" + ", ".join(extras) + ")"
        if len(runs) == 1:
            ax.plot(
                x,
                mean,
                color=color,
                label=label,
            )
        else:
            std = np.nanstd(alphas, axis=0)
            ax.plot(
                x,
                mean,
                color=color,
                label=label,
            )
            ax.fill_between(
                x,
                mean - std,
                mean + std,
                color=color,
                alpha=0.2,
            )

    ax.set_xlabel("Selected Rows")
    ax.set_ylabel(r"$\alpha$")
    ax.legend()
    fig.tight_layout()

    fig.savefig(
        output_dir / "alphas.pdf",
        bbox_inches="tight",
    )
    fig.savefig(
        output_dir / "alphas.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)