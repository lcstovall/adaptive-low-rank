import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from collections import defaultdict
from pathlib import Path


METHOD_MARKERS = {
    "adaptive": "o",
    "batch_max": "s",
    "greedy": "^",
    "greedy_pp": "D",
    "random": "x",
}
_assigned_markers = dict(METHOD_MARKERS)


def _is_batch_max(algorithm):
    return algorithm.replace("_", "").lower() == "batchmax"


def _marker_for_algorithm(algorithm):
    """Return the stable marker assigned to an algorithm."""
    if algorithm in _assigned_markers:
        return _assigned_markers[algorithm]

    marker_cycle = ["o", "s", "^", "D", "*", "P", "v", "<", ">", "h"]
    known_markers = set(_assigned_markers.values())
    available_markers = [
        marker for marker in marker_cycle
        if marker not in known_markers
    ]
    marker = available_markers[0] if available_markers else "o"
    _assigned_markers[algorithm] = marker
    return marker


def plot_residuals(results, output_dir, name="residuals"):
    """
    Plot normalized residual curves.

    Parameters
    ----------
    results : list
        Output of benchmark().

    output_dir : str or Path
        Directory in which to save the figures.

    name : str, default="residuals"
        Filename stem for the saved figures.

    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    grouped = defaultdict(list)

    # Group runs
    for run in results:
        params = run["parameters"].copy()
        params.pop("random_state", None)
        key = (
            run["algorithm"],
            tuple(sorted(params.items())),
        )
        grouped[key].append({"result": run["result"], "init_res": run["init_res"]})

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
    grouped = dict(
        sorted(
            grouped.items(),
            key=lambda item: _is_batch_max(item[0][0]),
        )
    )
    algorithm_indices = defaultdict(int)

    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)

    for (algorithm, params), runs in grouped.items():
        residuals = np.array([run["result"].residuals for run in runs])
        init_res = np.array([run["init_res"] for run in runs])

        # Normalize by initial residual
        residuals = residuals / init_res[:, None]

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
        marker = _marker_for_algorithm(algorithm)

        # Label
        label = algorithm.replace("_", " ").title()

        # Plot mean
        ax.plot(
            x,
            mean,
            color=color,
            linewidth=2.0,
            marker=marker,
            markevery=max(len(x) // 20, 1),
            markersize=8,
            markeredgewidth=3,
            label=label,
        )

        # Plot uncertainty
        if len(runs) > 1:
            std = residuals.std(axis=0)

            ax.fill_between(
                x,
                mean - std,
                mean + std,
                color=color,
                alpha=0.10,
            )

    ax.set_yscale("log")
    ax.set_xlabel("k", fontsize=13)
    ax.set_ylabel("Normalized Residual", fontsize=13)
    ax.tick_params(axis="both", labelsize=11)

    ax.grid(
        True,
        which="major",
        axis="y",
        alpha=0.3,
    )

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        fontsize=10,
        frameon=False,
    )

    fig.subplots_adjust(
        right=0.75,
        left=0.10,
        bottom=0.12,
        top=0.97,
    )

    fig.savefig(
        output_dir / f"{name}_residuals.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()


def plot_runtime_scaling(
    results,
    output_dir,
    fixed_d=None,
    fixed_n=None,
    name="runtime_scaling",
):
    """Plot mean runtime with one standard deviation across repeats.

    Parameters
    ----------
    results : dict
        Serialized output from the runtime scaling benchmark.

    output_dir : str or Path
        Directory in which to save the figures.

    fixed_d : int, optional
        Fix the number of rows and plot runtime as ``n`` varies.

    fixed_n : int, optional
        Fix the number of columns and plot runtime as ``d`` varies.

    name : str, default="runtime_scaling"
        Filename stem for the saved figures.

    show : bool, default=False
        Display the figure instead of closing it after saving.
    """

    if (fixed_d is None) == (fixed_n is None):
        raise ValueError("Specify exactly one of fixed_d or fixed_n.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    algorithms = results["algorithms"]
    dimensions = np.asarray(results["dimensions"])
    n_values = np.asarray(results["n_values"])
    runtime_matrix = np.asarray(results["runtimes"])

    if fixed_d is not None:
        dimension_index = results["dimensions"].index(fixed_d)
        runtimes = runtime_matrix[:, dimension_index, :, :]
        x = n_values
        xlabel = "Number of columns (n)"
        title = f"Runtime scaling with n, d={fixed_d}"
    else:
        n_index = results["n_values"].index(fixed_n)
        runtimes = runtime_matrix[:, :, :, n_index].transpose(0, 2, 1)
        x = dimensions
        xlabel = "Number of rows/features (d)"
        title = f"Runtime scaling with d, n={fixed_n:,}"

    means = runtimes.mean(axis=1)
    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)

    algorithm_order = sorted(
        range(len(algorithms)),
        key=lambda index: _is_batch_max(algorithms[index]),
    )

    for algorithm_index in algorithm_order:
        algorithm = algorithms[algorithm_index]
        mean = means[algorithm_index]

        ax.plot(
            x,
            mean,
            marker="o",
            label=algorithm,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Runtime (seconds)")
    ax.set_title(title)
    ax.tick_params(axis="both", labelsize=11)

    ax.grid(
        True,
        which="major",
        axis="y",
        alpha=0.3,
    )

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        fontsize=10,
        frameon=False,
    )

    fig.subplots_adjust(
        right=0.75,
        left=0.10,
        bottom=0.12,
        top=0.92,
    )

    fig.savefig(
        output_dir / f"{name}.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()


def plot_alphas(results, output_dir, name="alphas"):
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

    grouped = dict(
        sorted(
            grouped.items(),
            key=lambda item: _is_batch_max(item[0][0]),
        )
    )
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
        marker = _marker_for_algorithm(algorithm)

        # Plot
        label = algorithm.replace("_", " ").title()
        if len(runs) == 1:
            ax.plot(
                x,
                mean,
                color=color,
                marker=marker,
                markevery=max(len(x) // 20, 1),
                markersize=4,
                label=label,
            )
        else:
            std = np.nanstd(alphas, axis=0)
            ax.plot(
                x,
                mean,
                color=color,
                marker=marker,
                markevery=max(len(x) // 20, 1),
                markersize=4,
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
    ax.tick_params(axis="both", labelsize=11)

    ax.grid(
        True,
        which="major",
        axis="y",
        alpha=0.3,
    )

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        fontsize=10,
        frameon=False,
    )

    fig.subplots_adjust(
        right=0.75,
        left=0.10,
        bottom=0.12,
        top=0.97,
    )

    fig.savefig(
        output_dir / f"{name}_alphas.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()