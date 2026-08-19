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
    "random": "X",
}
_assigned_markers = dict(METHOD_MARKERS)


def _marker_for_algorithm(algorithm):
    """Return the stable marker assigned to an algorithm."""
    if algorithm in _assigned_markers:
        return _assigned_markers[algorithm]

    marker_cycle = ["o", "s", "^", "D", "X", "P", "v", "<", ">", "*", "h"]
    known_markers = set(_assigned_markers.values())
    available_markers = [
        marker for marker in marker_cycle
        if marker not in known_markers
    ]
    marker = available_markers[0] if available_markers else "o"
    _assigned_markers[algorithm] = marker
    return marker


def plot_residuals(results, output_dir, name="residuals", logx=False, show=False):
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

    logx : bool, default=False
        Use a logarithmic scale for the x-axis.

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
    grouped = dict(sorted(grouped.items()))
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
        params = dict(params)
        label = algorithm.replace("_", " ").title()

        extras = []
        for k, v in params.items():
            if k != "k":
                extras.append(f"{v} {k.replace('_', ' ')}")

        if extras:
            label += " (" + ", ".join(extras) + ")"

        # Plot mean
        ax.plot(
            x,
            mean,
            color=color,
            linewidth=2.0,
            marker=marker,
            markevery=max(len(x) // 20, 1),
            markersize=4,
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
    if logx:
        ax.set_xscale("log")

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
        output_dir / f"{name}.png",
        dpi=300,
        bbox_inches="tight",
    )

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_runtimes(results, output_dir):
    """
    Plot cumulative runtime curves.

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

        grouped[key].append(
            {
                "result": run["result"],
            }
        )

    # Assign one base color per algorithm
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    algorithm_names = sorted({
        alg for alg, _ in grouped.keys()
    })

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

        # Convert tuple of parameter pairs back to dictionary
        params = dict(params)

        # Get runtime history from every trial
        runtimes = np.array([
            run["result"].runtimes
            for run in runs
        ])

        # Mean runtime at each iteration
        mean = runtimes.mean(axis=0)

        # Iteration / number of selected columns
        x = np.arange(1, len(mean) + 1)

        # Choose shade of algorithm color
        base = np.array(
            mcolors.to_rgb(base_colors[algorithm])
        )

        n = algorithm_counts[algorithm]
        i = algorithm_indices[algorithm]
        algorithm_indices[algorithm] += 1

        # Dark -> light shading
        t = 0.15 + 0.55 * i / max(n - 1, 1)
        color = (1 - t) * base + t * np.ones(3)
        marker = _marker_for_algorithm(algorithm)

        # Label
        label = algorithm.replace("_", " ").title()

        extras = [
            f"{key}={value}"
            for key, value in params.items()
            if key != "k"
        ]

        if extras:
            label += " (" + ", ".join(extras) + ")"

        # Plot
        ax.plot(
            x,
            mean,
            color=color,
            marker=marker,
            markevery=max(len(x) // 20, 1),
            markersize=4,
            label=label,
        )

        # Standard deviation across random trials
        if len(runs) > 1:
            std = runtimes.std(axis=0)

            ax.fill_between(
                x,
                mean - std,
                mean + std,
                color=color,
                alpha=0.2,
            )

    ax.set_yscale("log")
    ax.set_xlabel("Selected Columns")
    ax.set_ylabel("Cumulative Runtime (seconds)")
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        output_dir / "runtimes.pdf",
        bbox_inches="tight",
    )

    fig.savefig(
        output_dir / "runtimes.png",
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
        marker = _marker_for_algorithm(algorithm)

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