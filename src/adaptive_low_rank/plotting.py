from collections import defaultdict
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator

plt.rcParams.update(
    {"font.family": "serif", "mathtext.fontset": "cm", "axes.unicode_minus": False}
)

METHOD_MARKERS = {
    "adaptive": "o",
    "batch_max": "s",
    "greedy": "^",
    "greedy_pp": "D",
    "random": "x",
}
_assigned_markers = dict(METHOD_MARKERS)


def _is_batch_max(algorithm):
    """Return whether an algorithm name identifies batch-max sampling."""
    return algorithm.replace("_", "").lower() == "batchmax"


def _marker_for_algorithm(algorithm):
    """Return the persistent marker assigned to an algorithm name."""
    if algorithm in _assigned_markers:
        return _assigned_markers[algorithm]

    marker_cycle = ["o", "s", "^", "D", "*", "P", "v", "<", ">", "h"]
    known_markers = set(_assigned_markers.values())
    available_markers = [
        marker for marker in marker_cycle if marker not in known_markers
    ]
    marker = available_markers[0] if available_markers else "o"
    _assigned_markers[algorithm] = marker
    return marker


def _marker_positions(length, max_markers=20):
    """Return approximately evenly spaced marker positions for a curve."""
    spacing = max(int(np.ceil(length / max_markers)), 1)
    return np.arange(0, length, spacing)


def plot_residuals(
    results,
    output_dir,
    name="residuals",
    n_candidates=None,
    compute_optimal=False,
):
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

    n_candidates : int, optional
        Batch Max candidate count to plot. If omitted, only the first
        ``n_candidates`` value encountered for Batch Max is plotted. Other
        algorithms are unaffected.

    compute_optimal : bool, default=False
        Whether to compute and plot the optimal rank-k SVD residual curve.
        When enabled, ``name`` must be the dataset name understood by
        :func:`adaptive_low_rank.datasets.load_dataset`.

    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if n_candidates is None:
        n_candidates = next(
            (
                run["parameters"]["n_candidates"]
                for run in results
                if _is_batch_max(run["algorithm"])
                and "n_candidates" in run["parameters"]
            ),
            None,
        )

    grouped = defaultdict(list)

    # Group repeated runs after removing the random seed from the key.
    for run in results:
        if (
            _is_batch_max(run["algorithm"])
            and n_candidates is not None
            and run["parameters"].get("n_candidates") != n_candidates
        ):
            continue

        params = run["parameters"].copy()
        params.pop("random_state", None)
        key = (run["algorithm"], tuple(sorted(params.items())))
        grouped[key].append({"result": run["result"], "init_res": run["init_res"]})

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    algorithm_names = sorted({alg for alg, _ in grouped})
    base_colors = {
        alg: color_cycle[i % len(color_cycle)] for i, alg in enumerate(algorithm_names)
    }

    algorithm_counts = defaultdict(int)
    for algorithm, _ in grouped:
        algorithm_counts[algorithm] += 1

    grouped = dict(sorted(grouped.items(), key=lambda item: _is_batch_max(item[0][0])))
    algorithm_indices = defaultdict(int)

    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)

    for (algorithm, params), runs in grouped.items():
        residuals = np.array([run["result"].residuals for run in runs])
        init_res = np.array([run["init_res"] for run in runs])

        residuals = residuals / init_res[:, None]

        mean = residuals.mean(axis=0)
        x = np.arange(1, len(mean) + 1)

        base = np.array(mcolors.to_rgb(base_colors[algorithm]))
        n = algorithm_counts[algorithm]
        i = algorithm_indices[algorithm]
        algorithm_indices[algorithm] += 1

        # Use lighter shades for additional parameter settings.
        t = 0.15 + 0.55 * i / max(n - 1, 1)
        color = (1 - t) * base + t * np.ones(3)
        marker = _marker_for_algorithm(algorithm)

        label = (
            "Greedy++"
            if algorithm == "greedy_pp"
            else algorithm.replace("_", " ").title()
        )

        ax.plot(
            x,
            mean,
            color=color,
            linewidth=2.0,
            marker=marker,
            markevery=_marker_positions(len(x)),
            markersize=8,
            markeredgewidth=3,
            label=label,
        )

        if len(runs) > 1:
            std = residuals.std(axis=0)

            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.10)

    if compute_optimal:
        from adaptive_low_rank.datasets import load_dataset

        X = load_dataset(name)
        singular_values = np.linalg.svd(X, compute_uv=False)
        residual_squared = np.concatenate(
            (
                [np.sum(singular_values**2)],
                np.cumsum(singular_values[::-1] ** 2)[::-1][1:],
            )
        )
        optimal = np.sqrt(residual_squared)
        max_rank = max(len(run["result"].residuals) for run in results)
        optimal = optimal[: max_rank + 1] / optimal[0]
        optimal_values = optimal[1:]
        optimal_values[optimal_values <= 0] = np.nan

        ax.plot(
            np.arange(1, len(optimal_values) + 1),
            optimal_values,
            color="black",
            linestyle="--",
            linewidth=2.0,
            label="Rank-$k$ trunc. SVD",
        )

    ax.set_yscale("log")

    fig.canvas.draw()
    labeled_minor_ticks = [
        tick.get_loc() for tick in ax.yaxis.get_minor_ticks() if tick.label1.get_text()
    ]
    ax.yaxis.set_minor_locator(FixedLocator(labeled_minor_ticks))

    ax.set_xlabel(r"Number of Selected Columms ($k$)", fontsize=13)
    ax.set_ylabel(r"Normalized Residual $(\|R_k\|_F / \|R_0\|_F)$", fontsize=13)

    ax.grid(True, which="both", axis="y")

    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=10, frameon=False)

    fig.subplots_adjust(right=0.75, left=0.10, bottom=0.12, top=0.97)

    fig.savefig(output_dir / f"{name}_residuals.png", dpi=300, bbox_inches="tight")

    plt.show()


def plot_runtime_scaling(
    results, output_dir, fixed_d=None, fixed_n=None, name="runtime_scaling"
):
    """Plot mean runtime across repeated benchmark runs.

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
        range(len(algorithms)), key=lambda index: _is_batch_max(algorithms[index])
    )

    for algorithm_index in algorithm_order:
        algorithm = algorithms[algorithm_index]
        mean = means[algorithm_index]

        ax.plot(x, mean, marker="o", label=algorithm)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Runtime (seconds)")
    ax.set_title(title)
    ax.tick_params(axis="both", labelsize=11)

    ax.grid(True, which="major", axis="y", alpha=0.3)

    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=10, frameon=False)

    fig.subplots_adjust(right=0.75, left=0.10, bottom=0.12, top=0.92)

    fig.savefig(output_dir / f"{name}.png", dpi=300, bbox_inches="tight")

    plt.show()


def runtime_table(results, fixed_d=None, fixed_n=None):
    """Tabulate mean and std runtime for a fixed-d or fixed-n cross section.

    Parameters
    ----------
    results : dict
        Serialized output from the runtime scaling benchmark.

    fixed_d : int, optional
        Fix the number of rows and tabulate runtime as ``n`` varies.

    fixed_n : int, optional
        Fix the number of columns and tabulate runtime as ``d`` varies.

    Returns
    -------
    mean_df, std_df : pandas.DataFrame
        Runtime relative to Adaptive, averaged/std'd over repeats, indexed by
        the varying dimension (``n`` or ``d``), with BatchMax and Greedy
        columns.
    """

    if (fixed_d is None) == (fixed_n is None):
        raise ValueError("Specify exactly one of fixed_d or fixed_n.")

    algorithms = list(results["algorithms"])
    runtime_matrix = np.asarray(results["runtimes"])

    if "Adaptive" not in algorithms:
        raise ValueError("Runtime results must include an Adaptive algorithm.")

    adaptive_index = algorithms.index("Adaptive")
    keep_algorithms = [
        algorithm for algorithm in ("BatchMax", "Greedy") if algorithm in algorithms
    ]
    keep_indices = [algorithms.index(algorithm) for algorithm in keep_algorithms]

    if fixed_d is not None:
        dimension_index = results["dimensions"].index(fixed_d)
        runtimes = runtime_matrix[:, dimension_index, :, :]
        index = pd.Index(results["n_values"], name="n")
    else:
        n_index = results["n_values"].index(fixed_n)
        runtimes = runtime_matrix[:, :, :, n_index].transpose(0, 2, 1)
        index = pd.Index(results["dimensions"], name="d")

    adaptive_runtimes = runtimes[adaptive_index]
    relative_runtimes = runtimes / adaptive_runtimes[None, ...]
    relative_runtimes = relative_runtimes[keep_indices]

    mean_df = pd.DataFrame(
        relative_runtimes.mean(axis=1).T,
        index=index,
        columns=keep_algorithms,
    )
    std_df = pd.DataFrame(
        relative_runtimes.std(axis=1).T,
        index=index,
        columns=keep_algorithms,
    )

    return mean_df, std_df


def style_runtime_table(mean_df, std_df, sig_figs=3, caption=None):
    """Format a runtime cross section for publication display/export.

    Values are rendered as ``mean \u00b1 std`` at a fixed number of
    significant figures.

    Parameters
    ----------
    mean_df, std_df : pandas.DataFrame
        Output of :func:`runtime_table`.

    sig_figs : int, default=3
        Number of significant figures used to format each value.

    caption : str, optional
        Table caption, used for notebook display and LaTeX export.

    Returns
    -------
    pandas.io.formats.style.Styler
        Styled table of ``mean \u00b1 std`` strings, ready to display in a
        notebook or export via ``.to_latex(hrules=True, convert_css=True)``.
    """

    def fmt(value):
        return f"{value:.{sig_figs}g}"

    display_df = pd.DataFrame(
        {
            col: [f"{fmt(m)} \u00b1 {fmt(s)}" for m, s in zip(mean_df[col], std_df[col])]
            for col in mean_df.columns
        },
        index=mean_df.index,
    )

    styler = display_df.style
    if caption is not None:
        styler = styler.set_caption(caption)

    return styler


def plot_alphas(results, output_dir, name="alphas"):
    """
    Plot alpha values over the iterations.

    Runs are grouped by algorithm and n_candidates, while runs
    differing only in random_state are averaged together.

    Parameters
    ----------
    results : list
        Output of benchmark().

    output_dir : str or Path
        Directory in which to save the figures.

    name : str
        Name of the output figure.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if name.endswith(("_alpha", "_alphas")):
        output_name = name
    else:
        output_name = f"{name}_alphas"

    grouped = defaultdict(list)

    for run in results:

        if run["result"].gains_bm is None or run["result"].gains_as is None:
            continue

        if np.all(np.isnan(run["result"].gains_bm)):
            continue

        params = run["parameters"].copy()
        params.pop("random_state", None)

        key = (run["algorithm"], tuple(sorted(params.items())))

        grouped[key].append(run["result"])

    if not grouped:
        return

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    algorithm_names = sorted({algorithm for algorithm, _ in grouped})

    base_colors = {
        algorithm: color_cycle[i % len(color_cycle)]
        for i, algorithm in enumerate(algorithm_names)
    }

    algorithm_counts = defaultdict(int)

    for algorithm, _ in grouped:
        algorithm_counts[algorithm] += 1

    # Sort groups by algorithm and then by the candidate count.
    def sort_key(item):
        (algorithm, params), _ = item

        params_dict = dict(params)

        n_candidates = params_dict.get("n_candidates", 0)

        return (algorithm, n_candidates)

    grouped = dict(sorted(grouped.items(), key=sort_key))

    # Track the shade used for each parameter setting.
    algorithm_indices = defaultdict(int)
    line_markers = ["o", "s", "^", "D", "*", "P", "v", "<", ">", "h"]

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    for line_index, ((algorithm, params), runs) in enumerate(grouped.items()):

        params = dict(params)

        gains_bm = np.array([r.gains_bm for r in runs], dtype=float)
        gains_as = np.array([r.gains_as for r in runs], dtype=float)

        mean_gains_bm = np.nanmean(gains_bm, axis=0)
        mean_gains_as = np.nanmean(gains_as, axis=0)
        mean = mean_gains_bm / mean_gains_as - 1
        x = np.arange(1, len(mean) + 1)

        base = np.array(mcolors.to_rgb(base_colors[algorithm]))

        n = algorithm_counts[algorithm]
        i = algorithm_indices[algorithm]
        algorithm_indices[algorithm] += 1

        # Use lighter shades for additional candidate-count settings.
        t = 0.08 + 0.55 * i / max(n - 1, 1)

        color = (1 - t) * base + t * np.ones(3)

        marker = line_markers[line_index % len(line_markers)]

        label = (
            "Greedy++"
            if algorithm == "greedy_pp"
            else algorithm.replace("_", " ").title()
        )

        if "n_candidates" in params:
            label += f" ($n_{{candidates}}={params['n_candidates']}$)"

        ax.plot(
            x,
            mean,
            color=color,
            marker=marker,
            markevery=_marker_positions(len(x), max_markers=30),
            markersize=6,
            label=label,
        )

    ax.set_xlabel("Selected Rows")
    ax.set_ylabel(r"$\alpha$")

    ax.tick_params(axis="both", labelsize=11)

    ax.grid(True, which="major", axis="y", alpha=0.3)

    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=10, frameon=False)

    fig.subplots_adjust(right=0.75, left=0.10, bottom=0.12, top=0.97)

    fig.savefig(output_dir / f"{output_name}.png", dpi=300, bbox_inches="tight")

    plt.show()
