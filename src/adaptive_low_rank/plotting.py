from collections import defaultdict
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator

from adaptive_low_rank.theory import empirical_trace_curve, generate_theory_curves
from adaptive_low_rank.theory import optimal_trace_curve

plt.rcParams.update(
    {"font.family": "serif", "mathtext.fontset": "cm", "axes.unicode_minus": False}
)

# Global font sizes for plot text and legends.
FONT_SIZE = 20
LEGEND_FONT_SIZE = 18

METHOD_MARKERS = {
    "adaptive": "o",
    "batch_max": "s",
    "greedy": "^",
    "greedy_pp": "D",
    "random": "x",
}
_assigned_markers = dict(METHOD_MARKERS)


def _is_batch_max(algorithm):
    """Return whether an algorithm name identifies Batch-Max sampling."""
    return algorithm.replace("_", "").lower() == "batchmax"


def _algorithm_label(algorithm):
    """Return the legend label for an algorithm name."""
    if algorithm == "greedy_pp":
        return "Greedy++"
    if _is_batch_max(algorithm):
        return "Batch-Max"
    return algorithm.replace("_", " ").title()


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
    logx=False,
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
        Batch-Max candidate count to plot. If omitted, only the first
        ``n_candidates`` value encountered for Batch-Max is plotted. Other
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

        label = _algorithm_label(algorithm)

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

    # Headroom is a fixed fraction of the log-scale span of the data.
    plotted_values = np.concatenate([line.get_ydata() for line in ax.lines])
    plotted_values = plotted_values[np.isfinite(plotted_values) & (plotted_values > 0)]
    log_span = np.log10(plotted_values.max() / plotted_values.min())
    ax.set_ylim(
        bottom=ax.get_ylim()[0], top=plotted_values.max() * 10 ** (0.04 * log_span)
    )

    if logx:
        ax.set_xscale("log")

    fig.canvas.draw()
    labeled_minor_ticks = [
        tick.get_loc() for tick in ax.yaxis.get_minor_ticks() if tick.label1.get_text()
    ]
    ax.yaxis.set_minor_locator(FixedLocator(labeled_minor_ticks))

    ax.set_xlabel(r"Number of Selected Columns ($k$)", fontsize=FONT_SIZE)
    ax.set_ylabel(r"Normalized Residual $(\|R_k\|_F / \|R_0\|_F)$", fontsize=FONT_SIZE)
    ax.tick_params(axis="both", which="both", labelsize=FONT_SIZE)

    ax.grid(True, which="both", axis="y")

    is_interactions_plot = name == "interactions"
    legend_loc = "lower left" if is_interactions_plot else "upper right"
    legend_anchor = (0.025, 0.04) if is_interactions_plot else (0.975, 0.96)

    ax.legend(
        loc=legend_loc,
        bbox_to_anchor=legend_anchor,
        borderaxespad=0,
        fontsize=LEGEND_FONT_SIZE,
        frameon=True,
        framealpha=1.0,
        facecolor="white",
        edgecolor="black",
    )

    fig.subplots_adjust(right=0.95, left=0.10, bottom=0.12, top=0.97)

    fig.savefig(output_dir / f"{name}_residuals.png", dpi=300, bbox_inches="tight")

    plt.show()


def plot_theory_curves(
    X,
    max_k,
    alphas,
    output_path=None,
    results=None,
    batchmax_n_candidates=500,
    legend_loc="lower left",
):
    """Plot theory curves and optional empirical adaptive/Batch-Max curves.

    ``legend_loc`` is ``"lower left"`` (default) or ``"upper right"``.
    """
    bounds_as, bounds_bm = generate_theory_curves(X, max_k, alphas)
    iterations = np.arange(1, len(bounds_as) + 1)
    optimal_iterations, optimal_curve = optimal_trace_curve(X, max_k)

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    def residual_color(index):
        base = np.array(mcolors.to_rgb(color_cycle[index]))
        return 0.85 * base + 0.15 * np.ones(3)

    adaptive_color = residual_color(0)
    batchmax_color = residual_color(1)

    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
    ax.plot(
        optimal_iterations,
        np.sqrt(optimal_curve),
        color="black",
        linestyle="--",
        linewidth=2.0,
        label="Rank-$k$ trunc. SVD",
    )
    ax.plot(
        iterations,
        np.sqrt(bounds_as),
        color=adaptive_color,
        linestyle=":",
        linewidth=2.0,
        label="Adaptive (theory)",
    )
    ax.plot(
        iterations,
        np.sqrt(bounds_bm),
        color=batchmax_color,
        linestyle=":",
        linewidth=2.0,
        label="Batch-Max (theory)",
    )

    if results is not None:
        adaptive_iterations, adaptive_empirical = empirical_trace_curve(
            results, "adaptive", max_k
        )
        batchmax_iterations, batchmax_empirical = empirical_trace_curve(
            results, "batch_max", max_k, batchmax_n_candidates, first_run_only=True
        )
        for x, y, color, marker, label in [
            (
                adaptive_iterations,
                adaptive_empirical,
                adaptive_color,
                "o",
                "Adaptive (empirical)",
            ),
            (
                batchmax_iterations,
                batchmax_empirical,
                batchmax_color,
                "s",
                "Batch-Max (empirical)",
            ),
        ]:
            ax.plot(
                x,
                y,
                color=color,
                linewidth=2.0,
                marker=marker,
                markevery=np.arange(0, len(x), max(int(np.ceil(len(x) / 20)), 1)),
                markersize=8,
                markeredgewidth=3,
                label=label,
            )

    ax.set_yscale("log")
    plotted_values = np.concatenate([line.get_ydata() for line in ax.lines])
    positive_values = plotted_values[plotted_values > 0]
    y_min, y_max = positive_values.min(), positive_values.max()
    log_span = np.log10(y_max / y_min)
    ax.set_ylim(bottom=max(y_min / 2.0, 1e-16), top=y_max * 10 ** (0.04 * log_span))

    fig.canvas.draw()
    labeled_minor_ticks = [
        tick.get_loc() for tick in ax.yaxis.get_minor_ticks() if tick.label1.get_text()
    ]
    ax.yaxis.set_minor_locator(FixedLocator(labeled_minor_ticks))

    ax.set_xlabel(r"Number of Selected Columns ($k$)", fontsize=FONT_SIZE)
    ax.set_ylabel(r"Normalized Residual $(\|R_k\|_F / \|R_0\|_F)$", fontsize=FONT_SIZE)
    ax.tick_params(axis="both", which="both", labelsize=FONT_SIZE)

    ax.grid(True, which="both", axis="y")

    legend_anchor = (0.975, 0.96) if legend_loc == "upper right" else (0.025, 0.04)
    handles, labels = ax.get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda i: labels[i].startswith("Rank-"))
    ax.legend(
        [handles[i] for i in order],
        [labels[i] for i in order],
        loc=legend_loc,
        bbox_to_anchor=legend_anchor,
        borderaxespad=0,
        fontsize=LEGEND_FONT_SIZE,
        frameon=True,
        framealpha=1.0,
        facecolor="white",
        edgecolor="black",
    )
    fig.subplots_adjust(right=0.95, left=0.10, bottom=0.12, top=0.97)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig, ax


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

        ax.plot(x, mean, marker="o", label=_algorithm_label(algorithm))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Runtime (seconds)")
    ax.set_title(title)
    ax.tick_params(axis="both", labelsize=11)

    ax.grid(True, which="major", axis="y", alpha=0.3)

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        fontsize=LEGEND_FONT_SIZE,
        frameon=False,
    )

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
        the varying dimension (``n`` or ``d``), with Batch-Max and Greedy
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
        relative_runtimes.mean(axis=1).T, index=index, columns=keep_algorithms
    )
    std_df = pd.DataFrame(
        relative_runtimes.std(axis=1).T, index=index, columns=keep_algorithms
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
            col: [
                f"{fmt(m)} \u00b1 {fmt(s)}" for m, s in zip(mean_df[col], std_df[col])
            ]
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

    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)

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

        if "n_candidates" in params:
            label = f"$n_{{candidates}}={params['n_candidates']}$"
        else:
            label = _algorithm_label(algorithm)

        ax.plot(
            x,
            mean,
            color=color,
            marker=marker,
            markevery=_marker_positions(len(x), max_markers=30),
            markersize=6,
            label=label,
        )

    ax.set_xlabel(r"Number of Selected Columns ($k$)", fontsize=FONT_SIZE)
    ax.set_ylabel(r"Batch-Max Improvement, $\alpha$", fontsize=FONT_SIZE)

    ax.tick_params(axis="both", which="both", labelsize=FONT_SIZE)

    ax.grid(True, which="major", axis="y", alpha=0.3)

    ax.legend(
        loc="upper right",
        bbox_to_anchor=(0.975, 0.96),
        borderaxespad=0,
        fontsize=LEGEND_FONT_SIZE,
        frameon=True,
        framealpha=1.0,
        facecolor="white",
        edgecolor="black",
    )

    fig.subplots_adjust(right=0.95, left=0.10, bottom=0.12, top=0.97)

    fig.savefig(output_dir / f"{output_name}.png", dpi=300, bbox_inches="tight")

    plt.show()
