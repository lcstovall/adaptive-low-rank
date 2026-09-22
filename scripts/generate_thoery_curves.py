from pathlib import Path
import pickle

import matplotlib.pyplot as plt
from adaptive_low_rank.datasets import load_dataset
import numpy as np


def compute_batchmax_alphas(results, n_candidates):
    """Compute BatchMax alpha values using the ``plot_alphas`` convention."""
    runs = [
        run
        for run in results
        if run["algorithm"] == "batch_max"
        and run["parameters"].get("n_candidates") == n_candidates
        and run["result"].gains_bm is not None
        and run["result"].gains_as is not None
    ]
    if not runs:
        raise ValueError(f"No BatchMax results found for n_candidates={n_candidates}")

    gains_bm = np.asarray([run["result"].gains_bm for run in runs], dtype=float)
    gains_as = np.asarray([run["result"].gains_as for run in runs], dtype=float)
    return np.nanmean(gains_bm, axis=0) / np.nanmean(gains_as, axis=0) - 1.0



def generate_theory_curves(X, max_k, alphas):
    svals_sq = np.linalg.svd(X, compute_uv=False) ** 2
    if len(alphas) < max_k:
        raise ValueError("alphas must contain at least max_k values")
    max_k = min(max_k, len(svals_sq) - 1)
    X_fro_sq = np.linalg.norm(X, ord="fro")**2
    bounds_as = []
    bounds_bm = []

    for k in range(1, max_k + 1):
        optimal_as = 1.0
        optimal_bm = 1.0
        alpha_mass = np.sum(alphas[: k])
        for r in range(1, min(k, len(svals_sq) - 1) + 1):
            phi_r = np.sum(svals_sq[r:])
            epsilon_a = bisect_as(k, r, X_fro_sq, phi_r)
            epsilon_b = bisect_bm(k, r, alpha_mass, X_fro_sq, phi_r)
            if epsilon_a is not None:
                opt_a = (1 + epsilon_a) * phi_r / X_fro_sq
                optimal_as = min(optimal_as, opt_a)
            if epsilon_b is not None:
                opt_b = (1 + epsilon_b) * phi_r / X_fro_sq
                optimal_bm = min(optimal_bm, opt_b)
        bounds_as.append(optimal_as)
        bounds_bm.append(optimal_bm)
    return bounds_as, bounds_bm


def optimal_trace_curve(X, max_k):
    """Return the relative trace error of the best rank-k approximation."""
    svals_sq = np.linalg.svd(X, compute_uv=False) ** 2
    max_k = min(max_k, len(svals_sq) - 1)
    tail_sums = np.cumsum(svals_sq[::-1])[::-1]
    iterations = np.arange(1, max_k + 1)
    return iterations, tail_sums[iterations] / svals_sq.sum()


def empirical_trace_curve(
    results, algorithm, max_k, n_candidates=None, first_run_only=True
):
    """Return empirical normalized Frobenius residuals for one algorithm."""
    runs = [
        run
        for run in results
        if run["algorithm"] == algorithm
        and (
            n_candidates is None
            or run["parameters"].get("n_candidates") == n_candidates
        )
    ]
    if not runs:
        raise ValueError(f"No results found for algorithm={algorithm}")

    if first_run_only:
        runs = runs[:1]

    residuals = np.asarray([run["result"].residuals for run in runs], dtype=float)
    initial_norms = np.asarray([run["init_res"] for run in runs], dtype=float)
    normalized_residuals = residuals / initial_norms[:, None]
    return np.arange(1, min(max_k, normalized_residuals.shape[1]) + 1), np.mean(
        normalized_residuals[:, :max_k], axis=0
    )


def plot_theory_curves(
    X, max_k, alphas, output_path=None, results=None, batchmax_n_candidates=500
):
    """Plot theory curves and optional empirical adaptive/BatchMax curves."""
    bounds_as, bounds_bm = generate_theory_curves(X, max_k, alphas)
    iterations = np.arange(1, len(bounds_as) + 1)
    optimal_iterations, optimal_curve = optimal_trace_curve(X, max_k)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.plot(
        optimal_iterations,
        np.sqrt(optimal_curve),
        color="black",
        linestyle=":",
        linewidth=2.0,
        label="Best rank-$k$ approximation",
    )
    ax.plot(iterations, np.sqrt(bounds_as), linewidth=2.0, label="Adaptive sampling")
    ax.plot(iterations, np.sqrt(bounds_bm), linewidth=2.0, label="BatchMax")

    if results is not None:
        adaptive_iterations, adaptive_empirical = empirical_trace_curve(
            results, "adaptive", max_k
        )
        batchmax_iterations, batchmax_empirical = empirical_trace_curve(
            results,
            "batch_max",
            max_k,
            batchmax_n_candidates,
            first_run_only=True,
        )
        ax.plot(
            adaptive_iterations,
            adaptive_empirical,
            color="tab:blue",
            linestyle="--",
            linewidth=1.8,
            label="Adaptive sampling (empirical)",
        )
        ax.plot(
            batchmax_iterations,
            batchmax_empirical,
            color="tab:orange",
            linestyle="--",
            linewidth=1.8,
            label="BatchMax (empirical)",
        )
    ax.set_xlabel("Number of iterations ($k$)")
    ax.set_ylabel(r"Normalized Residual $(\|R_k\|_F / \|R_0\|_F)$")
    ax.set_yscale("log")
    plotted_values = np.concatenate([line.get_ydata() for line in ax.lines])
    positive_values = plotted_values[plotted_values > 0]
    ax.set_ylim(
        bottom=max(positive_values.min() / 2.0, 1e-16),
        top=min(1.0, positive_values.max() * 2.0),
    )
    ax.legend(frameon=False)
    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig, ax


def bisect_as(k, r, X_fro_sq, phi_r, tol=1e-7):
    """Return the largest feasible epsilon for the adaptive-sampling bound.

    The feasible condition is

        r / epsilon + r * log(X_fro_sq / (epsilon * phi_r)) <= k.

    Its left-hand side is strictly decreasing for positive epsilon, so the
    optimal epsilon is the root at equality.  ``None`` means no epsilon in
    ``(0, 1]`` satisfies the iteration budget.
    """
    if k < 1 or r < 1 or X_fro_sq <= 0 or phi_r <= 0 or tol <= 0:
        raise ValueError("k, r, X_fro_sq, phi_r, and tol must be positive")

    def required_iterations(epsilon):
        return r / epsilon + r * np.log(X_fro_sq / (epsilon * phi_r))

    lower = tol
    upper = 1.0

    if required_iterations(upper) > k:
        return None

    for _ in range(100):
        if (
            upper - lower <= tol
            and abs(required_iterations(upper) - k) <= tol
        ):
            break

        midpoint = (lower + upper) / 2.0
        if required_iterations(midpoint) <= k:
            upper = midpoint
        else:
            lower = midpoint

    return upper


def bisect_bm(k, r, alpha_mass, X_fro_sq, phi_r, tol=1e-7):
    if k < 1 or r < 1 or X_fro_sq <= 0 or phi_r <= 0 or tol <= 0:
        raise ValueError("k, r, X_fro_sq, phi_r, and tol must be positive")

    def required_iterations(epsilon):
        return (
            r / epsilon
            + r * np.log(X_fro_sq / (epsilon * phi_r))
            - alpha_mass
        )

    lower = tol
    upper = 1.0

    if required_iterations(upper) > k:
        return None

    for _ in range(100):
        if (
            upper - lower <= tol
            and abs(required_iterations(upper) - k) <= tol
        ):
            break

        midpoint = (lower + upper) / 2.0
        if required_iterations(midpoint) <= k:
            upper = midpoint
        else:
            lower = midpoint

    return upper


if __name__ == "__main__":
    X = load_dataset("interactions")
    with open("results/interactions/results.pkl", "rb") as results_file:
        results = pickle.load(results_file)

    alphas = compute_batchmax_alphas(results, n_candidates=500)

    plot_theory_curves(
        X,
        max_k=120,
        alphas=alphas,
        results=results,
        batchmax_n_candidates=10,
        output_path="figures/interactions_theory.png",
    )
