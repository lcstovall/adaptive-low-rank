import time
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path


from adaptive_low_rank.algorithms import (
    Adaptive,
    BatchMax,
    Greedy,
    GreedyPP,
    Random,
)


def generate_svd_dataset(n, d, rank=20, random_state=0):
    rng = np.random.RandomState(random_state)

    U, _ = np.linalg.qr(rng.randn(d, rank))

    V = rng.randn(n, rank)
    V /= np.linalg.norm(V, axis=0, keepdims=True)

    singular_values = np.linspace(1.0, 0.1, rank)

    return U @ np.diag(singular_values) @ V.T


def benchmark_runtime(
    n_values,
    d,
    k=100,
    repeats=3,
    rank=20,
):
    algorithms = {
        "Adaptive": Adaptive(),
        "BatchMax": BatchMax(),
        "Greedy": Greedy(),
        "Greedy++": GreedyPP(),
        "Random": Random(),
    }

    runtimes = {
        name: np.zeros((repeats, len(n_values)))
        for name in algorithms
    }

    for n_idx, n in enumerate(n_values):

        print(f"d={d}, n={n}")

        X = generate_svd_dataset(
            n=n,
            d=d,
            rank=rank,
            random_state=0,
        )

        for name, algorithm in algorithms.items():

            for trial in range(repeats):

                random_state = trial

                start = time.perf_counter()

                result = algorithm.select_columns(
                    X,
                    k=k,
                    random_state=random_state,
                )

                elapsed = time.perf_counter() - start

                runtimes[name][trial, n_idx] = result.runtimes[-1]

                print(
                    f"  {name}: "
                    f"{elapsed:.4f} s"
                )

    return runtimes

def plot_runtime_scaling(
    n_values,
    runtimes,
    d,
    output_dir,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(
        figsize=(9, 6),
        dpi=300,
    )

    for name, values in runtimes.items():

        mean = values.mean(axis=0)
        std = values.std(axis=0)

        ax.plot(
            n_values,
            mean,
            marker="o",
            label=name,
        )

        ax.fill_between(
            n_values,
            mean - std,
            mean + std,
            alpha=0.2,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel("Number of columns (n)")
    ax.set_ylabel("Total runtime (seconds)")

    ax.set_title(
        f"Runtime scaling with number of columns (d={d}, k=100)"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        output_dir / f"runtime_scaling_d{d}.png",
        bbox_inches="tight",
    )

    fig.savefig(
        output_dir / f"runtime_scaling_d{d}.pdf",
        bbox_inches="tight",
    )

    plt.close(fig)

if __name__ == "__main__":

    n_values = [
        500,
        1_000,
        5_000,
        10_000,
        50_000,
        100_000,
    ]

    dimensions = [
        50,
        100,
        500,
        1000,
        5000,
    ]

    output_dir = Path("results/runtime_scaling")

    for d in dimensions:

        runtimes = benchmark_runtime(
            n_values=n_values,
            d=d,
            k=100,
            repeats=1,
            rank=20,
        )

        plot_runtime_scaling(
            n_values=n_values,
            runtimes=runtimes,
            d=d,
            output_dir=output_dir,
        )