import time
import pickle
import numpy as np

from pathlib import Path


from adaptive_low_rank.algorithms import (
    Adaptive,
    BatchMax,
    Greedy,
    GreedyPP,
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

if __name__ == "__main__":

    n_values = [
        500,
        1_000,
        5_000,
        10_000,
        50_000,
    ]

    dimensions = [
        50,
        100,
        500,
        1000,
    ]

    output_dir = Path("results/runtime_scaling")
    output_dir.mkdir(parents=True, exist_ok=True)

    algorithm_names = ["Adaptive", "BatchMax", "Greedy", "Greedy++"]
    repeats = 10
    runtime_matrix = np.empty(
        (
            len(algorithm_names),
            len(dimensions),
            repeats,
            len(n_values),
        )
    )

    for d in dimensions:

        runtimes = benchmark_runtime(
            n_values=n_values,
            d=d,
            k=100,
            repeats=repeats,
            rank=20,
        )

        dimension_index = dimensions.index(d)
        for algorithm_index, name in enumerate(algorithm_names):
            runtime_matrix[algorithm_index, dimension_index] = runtimes[name]

    mean_runtimes = runtime_matrix.mean(axis=2)

    with open(output_dir / "runtime_scaling.pkl", "wb") as file:
        pickle.dump(
            {
                "algorithms": algorithm_names,
                "n_values": n_values,
                "dimensions": dimensions,
                "runtimes": runtime_matrix,
                "mean_runtimes": mean_runtimes,
            },
            file,
        )