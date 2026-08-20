from copy import deepcopy
from adaptive_low_rank.registry import ALGORITHMS
from numpy.linalg import norm


def benchmark(X, runs, V=None):
    """Execute a collection of configured benchmark runs.

    Parameters
    ----------
    X : np.ndarray
        Input matrix with shape ``(d, n)``.

    runs : list[dict]
        Run specifications. Each dictionary must contain ``algorithm`` and
        ``k`` and may contain algorithm-specific keyword arguments.

    V : np.ndarray | None
        Truncated right singular vectors used for alpha computation when
        ``compute_alpha=True``.

    Returns
    -------
    list of dict
        One result dictionary per run, containing the algorithm name,
        parameters, initial squared Frobenius norm, and ``AlgorithmResult``.
    """
    results = []

    X_fro_norm2 = norm(X, ord="fro") ** 2

    for run in runs:
        params = deepcopy(run)

        algorithm_name = params.pop("algorithm")
        k = params.pop("k")

        Algorithm = ALGORITHMS[algorithm_name]
        algorithm = Algorithm()

        result = algorithm.select_columns(X, k=k, V=V, **params)

        results.append(
            {
                "algorithm": algorithm_name,
                "parameters": {"k": k, **params},
                "init_res": X_fro_norm2,
                "result": result,
            }
        )

    return results
