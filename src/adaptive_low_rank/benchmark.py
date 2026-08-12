from copy import deepcopy
from adaptive_low_rank.registry import ALGORITHMS
from numpy.linalg import norm

def benchmark(X, runs, V=None):
    """
    Execute a list of benchmark runs.

    Parameters
    ----------
    X : np.ndarray
        Data matrix.
    runs : list[dict]
        List of experiment dictionaries.
    V : np.ndarray | None
        Truncated right singular vectors. If None, alpha values are not
        computed.

    Returns
    -------
    list[dict]
        Benchmark results.
    """
    results = []
    X_fro_norm2 = norm(X, ord='fro')**2.

    for run in runs:
        params = deepcopy(run)

        algorithm_name = params.pop("algorithm")
        k = params.pop("k")

        Algorithm = ALGORITHMS[algorithm_name]
        algorithm = Algorithm()

        result = algorithm.select_columns(
            X,
            n_clusters=k,
            V=V,
            **params,
        )

        results.append(
            {
                "algorithm": algorithm_name,
                "parameters": {"k": k, **params},
                "init_res": X_fro_norm2,
                "result": result,
            }
        )

    return results