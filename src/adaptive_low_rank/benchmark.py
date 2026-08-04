# from copy import deepcopy
# from adaptive_low_rank.registry import ALGORITHMS

# def benchmark(X, runs):
#     """
#     Execute a list of benchmark runs.
#     """
#     results = []
#     for run in runs:
#         algorithm = run["algorithm"]
#         func = ALGORITHMS[algorithm]
#         kwargs = deepcopy(run)
#         kwargs.pop("algorithm")
#         k = kwargs.pop("k")
#         result = func(
#             X,
#             k,
#             **kwargs,
#         )
#         results.append({
#             "algorithm": algorithm,
#             "parameters": kwargs | {"k": k},
#             "result": result,
#         })
#     return results

from copy import deepcopy

from adaptive_low_rank.registry import ALGORITHMS


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
                "result": result,
            }
        )

    return results