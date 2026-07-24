from copy import deepcopy
from adaptive_low_rank.registry import ALGORITHMS

def benchmark(X, runs):
    """
    Execute a list of benchmark runs.
    """
    results = []
    for run in runs:
        algorithm = run["algorithm"]
        func = ALGORITHMS[algorithm]
        kwargs = deepcopy(run)
        kwargs.pop("algorithm")
        k = kwargs.pop("k")
        result = func(
            X,
            k,
            **kwargs,
        )
        results.append({
            "algorithm": algorithm,
            "parameters": kwargs | {"k": k},
            "result": result,
        })
    return results