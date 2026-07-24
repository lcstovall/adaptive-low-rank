from itertools import product


def generate_runs(experiment):
    """
    Expand an experiment configuration into a list of individual runs.
    """
    runs = []
    for algorithm, params in experiment["algorithms"].items():
        # Every parameter should be iterable
        params = {
            key: value if isinstance(value, list) else [value]
            for key, value in params.items()
        }
        keys = list(params.keys())
        for values in product(*(params[k] for k in keys)):
            run = {
                "algorithm": algorithm
            }
            run.update(dict(zip(keys, values)))
            runs.append(run)
    return runs