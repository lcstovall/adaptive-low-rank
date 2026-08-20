from itertools import product


def generate_runs(experiment):
    """Expand parameter grids into individual benchmark specifications.

    Parameters
    ----------
    experiment : dict
        Experiment configuration containing an ``algorithms`` mapping. Scalar
        parameter values are treated as one-element grids.

    Returns
    -------
    list of dict
        One run specification for every Cartesian-product combination.
    """
    runs = []

    for algorithm, params in experiment["algorithms"].items():

        params = {
            key: value if isinstance(value, list) else [value]
            for key, value in params.items()
        }
        keys = list(params.keys())

        for values in product(*(params[k] for k in keys)):
            run = {"algorithm": algorithm}
            run.update(dict(zip(keys, values)))
            runs.append(run)

    return runs
