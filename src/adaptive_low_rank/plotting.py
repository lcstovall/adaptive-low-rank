import matplotlib.pyplot as plt
import numpy as np

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


def plot_residuals(results):
    """
    Plot normalized residual curves.

    Algorithms with multiple random seeds are averaged and shown with
    ±1 standard deviation. Deterministic algorithms are plotted as a
    single curve.

    Parameters
    ----------
    results : list
        Output of benchmark().
    """
    grouped = defaultdict(list)
    # Group runs that differ only by random_state
    for run in results:
        params = run["parameters"].copy()
        params.pop("random_state", None)
        key = (
            run["algorithm"],
            tuple(sorted(params.items())),
        )
        grouped[key].append(run["result"])

    plt.figure()

    for (algorithm, params), runs in grouped.items():
        residuals = np.array(
            [r.residuals for r in runs]
        )

        # Normalize every trial by its initial residual
        residuals = residuals / residuals[:, [0]]
        mean = residuals.mean(axis=0)
        x = np.arange(1, len(mean) + 1)

        # Build legend label
        params = dict(params)
        extras = []
        for name, value in params.items():
            extras.append(f"{name}={value}")
        label = algorithm.replace("_", " ").title()
        if extras:
            label += " (" + ", ".join(extras) + ")"
        if len(runs) == 1:
            plt.plot(x, mean, label=label)
        else:
            std = residuals.std(axis=0)
            plt.plot(x, mean, label=label)
            plt.fill_between(
                x,
                mean - std,
                mean + std,
                alpha=0.2,
            )
    plt.yscale("log")
    plt.xlabel("Selected Rows")
    plt.ylabel("Normalized Residual")
    plt.legend()
    plt.tight_layout()
    plt.show()