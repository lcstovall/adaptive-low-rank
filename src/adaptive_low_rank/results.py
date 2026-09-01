from dataclasses import dataclass

import numpy as np


@dataclass
class AlgorithmResult:
    """Results recorded during one column-selection run.

    Attributes
    ----------
    indices : np.ndarray
        Selected column indices in selection order.
    residuals : np.ndarray
        Residual Frobenius norms after each selection.
    runtimes : np.ndarray
        Cumulative elapsed times after each selection, or an empty array when
        runtime measurement is disabled.
    alphas : np.ndarray
        Alpha diagnostics recorded after each selection; entries are NaN when
        an alpha diagnostic is not available.
    """

    indices: np.ndarray
    residuals: np.ndarray
    runtimes: np.ndarray
    alphas: list[float | None]
