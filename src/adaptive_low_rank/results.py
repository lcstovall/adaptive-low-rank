from dataclasses import dataclass
import numpy as np

@dataclass
class AlgorithmResult:
    rows: np.ndarray
    indices: np.ndarray
    residuals: np.ndarray
    runtimes: np.ndarray
    residual_matrix: np.ndarray