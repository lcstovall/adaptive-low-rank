from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class AlgorithmResult:
    indices: np.ndarray
    residuals: np.ndarray
    runtimes: np.ndarray
    alphas: list[Optional[float]]