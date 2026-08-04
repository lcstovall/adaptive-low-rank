"""
Adaptive Low-Rank Matrix Approximation.

Algorithms and utilities for benchmarking adaptive column subset
selection methods.
"""

from .algorithms import (
    LowRankAlgorithm,
    Adaptive,
    BatchMax,
    Greedy,
    GreedyPP,
    Random,
)

from .benchmark import benchmark
from .datasets import load_dataset

__version__ = "0.1.0"

__all__ = [
    "LowRankAlgorithm",
    "Adaptive",
    "BatchMax",
    "Greedy",
    "GreedyPP",
    "Random",
    "benchmark",
    "load_dataset",
]