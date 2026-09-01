"""
Adaptive Low-Rank Matrix Approximation.

Algorithms and utilities for benchmarking adaptive column subset
selection methods.
"""

from .algorithms import Adaptive, BatchMax, Greedy, GreedyPP, LowRankAlgorithm, Random
from .benchmark import benchmark
from .datasets import load_dataset

__version__ = "0.1.0"

__all__ = [
    "Adaptive",
    "BatchMax",
    "Greedy",
    "GreedyPP",
    "LowRankAlgorithm",
    "Random",
    "benchmark",
    "load_dataset",
]
