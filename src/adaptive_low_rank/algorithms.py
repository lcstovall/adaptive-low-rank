from __future__ import annotations
import time
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
from adaptive_low_rank.results import AlgorithmResult


class LowRankAlgorithm(ABC):
    """Base class for low-rank column selection algorithms."""

    name = "base"

    def select_columns(
        self,
        X: np.ndarray,
        n_clusters: int,
        random_state: Optional[np.random.RandomState | int] = None,
        n_candidates: Optional[int] = None,
        V: Optional[np.ndarray] = None,
    ) -> AlgorithmResult:
        """Select columns from X and return the resulting approximation data."""

        # Work directly with X.
        R = self._as_matrix(X.copy())

        indices = np.full(n_clusters, -1, dtype=int)
        residuals: list[float] = []
        alphas: list[Optional[float]] = []
        times: list[float] = []

        start = time.perf_counter()

        # Allow either an integer seed or an existing RandomState.
        if random_state is None:
            rng = np.random.RandomState()
        elif isinstance(random_state, (int, np.integer)):
            rng = np.random.RandomState(random_state)
        else:
            rng = random_state

        for c in range(n_clusters):
            idx, alpha = self.select_index(
                R,
                n_clusters,
                rng,
                n_candidates,
                V,
            )

            indices[c] = idx
            alphas.append(alpha)

            # Project out the selected COLUMN.
            R = self._project_out(R, R[:, idx])

            times.append(time.perf_counter() - start)
            residuals.append(float(np.linalg.norm(R, ord="fro")))

        return AlgorithmResult(
            indices=indices,
            residuals=np.asarray(residuals),
            runtimes=np.asarray(times),
            alphas=np.asarray(alphas, dtype=float),
        )

    @abstractmethod
    def select_index(
        self,
        R: np.ndarray,
        n_clusters: int,
        random_state: np.random.RandomState,
        n_candidates: Optional[int] = None,
        V: Optional[np.ndarray] = None,
    ) -> tuple[int, Optional[float]]:
        """Return the index of the next selected column."""
        pass

    @staticmethod
    def compute_v(X: np.ndarray, r: int) -> np.ndarray:
        """
        Compute the first r right singular vectors of X.

        Returns
        -------
        V : ndarray
            Matrix whose columns are the first r right singular vectors.
        """
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        return Vt.T[:, :r]

    @staticmethod 
    def _compute_alpha( R: np.ndarray, V: np.ndarray, n_candidates: int, ) -> Optional[float]: 
        """Compute the alpha value for a given iteration""" 
        if V is None: 
            return None 
        row_norms_sq = np.sum(R**2, axis=1) 
        total_norm = row_norms_sq.sum() 
        if np.isclose(total_norm, 0.0): 
            return 0.0 
        p = row_norms_sq / total_norm 
        M = (V.T @ R) @ R.T 
        g = np.linalg.norm(M, axis=0) ** 2 
        g = np.divide( g, row_norms_sq, out=np.zeros_like(g), where=row_norms_sq > 0, ) 
        g[row_norms_sq < 1e-13] = 0.0 
        order = np.argsort(g) 
        g = g[order] 
        p = p[order] 
        F = np.cumsum(p) 
        q = np.concatenate(([F[0] ** n_candidates], np.diff(F**n_candidates))) 
        d_q = np.sum(q * g) 
        d_p = np.sum(p * g) 
        return (d_q / d_p - 1) if d_p > 0 else 0.0

    @staticmethod
    def _as_matrix(X: np.ndarray) -> np.ndarray:
        arr = np.asarray(X, dtype=float)

        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)

        return arr

    @staticmethod
    def _project_out(
        R: np.ndarray,
        center_column: np.ndarray,
    ) -> np.ndarray:
        """Project every column of R orthogonally away from center_column."""

        norm_sq = float(np.dot(center_column, center_column))

        if np.isclose(norm_sq, 0.0):
            return R

        projection = np.outer(
            center_column,
            center_column @ R,
        ) / norm_sq

        return R - projection


class Adaptive(LowRankAlgorithm):
    """Sample a column proportional to its squared residual norm."""

    name = "adaptive_sampling"

    def select_index(
        self,
        R: np.ndarray,
        n_clusters: int,
        random_state: np.random.RandomState,
        n_candidates: Optional[int] = None,
        V: Optional[np.ndarray] = None,
    ) -> tuple[int, Optional[float]]:

        column_norms_sq = np.sum(R**2, axis=0)

        cumulative = np.cumsum(column_norms_sq)

        if cumulative.size == 0 or np.isclose(cumulative[-1], 0.0):
            return 0, None

        rand_val = random_state.uniform() * cumulative[-1]

        idx = int(np.searchsorted(cumulative, rand_val))

        alpha = self._compute_alpha(
            R,
            V,
            n_candidates,
        )

        return min(idx, R.shape[1] - 1), alpha


class BatchMax(LowRankAlgorithm):
    """Sample candidate columns and choose the best one."""

    name = "batch_max"

    def select_index(
        self,
        R: np.ndarray,
        n_clusters: int,
        random_state: np.random.RandomState,
        n_candidates: Optional[int] = None,
        V: Optional[np.ndarray] = None,
    ) -> tuple[int, Optional[float]]:

        n_samples = R.shape[1]

        if n_candidates is None:
            n_candidates = max(
                1,
                2 + int(np.log(n_clusters)),
            )

        column_norms_sq = np.sum(R**2, axis=0)

        cumulative = np.cumsum(column_norms_sq)

        if cumulative.size == 0 or np.isclose(cumulative[-1], 0.0):
            return 0, None

        rand_vals = (
            random_state.uniform(size=n_candidates)
            * cumulative[-1]
        )

        candidate_ids = np.searchsorted(
            cumulative,
            rand_vals,
        )

        candidate_ids = np.asarray(
            candidate_ids,
            dtype=int,
        )

        np.clip(
            candidate_ids,
            0,
            n_samples - 1,
            out=candidate_ids,
        )

        RR = R.T @ R[:, candidate_ids]

        numerators = np.linalg.norm(RR, axis=0) ** 2

        denominators = np.linalg.norm(
            R[:, candidate_ids],
            axis=0,
        ) ** 2

        scores = np.zeros_like(numerators)

        mask = ~np.isclose(denominators, 0.0)

        scores[mask] = (
            numerators[mask]
            / denominators[mask]
        )

        best_idx = int(np.argmax(scores))

        alpha = self._compute_alpha(
            R,
            V,
            n_candidates,
        )

        return int(candidate_ids[best_idx]), alpha


class Greedy(LowRankAlgorithm):
    """Greedily select the column that best reduces residual energy."""

    name = "greedy"

    def select_index(
        self,
        R: np.ndarray,
        n_clusters: int,
        random_state: np.random.RandomState,
        n_candidates: Optional[int] = None,
        V: Optional[np.ndarray] = None,
    ) -> tuple[int, Optional[float]]:

        RR = R.T @ R

        numerators = np.linalg.norm(
            RR,
            axis=0,
        ) ** 2

        denominators = np.linalg.norm(
            R,
            axis=0,
        ) ** 2

        mask = ~np.isclose(denominators, 0.0)

        scores = np.zeros_like(numerators)

        scores[mask] = (
            numerators[mask]
            / denominators[mask]
        )

        return int(np.argmax(scores)), None


class GreedyPP(LowRankAlgorithm):
    """Randomly sample candidate columns and greedily choose the best."""

    name = "greedy_plus_plus"

    def select_index(
        self,
        R: np.ndarray,
        n_clusters: int,
        random_state: np.random.RandomState,
        n_candidates: Optional[int] = None,
        V: Optional[np.ndarray] = None,
    ) -> tuple[int, Optional[float]]:

        n_samples = R.shape[1]

        if n_candidates is None:
            n_candidates = max(
                1,
                2 + int(np.log(n_clusters)),
            )

        candidate_ids = random_state.choice(
            n_samples,
            size=n_candidates,
            replace=False,
        )

        RR = R.T @ R[:, candidate_ids]

        numerators = np.linalg.norm(
            RR,
            axis=0,
        ) ** 2

        denominators = np.linalg.norm(
            R[:, candidate_ids],
            axis=0,
        ) ** 2

        scores = np.zeros_like(numerators)

        mask = ~np.isclose(denominators, 0.0)

        scores[mask] = (
            numerators[mask]
            / denominators[mask]
        )

        best_idx = int(np.argmax(scores))

        return int(candidate_ids[best_idx]), None


class Random(LowRankAlgorithm):
    """Select columns uniformly at random."""

    name = "random_selection"

    def select_index(
        self,
        R: np.ndarray,
        n_clusters: int,
        random_state: np.random.RandomState,
        n_candidates: Optional[int] = None,
        V: Optional[np.ndarray] = None,
    ) -> tuple[int, Optional[float]]:

        column_norms = np.linalg.norm(
            R,
            axis=0,
        )

        tol = 1e-12 * np.linalg.norm(
            R,
            "fro",
        )

        candidates = np.flatnonzero(
            column_norms > tol
        )

        if len(candidates) == 0:
            return 0, None

        return (
            int(random_state.choice(candidates)),
            None,
        )