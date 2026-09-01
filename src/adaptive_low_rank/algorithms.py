from __future__ import annotations

import time
from abc import ABC, abstractmethod

import numpy as np

from adaptive_low_rank.results import AlgorithmResult


class LowRankAlgorithm(ABC):
    """Base class for iterative column-selection algorithms."""

    name = "base"

    def select_columns(
        self,
        X: np.ndarray,
        k: int,
        random_state: np.random.RandomState | int | None = None,
        n_candidates: int | None = None,
        V: np.ndarray | None = None,
        compute_alpha: bool = False,
        compute_runtime: bool = False,
    ) -> AlgorithmResult:
        """
        Select columns from a matrix and record the approximation trajectory.

        Parameters
        ----------
        X : np.ndarray
            Input matrix with shape ``(d, n)``, where ``d`` is the number of
            rows and ``n`` is the number of columns.

        k : int
            Number of columns to select.

        random_state : np.random.RandomState, int, or None
            Random-number generator or seed. If ``None``, a new generator is
            created.

        n_candidates : int or None
            Number of candidate columns used by randomized algorithms.

        V : np.ndarray or None
            Truncated right singular vectors used to compute the alpha
            diagnostic. Ignored when ``compute_alpha=False``.

        compute_alpha : bool, default=False
            Whether to compute the alpha diagnostic. Only ``Adaptive`` and
            ``BatchMax`` use this option.

        compute_runtime : bool, default=False
            Whether to record cumulative elapsed time after each iteration.

        Returns
        -------
        AlgorithmResult
            Selected column indices, residual Frobenius norms, optional
            cumulative runtimes, and optional alpha values.
        """

        R = self._as_matrix(X.copy())

        indices = np.full(k, -1, dtype=int)

        residuals: list[float] = []
        alphas: list[float | None] = []
        times: list[float] = []

        start = time.perf_counter() if compute_runtime else None

        # Normalize the random-state argument to a RandomState instance.
        if random_state is None:
            rng = np.random.RandomState()

        elif isinstance(random_state, (int, np.integer)):
            rng = np.random.RandomState(random_state)

        else:
            rng = random_state

        # Select and project out one column at each iteration.
        for c in range(k):

            idx, alpha = self.select_index(R, k, rng, n_candidates, V, compute_alpha)

            indices[c] = idx
            alphas.append(alpha)

            # Remove the component in the direction of the selected column.
            R = self._project_out(R, R[:, idx])

            if compute_runtime:
                times.append(time.perf_counter() - start)

            # Record the residual Frobenius norm after the projection.
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
        k: int,
        random_state: np.random.RandomState,
        n_candidates: int | None = None,
        V: np.ndarray | None = None,
        compute_alpha: bool = False,
    ) -> tuple[int, float | None]:
        """Return the next column index and an optional alpha diagnostic.

        Parameters
        ----------
        R : np.ndarray
            Current residual matrix.
        k : int
            Target number of selected columns.
        random_state : np.random.RandomState
            Random-number generator used by randomized methods.
        n_candidates : int or None
            Number of candidates considered by randomized methods.
        V : np.ndarray or None
            Truncated right singular vectors for alpha computation.
        compute_alpha : bool, default=False
            Whether to compute the alpha diagnostic when supported.

        Returns
        -------
        index : int
            Index of the selected column.
        alpha : float or None
            Alpha diagnostic, if computed by the implementation.
        """

    @staticmethod
    def compute_v(X: np.ndarray, r: int) -> np.ndarray:
        """
        Compute the first ``r`` left singular vectors of ``X``.

        Parameters
        ----------
        X : np.ndarray
            Input matrix.

        r : int
            Number of left singular vectors.

        Returns
        -------
        V : np.ndarray
            Matrix whose columns are the first ``r`` right singular vectors.
        """

        _, _, Vt = np.linalg.svd(X, full_matrices=False)

        return Vt.T[:, :r]

    @staticmethod
    def _compute_alpha(R: np.ndarray, V: np.ndarray, n_candidates: int) -> float | None:
        """
        Compute the alpha diagnostic for the current iteration.

        Parameters
        ----------
        R : np.ndarray
            Current residual matrix.

        U : np.ndarray
            Truncated left singular vectors.

        n_candidates : int
            Number of candidate columns sampled by the method.

        Returns
        -------
        float or None
            Alpha value.
        """

        if V is None:
            return None

        # Compute squared norms of the residual columns.
        col_norms_sq = np.sum(R**2, axis=0)

        total_norm = col_norms_sq.sum()

        if np.isclose(total_norm, 0.0):
            return 0.0

        p = col_norms_sq / total_norm
        M = (V.T @ R.T) @ R

        g = np.linalg.norm(M, axis=0) ** 2

        g = np.divide(g, col_norms_sq, out=np.zeros_like(g), where=col_norms_sq > 1e-16)

        g[col_norms_sq < 1e-16] = 0.0
        order = np.argsort(g)
        g = g[order]
        p = p[order]
        F = np.cumsum(p)

        if n_candidates < 2501:
            q = np.concatenate(
                ([F[0] ** n_candidates], np.diff(F**n_candidates))
            )  # q_i = F_i^b - F_{i-1}^b. F_n = 1, q_n = 1 - F_{n-1}^b

        else:
            q = np.zeros_like(p)
            q[-1] = 1.0

        d_q = np.sum(q * g)

        d_p = np.sum(p * g)

        if d_p > 0:
            return d_q / d_p - 1

        return 0.0

    @staticmethod
    def _as_matrix(X: np.ndarray) -> np.ndarray:
        """Convert ``X`` to a two-dimensional floating-point array."""

        arr = np.asarray(X, dtype=float)

        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)

        return arr

    @staticmethod
    def _project_out(R: np.ndarray, center_column: np.ndarray) -> np.ndarray:
        """Project each column of ``R`` orthogonally away from a vector."""

        norm_sq = float(np.dot(center_column, center_column))

        if np.isclose(norm_sq, 0.0):
            return R

        projection = np.outer(center_column, center_column @ R) / norm_sq

        return R - projection


class Adaptive(LowRankAlgorithm):
    """Select a column with probability proportional to residual energy."""

    name = "adaptive_sampling"

    def select_index(
        self,
        R: np.ndarray,
        k: int,
        random_state: np.random.RandomState,
        n_candidates: int | None = None,
        V: np.ndarray | None = None,
        compute_alpha: bool = False,
    ) -> tuple[int, float | None]:

        if n_candidates is None:
            n_candidates = 1

        column_norms_sq = np.sum(R**2, axis=0)

        cumulative = np.cumsum(column_norms_sq)

        if cumulative.size == 0 or np.isclose(cumulative[-1], 0.0):
            return 0, None

        # Sample columns in proportion to their squared residual norms.
        rand_val = random_state.uniform() * cumulative[-1]

        idx = int(np.searchsorted(cumulative, rand_val))

        alpha = None

        if compute_alpha and V is not None:
            alpha = self._compute_alpha(R, V, n_candidates)

        return (min(idx, R.shape[1] - 1), alpha)


class BatchMax(LowRankAlgorithm):
    """Sample candidates and select the one with the largest score."""

    name = "batch_max"

    def select_index(
        self,
        R: np.ndarray,
        k: int,
        random_state: np.random.RandomState,
        n_candidates: int | None = None,
        V: np.ndarray | None = None,
        compute_alpha: bool = False,
    ) -> tuple[int, float | None]:

        n_samples = R.shape[1]

        if n_candidates is None:
            n_candidates = max(1, 2 + int(np.log(k)))

        # Sample candidate columns in proportion to residual energy.

        column_norms_sq = np.sum(R**2, axis=0)

        cumulative = np.cumsum(column_norms_sq)

        if cumulative.size == 0 or np.isclose(cumulative[-1], 0.0):
            return 0, None

        rand_vals = random_state.uniform(size=n_candidates) * cumulative[-1]

        candidate_ids = np.searchsorted(cumulative, rand_vals)

        candidate_ids = np.asarray(candidate_ids, dtype=int)

        np.clip(candidate_ids, 0, n_samples - 1, out=candidate_ids)

        RR = R.T @ R[:, candidate_ids]

        numerators = np.linalg.norm(RR, axis=0) ** 2

        denominators = np.linalg.norm(R[:, candidate_ids], axis=0) ** 2

        scores = np.zeros_like(numerators)

        mask = ~np.isclose(denominators, 0.0)

        scores[mask] = numerators[mask] / denominators[mask]

        best_idx = int(np.argmax(scores))

        alpha = None

        if compute_alpha and V is not None:
            alpha = self._compute_alpha(R, V, n_candidates)

        return (int(candidate_ids[best_idx]), alpha)


class Greedy(LowRankAlgorithm):
    """Select the column that maximizes the residual-energy reduction."""

    name = "greedy"

    def select_index(
        self,
        R: np.ndarray,
        k: int,
        random_state: np.random.RandomState,
        n_candidates: int | None = None,
        V: np.ndarray | None = None,
        compute_alpha: bool = False,
    ) -> tuple[int, float | None]:

        RR = R @ R.T

        numerators = np.sum(R * (RR @ R), axis=0)

        denominators = np.sum(R**2, axis=0)

        scores = np.zeros_like(numerators)

        mask = ~np.isclose(denominators, 0.0)

        scores[mask] = numerators[mask] / denominators[mask]

        return (int(np.argmax(scores)), None)


class GreedyPP(LowRankAlgorithm):
    """Sample candidate columns uniformly and select the best candidate."""

    name = "greedy_plus_plus"

    def select_index(
        self,
        R: np.ndarray,
        k: int,
        random_state: np.random.RandomState,
        n_candidates: int | None = None,
        V: np.ndarray | None = None,
        compute_alpha: bool = False,
    ) -> tuple[int, float | None]:

        n_samples = R.shape[1]

        if n_candidates is None:
            n_candidates = max(1, 2 + int(np.log(k)))

        candidate_ids = random_state.choice(n_samples, size=n_candidates, replace=False)

        RR = R.T @ R[:, candidate_ids]

        numerators = np.linalg.norm(RR, axis=0) ** 2

        denominators = np.linalg.norm(R[:, candidate_ids], axis=0) ** 2

        scores = np.zeros_like(numerators)

        mask = ~np.isclose(denominators, 0.0)

        scores[mask] = numerators[mask] / denominators[mask]

        best_idx = int(np.argmax(scores))

        return (int(candidate_ids[best_idx]), None)


class Random(LowRankAlgorithm):
    """Select a nonzero-residual column uniformly at random."""

    name = "random_selection"

    def select_index(
        self,
        R: np.ndarray,
        k: int,
        random_state: np.random.RandomState,
        n_candidates: int | None = None,
        V: np.ndarray | None = None,
        compute_alpha: bool = False,
    ) -> tuple[int, float | None]:

        column_norms = np.linalg.norm(R, axis=0)

        tol = 1e-12 * np.linalg.norm(R, "fro")

        candidates = np.flatnonzero(column_norms > tol)

        if len(candidates) == 0:
            return 0, None

        return (int(random_state.choice(candidates)), None)
