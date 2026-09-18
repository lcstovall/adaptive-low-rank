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
            cumulative runtimes, and optional gain diagnostics.
        """

        R = self._as_matrix(X.copy())

        indices = np.full(k, -1, dtype=int)

        residuals: list[float] = []
        gains_bm: list[float | None] = []
        gains_as: list[float | None] = []
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

            idx, gains = self.select_index(R, k, rng, n_candidates, V, compute_alpha)

            indices[c] = idx

            if gains is None:
                gains_bm.append(None)
                gains_as.append(None)
            else:
                gains_bm.append(gains[0])
                gains_as.append(gains[1])

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
            gains_bm=np.asarray(gains_bm, dtype=float),
            gains_as=np.asarray(gains_as, dtype=float),
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
    ) -> tuple[int, tuple[float, float] | None]:
        """Return the next column index and optional gain diagnostics.

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
        gains : tuple[float, float] or None
            Batch-max and adaptive-sampling gains, if computed.
        """

    @staticmethod
    def _compute_alpha(
        R: np.ndarray, n_candidates: int
    ) -> tuple[float, float] | None:
        """
        Compute the alpha diagnostic for the current iteration.

        Parameters
        ----------
        R : np.ndarray
            Current residual matrix.

        n_candidates : int
            Number of candidate columns sampled by the method.

        Returns
        -------
        tuple[float, float] or None
            Batch-max and adaptive-sampling gains, respectively.
        """
        C = R @ R.T
        e_p = np.sum(np.diag(C@C)) / np.sum(np.diag(C))
        col_norms_sq = np.linalg.norm(R, axis=0) ** 2

        if np.isclose(col_norms_sq.sum(), 0.0):
            return None, None

        p = col_norms_sq / np.sum(col_norms_sq)
        g = (R * (C @ R)).sum(axis=0)

        g = np.divide(g, col_norms_sq, out=np.zeros_like(g), where=col_norms_sq > 1e-16)
        g[col_norms_sq < 1e-16] = 0.0

        ordering = np.argsort(g)
        g = g[ordering]
        p = p[ordering]
        Fb = np.cumsum(np.concatenate(([0], p)))**n_candidates

        q = Fb[1:] - Fb[:-1]
        e_q = np.sum(q * g)

        return e_q, e_p

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
    ) -> tuple[int, tuple[float, float] | None]:

        if n_candidates is None:
            n_candidates = 1

        column_norms_sq = np.sum(R**2, axis=0)

        cumulative = np.cumsum(column_norms_sq)

        if cumulative.size == 0 or np.isclose(cumulative[-1], 0.0):
            return 0, None

        # Sample columns in proportion to their squared residual norms.
        rand_val = random_state.uniform() * cumulative[-1]

        idx = int(np.searchsorted(cumulative, rand_val))

        return (min(idx, R.shape[1] - 1), None)


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
    ) -> tuple[int, tuple[float, float] | None]:

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

        gains = None

        if compute_alpha:
            gains = self._compute_alpha(R, n_candidates)

        return (int(candidate_ids[best_idx]), gains)


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


class Sequential(LowRankAlgorithm):
    """Select columns in their original order, starting with column zero."""

    name = "sequential_selection"

    def select_columns(self, *args, **kwargs) -> AlgorithmResult:
        self._next_index = 0

        return super().select_columns(*args, **kwargs)

    def select_index(
        self,
        R: np.ndarray,
        k: int,
        random_state: np.random.RandomState,
        n_candidates: int | None = None,
        V: np.ndarray | None = None,
        compute_alpha: bool = False,
    ) -> tuple[int, float | None]:
        index = self._next_index
        self._next_index += 1

        return index, None
