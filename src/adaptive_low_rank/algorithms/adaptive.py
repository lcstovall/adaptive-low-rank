import numpy as np
import time
from adaptive_low_rank.results import AlgorithmResult

def select_rows(
    X, k, random_state=None
):
    """Adaptive low-rank matrix approximation by sampling with residual-based weights.

    At each iteration, this function selects a new center by sampling one point
    with probability proportional to the squared row norm of the current
    residual matrix.

    Parameters
    ----------
    X : {ndarray, sparse matrix} of shape (n_samples, n_features)
        Input data matrix.

    k : int
        Number of centers to choose.

    random_state : RandomState instance
        Random number generator used to select points.

    Returns
    -------
    centers : ndarray of shape (k, n_features)
        The selected centers from X.

    indices : ndarray of shape (k,)
        The indices of the chosen centers in X.

    residuals : list of float
        Frobenius norm of the residual after each center selection.

    times : list of float
        Elapsed time at each selection step.
    """
    random_state = np.random.default_rng(random_state)

    # Initialize matrix of Residuals (R) and array of selected indices
    R = X.copy()
    indices = np.full(k, -1, dtype=int)

    # Initialize residuals
    residuals = []

    # Initialize times
    times = []
    start = time.perf_counter()

    # Pick k points
    for c in range(0, k):
        
        # Choose row by sampling with probability proportional to the squared row norms of R
        row_norms_sq = np.sum(R**2, axis=1)
        cum_row_norms = np.cumsum(row_norms_sq)
        rand_vals = random_state.uniform(size=1) * cum_row_norms[-1]
        center_id = np.searchsorted(cum_row_norms, rand_vals)[0]

        # Add the chosen row index to indices
        indices[c] = center_id
        
        # Update R after adding the new center
        r_star = R[center_id]
        projection = np.outer(R @ r_star, r_star) / (np.linalg.norm(r_star) ** 2)
        R = R - projection
        
        # Record running time
        end = time.perf_counter()
        times.append(end - start)
        
        # Update residuals
        residuals.append(np.linalg.norm(R, ord='fro'))

    # Find the lowrank matrix approximation of X
    centers = X[indices]
    return AlgorithmResult(
    rows=centers,
    indices=indices,
    residuals=np.asarray(residuals),
    runtimes=np.asarray(times),
    residual_matrix=R,
)