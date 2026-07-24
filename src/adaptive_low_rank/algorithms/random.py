import numpy as np
import time
from adaptive_low_rank.results import AlgorithmResult

def select_rows(
    X, k, random_state=None
):
    """Random low-rank matrix approximation which selects rows uniformly 
    from X.

    This function chooses k distinct points at random from X and
    updates the residual by projecting out the selected centers.

    Parameters
    ----------
    X : {ndarray, sparse matrix} of shape (n_samples, n_features)
        Input data matrix.

    k : int
        Number of centers to choose.

    random_state : RandomState instance
        Random number generator used to sample the centers.

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

    # Initialize matrix of residuals (R), an array of selected indices, and residuals
    R = X.copy()
    indices = np.full(k, -1, dtype=int)
    residuals = []
    times = []
    n_samples, n_features = X.shape

    # Randomly select k distinct indices
    indices = random_state.choice(n_samples, size=k, replace=False)
    start = time.perf_counter()

    # Calculate residuals for each iteration
    for c in range(0, k):
        idx = indices[c]

        # Update R after adding the new center
        r_star = R[idx]
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