import numpy as np
import time
from adaptive_low_rank.results import AlgorithmResult

def select_rows(
    X, k, random_state=None, n_candidates=None
):
    """Adaptive low-rank matrix approximation algorithm using weighted 
    candidate sampling.

    This algorithm samples candidate points with probability proportional to
    the squared norm of the current residual rows, then chooses the best
    candidate based on projected residual reduction.

    Parameters
    ----------
    X : {ndarray, sparse matrix} of shape (n_samples, n_features)
        Input data matrix.

    k : int
        Number of centers to choose.

    random_state : RandomState instance
        Random number generator used to sample candidate points.

    n_candidates : int, default=None
        Number of random candidates to consider at each iteration. If None,
        the value is set to 2 + log(k).

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
    n_samples, _ = X.shape

    # Set the number of local seeding trials if None is given
    if n_candidates is None:
        n_candidates = 2 + int(np.log(k))

    # Initialize residuals
    residuals = []

    # Initialize times
    times = []
    start = time.perf_counter()

    # Pick k points
    for c in range(0, k):

        # Choose center candidates by sampling with probability proportional to the squared row norms of R
        row_norms_sq = np.sum(R**2, axis=1)
        cum_row_norms = np.cumsum(row_norms_sq)
        rand_vals = random_state.uniform(size=n_candidates) * cum_row_norms[-1]
        candidate_ids = np.searchsorted(cum_row_norms, rand_vals)
        
        # XXX: numerical imprecision can result in a candidate_id out of range
        np.clip(candidate_ids, 0, n_samples - 1, out=candidate_ids)

        # Greedily select the best candidate
        RR = R @ R[candidate_ids].T
        numerators = np.linalg.norm(RR, axis=0)**2.  
        denominators = np.linalg.norm(R[candidate_ids], axis=1)**2.
        scores = numerators / denominators
        best_idx = np.argmax(scores)
        best_candidate = candidate_ids[best_idx]
        
        # Add the chosen row index to indices
        indices[c] = best_candidate
        
        # Update R after adding the new center
        r_star = R[best_candidate]
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