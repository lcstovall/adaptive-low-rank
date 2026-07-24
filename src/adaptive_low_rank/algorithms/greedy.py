import numpy as np
import time
from adaptive_low_rank.results import AlgorithmResult

def select_rows(
    X, k
):
    """Greedy low-rank matrix approximation algorithm.
 
    This function selects rows iteratively to minimize the Frobenius norm of
    the residual matrix after projecting onto chosen centers.

    Parameters
    ----------
    X : {ndarray, sparse matrix} of shape (n_samples, n_features)
        Input data matrix.

    k : int
        Number of centers to choose.

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
        
        # Greedily choose the best row
        RR = R @ R.T
        numerators = np.linalg.norm(RR, axis=0)**2.  
        denominators = np.linalg.norm(R, axis=1)**2.
        mask = ~np.isclose(denominators, 0.0)
        scores = np.zeros_like(numerators)
        scores[mask] = numerators[mask] / denominators[mask]
        best_idx = np.argmax(scores)
       
        # Add the chosen row index to indices
        indices[c] = best_idx
        
        # Update R
        r_star = R[best_idx]
        norm_r_star_sq = np.dot(r_star, r_star)
        projection = np.outer(R @ r_star, r_star) / norm_r_star_sq
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