import numpy as np
import time

def _greedy_algorithm(
    X, n_clusters
):
    """Greedy low-rank matrix approximation algorithm.
 
    This function selects rows iteratively to minimize the Frobenius norm of
    the residual matrix after projecting onto chosen centers.

    Parameters
    ----------
    X : {ndarray, sparse matrix} of shape (n_samples, n_features)
        Input data matrix.

    n_clusters : int
        Number of centers to choose.

    Returns
    -------
    centers : ndarray of shape (n_clusters, n_features)
        The selected centers from X.

    indices : ndarray of shape (n_clusters,)
        The indices of the chosen centers in X.

    residuals : list of float
        Frobenius norm of the residual after each center selection.

    times : list of float
        Elapsed time at each selection step.
    """
    # Initialize matrix of Residuals (R) and array of selected indices
    R = X.copy()
    indices = np.full(n_clusters, -1, dtype=int)

    # Initialize residuals
    residuals = []

    # Initialize times
    times = []
    start = time.perf_counter()

    # Pick n_clusters points
    for c in range(0, n_clusters):
        
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
    return centers, indices, residuals, times