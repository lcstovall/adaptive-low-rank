import numpy as np
import time

def _greedy_plus_plus_algorithm(
    X, n_clusters, random_state, n_local_trials=None
):
    """Adaptive low-rank matrix approximation algorithm using uniform random 
    candidate sampling.

    At each iteration, this function selects a small set of random candidate
    centers and chooses the one that gives the best reduction in residual
    energy.

    Parameters
    ----------
    X : {ndarray, sparse matrix} of shape (n_samples, n_features)
        Input data matrix.

    n_clusters : int
        Number of centers to choose.

    random_state : RandomState instance
        Random number generator used to sample candidate points.

    n_local_trials : int, default=None
        Number of random candidate centers to evaluate at each iteration. If
        None, the value is set to 2 + log(n_clusters).

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
    n_samples, _ = X.shape

    # Set the number of local seeding trials if None is given
    if n_local_trials is None:
        n_local_trials = 2 + int(np.log(n_clusters))

    # Initialize residuals
    residuals = []

    # Initialize times
    times = []
    start = time.perf_counter()

    # Pick n_clusters points
    for c in range(0, n_clusters):
        
        # Choose center candidates by uniform random sampling
        candidate_ids = random_state.choice(n_samples, size=n_local_trials, replace=False)

        # Greedily select the best candidate
        RR = R @ R[candidate_ids].T
        numerators = np.linalg.norm(RR, axis=0)**2.  
        denominators = np.linalg.norm(R[candidate_ids], axis=1)**2.
        mask = ~np.isclose(denominators, 0.0)
        scores = np.zeros_like(numerators)
        scores[mask] = numerators[mask] / denominators[mask]
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
    return centers, indices, residuals, times