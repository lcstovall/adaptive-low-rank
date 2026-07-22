import numpy as np

def _random_selection_algorithm(
    X, n_clusters, random_state
):
    """Random low-rank matrix approximation which selects rows uniformly 
    from X.

    This function chooses n_clusters distinct points at random from X and
    updates the residual by projecting out the selected centers.

    Parameters
    ----------
    X : {ndarray, sparse matrix} of shape (n_samples, n_features)
        Input data matrix.

    n_clusters : int
        Number of centers to choose.

    random_state : RandomState instance
        Random number generator used to sample the centers.

    Returns
    -------
    centers : ndarray of shape (n_clusters, n_features)
        The selected centers from X.

    indices : ndarray of shape (n_clusters,)
        The indices of the chosen centers in X.

    residuals : list of float
        Frobenius norm of the residual after each center selection.
    """
    # Initialize matrix of Residuals (R), an array of selected indices, and residuals
    R = X.copy()
    indices = np.full(n_clusters, -1, dtype=int)
    residuals = []
    n_samples, n_features = X.shape

    # Randomly select n_clusters distinct indices
    indices = random_state.choice(n_samples, size=n_clusters, replace=False)
    
    # Calculate residuals for each iteration
    for c, idx in enumerate(indices):

        # Update R after adding the new center
        r_star = R[idx]
        projection = np.outer(R @ r_star, r_star) / (np.linalg.norm(r_star) ** 2)
        R = R - projection

        # Update residuals
        residuals.append(np.linalg.norm(R, ord='fro'))

    # Find the lowrank matrix approximation of X
    centers= X[indices]
    return centers, indices, residuals