"""Synthetic dataset generation, including config-driven on-demand builds."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
MULTISCALE_MODELS = ("exact", "orthogonal_tubes", "ambient_noise", "tight_frame_union")

Array = NDArray[np.float64]
ModelName = Literal["exact", "orthogonal_tubes", "ambient_noise", "tight_frame_union"]


def generate_synthetic_dataset(decay_type, decay_param, n=2000, d=2000, random_state=0):
    """
    Generate a synthetic matrix with prescribed singular-value decay.

    Parameters
    ----------
    decay_type : {"poly", "exp"}
        Type of singular-value decay.
    decay_param : float
        Decay parameter:
            poly: sigma_i = i^(-decay_param)
            exp:  sigma_i = exp(-decay_param * (i - 1))
    n : int
        Number of columns.
    d : int
        Number of rows.

    Returns
    -------
    X : ndarray
        Synthetic d x n matrix.
    """

    r = min(d, n)

    rng = np.random.default_rng(random_state)

    # Random orthonormal left singular vectors
    U_random = rng.standard_normal((d, r))
    U, _ = np.linalg.qr(U_random)

    # Random orthonormal right singular vectors
    V_random = rng.standard_normal((n, r))
    V, _ = np.linalg.qr(V_random)

    # Construct singular values
    i = np.arange(1, r + 1)

    if decay_type == "poly":
        singular_values = i ** (-decay_param)

    elif decay_type == "exp":
        singular_values = np.exp(-decay_param * (i - 1))

    else:
        raise ValueError(f"Unknown decay type: {decay_type}")

    # Normalize so ||X||_F = 1
    singular_values /= np.linalg.norm(singular_values)

    # X = U Sigma V^T
    X = (U * singular_values) @ V.T

    return X


@dataclass
class MultiscaleDataset:
    """Container returned by :func:`generate_multiscale_dataset`."""

    X: Array
    labels: NDArray[np.int64]
    central_energies: Array
    central_directions: Array
    amplitudes: Array | None
    model: str
    metadata: dict[str, Any]
    tube_bases: list[Array] | Array | None = None

    def save(self, path: str | Path, generation_config: str | None = None) -> Path:
        """Save the dataset and its construction metadata to a compressed NPZ."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "X": self.X,
            "labels": self.labels,
            "central_energies": self.central_energies,
            "central_directions": self.central_directions,
            "model": np.asarray(self.model),
            "metadata_json": np.asarray(json.dumps(self.metadata, sort_keys=True)),
        }
        if generation_config is not None:
            payload["generation_config_json"] = np.asarray(generation_config)
        if self.amplitudes is not None:
            payload["amplitudes"] = self.amplitudes
        if isinstance(self.tube_bases, list):
            if len({block.shape[1] for block in self.tube_bases}) == 1:
                payload["tube_bases"] = np.stack(self.tube_bases)
            else:
                payload["tube_bases"] = np.concatenate(self.tube_bases, axis=1)
                payload["tube_basis_offsets"] = np.cumsum(
                    [0] + [block.shape[1] for block in self.tube_bases]
                )
        elif self.tube_bases is not None:
            payload["tube_bases"] = self.tube_bases
        np.savez_compressed(destination, **payload)
        return destination


def log_spaced_energies(
    n_clusters: int,
    high_energy: float = 1.0,
    condition_number: float = 1.0e4,
    decay_rate: Literal["linear", "polynomial", "exponential"] = "polynomial",
) -> Array:
    """Return energies with specified decay pattern from H to H/kappa.
    
    Parameters
    ----------
    n_clusters : int
        Number of energy levels to generate.
    high_energy : float, default 1.0
        The highest energy level (λ_1).
    condition_number : float, default 1e4
        Ratio between highest and lowest: κ = λ_1 / λ_L.
    decay_rate : {'linear', 'polynomial', 'exponential'}, default 'polynomial'
        How energy decreases across clusters:
        - 'linear': arithmetic sequence (constant subtraction)
        - 'polynomial': polynomial decay 1/t² (default)
        - 'exponential': exponential decay
    
    Returns
    -------
    Array
        Energy levels in decreasing order.
    """
    if n_clusters < 2:
        raise ValueError("n_clusters must be at least 2")
    if high_energy <= 0:
        raise ValueError("high_energy must be positive")
    if condition_number < 1:
        raise ValueError("condition_number must be at least 1")
    if decay_rate not in {"linear", "polynomial", "exponential"}:
        raise ValueError(f"decay_rate must be one of {{'linear', 'polynomial', 'exponential'}}, got {decay_rate}")
    
    low_energy = high_energy / condition_number
    
    if decay_rate == "linear":
        # Arithmetic sequence
        return np.linspace(high_energy, low_energy, num=n_clusters, dtype=np.float64)
    elif decay_rate == "polynomial":
        # Polynomial decay: 1/t^2
        t = np.linspace(1, np.sqrt(condition_number), num=n_clusters, dtype=np.float64)
        return high_energy / (t ** 2)
    else:  # exponential
        # Exponential decay
        return np.geomspace(high_energy, low_energy, num=n_clusters, dtype=np.float64)


def _orthonormal_columns(
    ambient_dim: int,
    count: int,
    rng: np.random.Generator,
    basis: Literal["random", "coordinate"],
) -> Array:
    if ambient_dim < count:
        raise ValueError(
            f"ambient_dim={ambient_dim} must be at least the requested "
            f"orthogonal dimension {count}"
        )
    if basis == "coordinate":
        return np.eye(ambient_dim, count, dtype=np.float64)
    if basis != "random":
        raise ValueError("basis must be 'random' or 'coordinate'")
    Q, R = np.linalg.qr(rng.standard_normal((ambient_dim, count)), mode="reduced")
    # Fix QR sign ambiguity to make seeded output reproducible across calls.
    signs = np.where(np.diag(R) < 0.0, -1.0, 1.0)
    return Q * signs


def _normalize_per_cluster_param(
    param: int | Array,
    n_clusters: int,
    param_name: str,
    min_value: int = 1,
) -> NDArray[np.int64]:
    """Convert per-cluster parameter from int or array to normalized array.
    
    Parameters
    ----------
    param : int or Array
        Either a single int (broadcast to all clusters) or array of length n_clusters.
    n_clusters : int
        Number of clusters.
    param_name : str
        Name for error messages (e.g., 'tube_dim').
    min_value : int, default 1
        Minimum allowed value.
    
    Returns
    -------
    Array
        Array of shape (n_clusters,) with values per cluster.
    """
    if isinstance(param, (int, np.integer)):
        if param < min_value:
            raise ValueError(f"{param_name} must be at least {min_value}, got {param}")
        return np.full(n_clusters, param, dtype=np.int64)
    
    param = np.asarray(param, dtype=np.int64)
    if param.ndim != 1:
        raise ValueError(f"{param_name} must be 1-D or scalar")
    if len(param) != n_clusters:
        raise ValueError(
            f"{param_name} has length {len(param)} but n_clusters={n_clusters}"
        )
    if np.any(param < min_value):
        raise ValueError(f"All {param_name} values must be at least {min_value}")
    return param


def _cluster_amplitudes(
    energies: Array, columns_per_cluster: int, rng: np.random.Generator, spread: float
) -> Array:
    """Create differently scaled columns with the prescribed central energies."""
    if columns_per_cluster < 1:
        raise ValueError("columns_per_cluster must be positive")
    if spread < 0:
        raise ValueError("amplitude_spread must be nonnegative")

    n_clusters = energies.size
    if spread == 0:
        raw = np.ones((n_clusters, columns_per_cluster), dtype=np.float64)
    else:
        raw = np.exp(spread * rng.standard_normal((n_clusters, columns_per_cluster)))
    # Random signs do not change cluster energy but avoid a positivity artifact.
    raw *= rng.choice(np.array([-1.0, 1.0]), size=raw.shape)
    raw /= np.linalg.norm(raw, axis=1, keepdims=True)
    return raw * np.sqrt(energies)[:, None]


def _transverse_coordinates(
    transverse_dim: int, amplitudes: Array, rng: np.random.Generator, balanced: bool
) -> Array:
    """Return z[:, s] with norm at most one for one cluster.

    If ``balanced`` is true, the construction also satisfies

        sum_s amplitude[s]**2 * z[:, s] == 0,

    up to floating-point error.  This makes the declared central direction an
    exact eigenvector of the cluster covariance.
    """
    n_columns = amplitudes.size
    if transverse_dim == 0:
        return np.zeros((0, n_columns), dtype=np.float64)

    z = rng.standard_normal((transverse_dim, n_columns))
    norms = np.linalg.norm(z, axis=0, keepdims=True)
    z /= np.maximum(norms, np.finfo(np.float64).tiny)
    z *= rng.random((1, n_columns)) ** (1.0 / max(transverse_dim, 1))

    if balanced:
        if n_columns < 2:
            raise ValueError(
                "balanced_transverse=True requires at least two columns per cluster"
            )
        weights = amplitudes**2
        z -= (z @ weights / weights.sum())[:, None]

    max_norm = float(np.max(np.linalg.norm(z, axis=0)))
    if max_norm > 1.0:
        z /= max_norm
    return z


def _validate_tight_frame_dimensions(
    columns_per_cluster: NDArray[np.int64],
    tube_dim: NDArray[np.int64],
    cached_path: Path | None = None,
) -> None:
    invalid = np.flatnonzero(columns_per_cluster < tube_dim)
    if invalid.size:
        i = int(invalid[0])
        message = (
            f"Cluster {i}: columns_per_cluster={columns_per_cluster[i]} < "
            f"tube_dim={tube_dim[i]}; a tight frame must span the full subspace"
        )
        if cached_path is not None:
            message += f"; regenerate cached dataset {cached_path}"
        raise ValueError(message)


def _validate_cached_tight_frame_dimensions(path: Path) -> None:
    with np.load(path, allow_pickle=False) as archive:
        if "labels" not in archive or "tube_bases" not in archive:
            raise ValueError(f"{path} lacks tight-frame basis metadata; regenerate the dataset")
        labels = archive["labels"]
        bases = archive["tube_bases"]
        if bases.ndim == 3:
            dimensions = np.full(bases.shape[0], bases.shape[2], dtype=np.int64)
        elif bases.ndim == 2 and "tube_basis_offsets" in archive:
            offsets = archive["tube_basis_offsets"]
            if (
                offsets.ndim != 1
                or offsets.size < 2
                or offsets[0] != 0
                or offsets[-1] != bases.shape[1]
                or np.any(np.diff(offsets) <= 0)
            ):
                raise ValueError(f"{path} has invalid tube basis offsets; regenerate the dataset")
            dimensions = np.diff(offsets)
        else:
            raise ValueError(f"{path} has invalid tube bases; regenerate the dataset")
        if (
            labels.ndim != 1
            or not np.issubdtype(labels.dtype, np.integer)
            or np.any(labels < 0)
            or np.any(labels >= len(dimensions))
        ):
            raise ValueError(f"{path} has invalid cluster labels; regenerate the dataset")
        counts = np.bincount(labels, minlength=len(dimensions))
        _validate_tight_frame_dimensions(counts, dimensions, cached_path=path)


def _tight_frame_in_subspace(
    subspace_basis: Array,
    n_vectors: int,
    frame_level: float,
    rng: np.random.Generator,
) -> Array:
    """Generate a tight frame within a given subspace.

    Create equal-norm tight vectors spanning the supplied subspace.

    Parameters
    ----------
    subspace_basis : ndarray, shape (ambient_dim, subspace_dim)
        Orthonormal basis for the subspace.
    n_vectors : int
        Number of frame vectors to generate.
    frame_level : float
        Level λ such that X @ X.T = λ * P for the supplied subspace.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    X : ndarray, shape (ambient_dim, n_vectors)
        Tight frame vectors satisfying X @ X.T = frame_level * P, where P
        projects onto the supplied subspace.
    """
    subspace_dim = subspace_basis.shape[1]
    # Real Fourier modes have orthonormal rows and constant column norm.
    angles = 2 * np.pi * np.arange(n_vectors) / n_vectors
    rows = []
    if subspace_dim % 2:
        rows.append(np.ones(n_vectors) / np.sqrt(n_vectors))
    elif subspace_dim == n_vectors:
        rows.extend(
            [
                np.ones(n_vectors) / np.sqrt(n_vectors),
                (-1.0) ** np.arange(n_vectors) / np.sqrt(n_vectors),
            ]
        )
    for frequency in range(1, (subspace_dim - len(rows)) // 2 + 1):
        rows.append(np.sqrt(2 / n_vectors) * np.cos(frequency * angles))
        rows.append(np.sqrt(2 / n_vectors) * np.sin(frequency * angles))
    rotation, _ = np.linalg.qr(rng.standard_normal((subspace_dim, subspace_dim)))
    coefficients = rotation @ np.asarray(rows)[:, rng.permutation(n_vectors)]
    return np.sqrt(frame_level) * (subspace_basis @ coefficients)


def generate_multiscale_dataset(
    *,
    model: ModelName = "exact",
    n_clusters: int = 64,
    columns_per_cluster: int | Array = 8,
    high_energy: float = 1.0,
    condition_number: float = 1.0e4,
    ambient_dim: int | None = None,
    tube_dim: int | Array = 3,
    eta: float = 0.03,
    amplitude_spread: float = 0.8,
    balanced_transverse: bool = True,
    basis: Literal["random", "coordinate"] = "random",
    level_weights: Array | None = None,
    decay_rate: Literal["linear", "polynomial", "exponential"] = "polynomial",
    seed: int = 0,
) -> MultiscaleDataset:
    """Generate one exact, orthogonal-tube, tight-frame-union, or ambient-noise dataset.

    Parameters
    ----------
    model:
        ``exact``, ``orthogonal_tubes``, ``tight_frame_union``, or ``ambient_noise``.
    n_clusters:
        Number of distinct central directions and energy levels.
    columns_per_cluster:
        Number of vectors per cluster/subspace. Can be an int (same for all clusters)
        or an array of length n_clusters (different per cluster). For ``tight_frame_union``:
        frame vectors per subspace. For other models: scaled columns per cluster.
    high_energy:
        Highest energy level. Ignored if ``level_weights`` is provided.
    condition_number:
        Ratio between the largest and smallest prescribed central energies/frame levels.
        Ignored if ``level_weights`` is provided.
    ambient_dim:
        Row dimension of ``X``. The minimum is ``n_clusters`` for ``exact``
        and ``ambient_noise``, and ``sum(tube_dim)`` for ``orthogonal_tubes``
        and ``tight_frame_union``.
    tube_dim:
        Dimension of each mutually orthogonal tube subspace. Can be an int (same for all)
        or an array of length n_clusters (different per cluster). For ``tight_frame_union``:
        dimension of the subspace basis; must not exceed the number of vectors.
    eta:
        Maximum transverse-to-central norm ratio for each column (unused for
        ``tight_frame_union``).
    amplitude_spread:
        Standard deviation of lognormal raw column magnitudes (unused for
        ``tight_frame_union``).
    balanced_transverse:
        In the orthogonal-tube model, center transverse deviations so that the
        declared central direction is an exact covariance eigenvector.
    basis:
        Random or coordinate basis for orthogonal subspaces.
    level_weights:
        Custom energy/frame levels per cluster. If provided, must be an array of length
        n_clusters with positive values. Overrides automatic generation via
        ``high_energy``, ``condition_number``, and ``decay_rate``. Order must match
        ``tube_dim`` and ``columns_per_cluster``.
    decay_rate:
        How energy levels decrease: ``linear`` (arithmetic), ``polynomial`` (1/t²,
        default), or ``exponential`` (geometric). Ignored if ``level_weights`` is provided.
    seed:
        Random seed for reproducibility.
    """
    if model not in {"exact", "orthogonal_tubes", "ambient_noise", "tight_frame_union"}:
        raise ValueError(f"unknown model: {model}")
    if eta < 0:
        raise ValueError("eta must be nonnegative")

    # Normalize per-cluster parameters to arrays
    columns_per_cluster_array = _normalize_per_cluster_param(
        columns_per_cluster, n_clusters, "columns_per_cluster"
    )
    tube_dim_array = _normalize_per_cluster_param(
        tube_dim, n_clusters, "tube_dim"
    )
    
    if model == "tight_frame_union":
        _validate_tight_frame_dimensions(columns_per_cluster_array, tube_dim_array)
    elif np.any(tube_dim_array < columns_per_cluster_array):
        for i in range(n_clusters):
            if tube_dim_array[i] < columns_per_cluster_array[i]:
                raise ValueError(
                    f"Cluster {i}: tube_dim={tube_dim_array[i]} < "
                    f"columns_per_cluster={columns_per_cluster_array[i]} (must be ≥)"
                )

    rng = np.random.default_rng(seed)
    
    # Use custom level_weights if provided, otherwise generate automatically
    if level_weights is not None:
        energies = np.asarray(level_weights, dtype=np.float64)
        if energies.ndim != 1:
            raise ValueError("level_weights must be 1-D")
        if len(energies) != n_clusters:
            raise ValueError(
                f"level_weights has length {len(energies)} but n_clusters={n_clusters}"
            )
        if np.any(energies <= 0):
            raise ValueError("All level_weights values must be positive")
    else:
        energies = log_spaced_energies(
            n_clusters, high_energy, condition_number, decay_rate=decay_rate
        )
    
    # For tight_frame_union, energies are frame levels; skip amplitude scaling
    if model != "tight_frame_union":
        amplitudes = _cluster_amplitudes(
            energies, columns_per_cluster_array[0], rng, amplitude_spread
        )
    else:
        # For tight_frame_union, we don't use amplitudes in the traditional sense
        amplitudes = None

    required_dim = (
        int(np.sum(tube_dim_array))
        if model in {"orthogonal_tubes", "tight_frame_union"}
        else n_clusters
    )
    if ambient_dim is None:
        ambient_dim = required_dim
    if ambient_dim < required_dim:
        raise ValueError(
            f"ambient_dim={ambient_dim} is too small for model={model}; "
            f"it must be at least {required_dim}"
        )

    total_basis_dim = (
        int(np.sum(tube_dim_array))
        if model in {"orthogonal_tubes", "tight_frame_union"}
        else n_clusters
    )
    Q = _orthonormal_columns(ambient_dim, total_basis_dim, rng, basis)

    if model == "tight_frame_union":
        # Tight frame union of orthogonal subspaces with per-cluster dimensions
        # tube_bases size is tricky now: each cluster has different tube_dim
        # Store as a list since they can be different sizes
        tube_bases_list = []
        central_directions = np.empty((ambient_dim, n_clusters), dtype=np.float64)
        
        n_columns = int(np.sum(columns_per_cluster_array))
        X = np.empty((ambient_dim, n_columns), dtype=np.float64)
        labels_list = []
        
        col_start = 0  # Track column position in X
        basis_start = 0  # Track basis column position in Q
        
        for i in range(n_clusters):
            m_i = tube_dim_array[i]  # Dimension of subspace i
            n_i = columns_per_cluster_array[i]  # Number of points in subspace i
            
            # Extract basis for subspace i
            block = Q[:, basis_start : basis_start + m_i]
            tube_bases_list.append(block)
            central_directions[:, i] = block[:, 0]
            
            col_stop = col_start + n_i
            
            # Generate tight frame with frame level = energies[i]
            X[:, col_start:col_stop] = _tight_frame_in_subspace(
                block, n_i, energies[i], rng
            )
            
            # Track labels for this cluster
            labels_list.append(np.full(n_i, i, dtype=np.int64))
            
            col_start = col_stop
            basis_start += m_i
        
        # Convert labels to single array
        labels = np.concatenate(labels_list)
        
        # Store tube_bases as a dict or list since sizes vary
        tube_bases = tube_bases_list
        
    elif model == "orthogonal_tubes":
        tube_bases_list = []
        central_directions = np.empty((ambient_dim, n_clusters), dtype=np.float64)
        
        basis_start = 0
        for i in range(n_clusters):
            m_i = tube_dim_array[i]
            block = Q[:, basis_start : basis_start + m_i]
            tube_bases_list.append(block)
            central_directions[:, i] = block[:, 0]
            basis_start += m_i
        
        tube_bases = tube_bases_list
    else:
        tube_bases = None
        central_directions = Q[:, :n_clusters]

    if model != "tight_frame_union":
        n_columns = int(np.sum(columns_per_cluster_array))
        X = np.empty((ambient_dim, n_columns), dtype=np.float64)
        
        # Create labels for each cluster based on its per-cluster column count
        labels_list = []
        for i in range(n_clusters):
            labels_list.append(np.full(columns_per_cluster_array[i], i, dtype=np.int64))
        labels = np.concatenate(labels_list)

        col_start = 0
        for i in range(n_clusters):
            n_i = columns_per_cluster_array[i]
            col_stop = col_start + n_i
            a_i = amplitudes[i]
            u_i = central_directions[:, i]

            if model == "exact" or eta == 0:
                X[:, col_start:col_stop] = u_i[:, None] * a_i[None, :]
                col_start = col_stop
                continue

            if model == "orthogonal_tubes":
                # tube_bases is now a list, get the basis for cluster i
                block_i = tube_bases[i] if isinstance(tube_bases, list) else tube_bases[i]
                m_i = tube_dim_array[i]
                V_i = block_i[:, 1:] if m_i > 1 else np.zeros((ambient_dim, 0))
                z_i = _transverse_coordinates(m_i - 1, a_i, rng, balanced_transverse)
                transverse = V_i @ z_i if m_i > 1 else np.zeros((ambient_dim, n_i))
            else:
                # Arbitrary ambient perturbations: only remove the component along
                # this cluster's own center.  The result may point toward another
                # cluster and deliberately tests cross-cluster leakage.
                transverse = rng.standard_normal((ambient_dim, n_i))
                transverse -= u_i[:, None] * (u_i @ transverse)[None, :]
                norms = np.linalg.norm(transverse, axis=0, keepdims=True)
                transverse /= np.maximum(norms, np.finfo(np.float64).tiny)
                transverse *= rng.random((1, n_i))

            X[:, col_start:col_stop] = u_i[:, None] * a_i[None, :] + eta * transverse * a_i[None, :]
            col_start = col_stop

    metadata: dict[str, Any] = {
        "model": model,
        "seed": seed,
        "ambient_dim": ambient_dim,
        "n_columns": n_columns,
        "n_clusters": n_clusters,
        "columns_per_cluster": int(columns_per_cluster_array[0]) if np.all(columns_per_cluster_array == columns_per_cluster_array[0]) else "variable",
        "tube_dim": int(tube_dim_array[0]) if np.all(tube_dim_array == tube_dim_array[0]) else "variable",
        "high_energy": high_energy if level_weights is None else None,
        "condition_number": condition_number if level_weights is None else None,
        "decay_rate": decay_rate if level_weights is None else None,
        "level_weights": "custom" if level_weights is not None else None,
        "eta": eta if model not in {"exact", "tight_frame_union"} else 0.0,
        "amplitude_spread": amplitude_spread,
        "balanced_transverse": (
            balanced_transverse if model == "orthogonal_tubes" else None
        ),
        "basis": basis,
        "orientation": "columns_are_data_points",
    }
    return MultiscaleDataset(
        X=X,
        labels=labels,
        central_energies=energies,
        central_directions=central_directions,
        amplitudes=amplitudes,
        model=model,
        metadata=metadata,
        tube_bases=tube_bases,
    )


def _resolve_config(config):
    """Classify a config as a ``decay`` or ``multiscale`` dataset, or neither.

    A config describes a synthetic dataset only if it sets ``synthetic: true``.
    ``dataset`` names the output file. ``dataset_type`` defaults to
    ``multiscale`` when ``model`` is given and ``decay`` otherwise.

    Returns ``(kind, name, config)`` where ``kind`` is ``None`` when the config
    does not describe a synthetic dataset.
    """
    config = dict(config or {})
    name = str(config.get("dataset"))
    if not config.get("synthetic", False):
        return None, name, config

    dataset_type = str(
        config.get("dataset_type", "multiscale" if "model" in config else "decay")
    ).lower()
    if dataset_type == "multiscale":
        return "multiscale", name, config
    if dataset_type != "decay":
        raise ValueError(f"Unsupported dataset_type={dataset_type!r} in {name}")

    missing = [key for key in ("decay_type", "decay_param") if key not in config]
    if missing:
        raise ValueError(f"{name} is synthetic but is missing: {', '.join(missing)}")
    decay_type = str(config["decay_type"]).lower()
    if decay_type not in {"exp", "poly"}:
        raise ValueError(f"Unsupported decay_type {decay_type!r} in {name}")
    return "decay", name, config


def synthetic_dataset_path(config, data_dir=DATA_DIR):
    """Return the NPZ path for a synthetic config, or ``None`` if not synthetic."""
    kind, name, _ = _resolve_config(config)
    return None if kind is None else Path(data_dir) / f"{name}.npz"


def _generation_parameters(kind, config):
    if kind == "decay":
        return {
            "decay_type": str(config["decay_type"]).lower(),
            "decay_param": float(config["decay_param"]),
            "n": int(config.get("n", 2000)),
            "d": int(config.get("d", 2000)),
            "random_state": config.get("random_state", 0),
        }

    level_weights = config.get("level_weights")
    ambient_dim = config.get("ambient_dim")
    return {
        "model": str(config.get("model", "exact")),
        "n_clusters": int(config.get("n_clusters", 64)),
        "columns_per_cluster": np.asarray(config.get("columns_per_cluster", 8)).tolist(),
        "high_energy": float(config.get("high_energy", 1.0)),
        "condition_number": float(config.get("condition_number", 1.0e4)),
        "ambient_dim": None if ambient_dim is None else int(ambient_dim),
        "tube_dim": np.asarray(config.get("tube_dim", 3)).tolist(),
        "eta": float(config.get("eta", 0.03)),
        "amplitude_spread": float(config.get("amplitude_spread", 0.8)),
        "balanced_transverse": bool(config.get("balanced_transverse", True)),
        "basis": str(config.get("basis", "random")),
        "level_weights": (
            None if level_weights is None
            else np.asarray(level_weights, dtype=np.float64).tolist()
        ),
        "decay_rate": str(config.get("decay_rate", "polynomial")),
        "seed": int(config.get("seed", 0)),
    }


def ensure_synthetic_dataset(config, force=False, data_dir=DATA_DIR):
    """Generate the dataset described by ``config`` unless it already exists.

    Parameters
    ----------
    config : dict
        Parsed experiment YAML.
    force : bool, default=False
        Regenerate even when the NPZ already exists.

    Returns
    -------
    Path or None
        Location of the dataset, or ``None`` if ``config`` does not describe a
        synthetic dataset (nothing is generated in that case).
    """
    kind, name, config = _resolve_config(config)
    if kind is None:
        return None

    parameters = _generation_parameters(kind, config)
    generation_config = json.dumps(
        {"kind": kind, "parameters": parameters},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    output_path = Path(data_dir) / f"{name}.npz"
    if output_path.exists() and not force:
        with np.load(output_path, allow_pickle=False) as archive:
            if (
                "generation_config_json" not in archive
                or archive["generation_config_json"].item() != generation_config
            ):
                raise ValueError(
                    f"Cached dataset {output_path} has missing or mismatched generation "
                    "metadata; regenerate it with force=True or remove the archive."
                )
        if kind == "multiscale" and config.get("model", "exact") == "tight_frame_union":
            n_clusters = int(config.get("n_clusters", 64))
            _validate_tight_frame_dimensions(
                _normalize_per_cluster_param(
                    config.get("columns_per_cluster", 8), n_clusters, "columns_per_cluster"
                ),
                _normalize_per_cluster_param(
                    config.get("tube_dim", 3), n_clusters, "tube_dim"
                ),
            )
            _validate_cached_tight_frame_dimensions(output_path)
        return output_path

    if kind == "multiscale":
        model = parameters["model"]
        if model not in MULTISCALE_MODELS:
            raise ValueError(f"Unsupported multiscale model {model!r} in {name}")
        dataset = generate_multiscale_dataset(**parameters)
        dataset.save(output_path, generation_config=generation_config)
        return output_path

    n = parameters["n"]
    d = parameters["d"]
    if n < 1 or d < 1:
        raise ValueError(f"n and d must be positive in {name}")
    X = generate_synthetic_dataset(**parameters)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path, X=X, generation_config_json=np.asarray(generation_config)
    )
    return output_path
