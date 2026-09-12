"""Generate synthetic datasets for experiments.

The script's command-line entry point generates the YAML-described datasets.
The public :func:`generate_multiscale_dataset` function below provides a
structured multiscale dataset for experiments that need cluster metadata.
"""

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from numpy.typing import NDArray

from adaptive_low_rank.datasets import generate_synthetic_dataset

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"
DATA_DIR = ROOT / "data"
SYNTHETIC_NAME = re.compile(r"^(exp|poly).+$")


def _generate(config_path, force=False):
    name = config_path.stem

    with config_path.open() as file:
        config = yaml.safe_load(file) or {}

    dataset_type = str(config.get("dataset_type", "decay")).lower()
    if dataset_type == "multiscale":
        return _generate_multiscale(name, config, force=force)
    if dataset_type != "decay":
        raise ValueError(
            f"Unsupported dataset_type={dataset_type!r} in {config_path.name}"
        )
    if not SYNTHETIC_NAME.fullmatch(name):
        return False

    missing = [key for key in ("decay_type", "decay_param") if key not in config]
    if missing:
        raise ValueError(
            f"{config_path.name} is synthetic but is missing: {', '.join(missing)}"
        )

    output_path = DATA_DIR / f"{name}.npz"
    if output_path.exists() and not force:
        print(f"Skipping {name}: {output_path.name} already exists")
        return True

    decay_type = str(config["decay_type"]).lower()
    if decay_type not in {"exp", "poly"}:
        raise ValueError(f"Unsupported decay_type {decay_type!r} in {config_path.name}")
    if not name.startswith(decay_type):
        raise ValueError(
            f"{config_path.name} starts with a different decay type than "
            f"decay_type={decay_type!r}"
        )

    n = int(config.get("n", 2000))
    d = int(config.get("d", 2000))
    random_state = config.get("random_state", 0)
    if n < 1 or d < 1:
        raise ValueError(f"n and d must be positive in {config_path.name}")

    X = generate_synthetic_dataset(
        decay_type=decay_type,
        decay_param=float(config["decay_param"]),
        n=n,
        d=d,
        random_state=random_state,
    )
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, X=X)
    print(f"Generated {output_path} with shape {X.shape}")
    return True


def _generate_multiscale(name, config, force=False):
    output_path = DATA_DIR / f"{name}.npz"
    if output_path.exists() and not force:
        print(f"Skipping {name}: {output_path.name} already exists")
        return True

    model = str(config.get("model", "exact"))
    if model not in {"exact", "orthogonal_tubes", "ambient_noise"}:
        raise ValueError(f"Unsupported multiscale model {model!r} in {name}.yml")

    dataset = generate_multiscale_dataset(
        model=model,
        n_clusters=int(config.get("n_clusters", 64)),
        columns_per_cluster=int(config.get("columns_per_cluster", 8)),
        high_energy=float(config.get("high_energy", 1.0)),
        condition_number=float(config.get("condition_number", 1.0e4)),
        ambient_dim=(
            None
            if config.get("ambient_dim") is None
            else int(config["ambient_dim"])
        ),
        tube_dim=int(config.get("tube_dim", 3)),
        eta=float(config.get("eta", 0.03)),
        amplitude_spread=float(config.get("amplitude_spread", 0.8)),
        balanced_transverse=bool(config.get("balanced_transverse", True)),
        basis=str(config.get("basis", "random")),
        seed=int(config.get("seed", 0)),
    )
    dataset.save(output_path)
    print(f"Generated {output_path} with shape {dataset.X.shape}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Generate datasets described by synthetic YAML configs."
    )
    parser.add_argument(
        "--pattern",
        default=None,
        help="Only process matching config filenames, such as exp*.yml or multiscale*.yml.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate datasets even when the NPZ already exists.",
    )
    args = parser.parse_args()

    if args.pattern is None:
        configs = sorted(set(CONFIG_DIR.glob("*.yml")) | set(CONFIG_DIR.glob("*.yaml")))
    else:
        configs = sorted(CONFIG_DIR.glob(args.pattern))
    generated = [_generate(path, force=args.force) for path in configs]
    count = sum(generated)
    print(f"Processed {count} synthetic dataset(s)")


# Multiscale dataset API
Array = NDArray[np.float64]
ModelName = Literal["exact", "orthogonal_tubes", "ambient_noise"]


@dataclass
class MultiscaleDataset:
    """Container returned by :func:`generate_multiscale_dataset`."""

    X: Array
    labels: NDArray[np.int64]
    central_energies: Array
    central_directions: Array
    amplitudes: Array
    model: str
    metadata: dict[str, Any]
    tube_bases: Array | None = None

    def save(self, path: str | Path) -> Path:
        """Save the dataset and its construction metadata to a compressed NPZ."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "X": self.X,
            "labels": self.labels,
            "central_energies": self.central_energies,
            "central_directions": self.central_directions,
            "amplitudes": self.amplitudes,
            "model": np.asarray(self.model),
            "metadata_json": np.asarray(json.dumps(self.metadata, sort_keys=True)),
        }
        if self.tube_bases is not None:
            payload["tube_bases"] = self.tube_bases
        np.savez_compressed(destination, **payload)
        return destination


def log_spaced_energies(
    n_clusters: int,
    high_energy: float = 1.0,
    condition_number: float = 1.0e4,
) -> Array:
    """Return energies decreasing geometrically from H to H/kappa."""
    if n_clusters < 2:
        raise ValueError("n_clusters must be at least 2")
    if high_energy <= 0:
        raise ValueError("high_energy must be positive")
    if condition_number < 1:
        raise ValueError("condition_number must be at least 1")
    return np.geomspace(
        high_energy,
        high_energy / condition_number,
        num=n_clusters,
        dtype=np.float64,
    )


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


def _cluster_amplitudes(
    energies: Array,
    columns_per_cluster: int,
    rng: np.random.Generator,
    spread: float,
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
    transverse_dim: int,
    amplitudes: Array,
    rng: np.random.Generator,
    balanced: bool,
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


def generate_multiscale_dataset(
    *,
    model: ModelName = "exact",
    n_clusters: int = 64,
    columns_per_cluster: int = 8,
    high_energy: float = 1.0,
    condition_number: float = 1.0e4,
    ambient_dim: int | None = None,
    tube_dim: int = 3,
    eta: float = 0.03,
    amplitude_spread: float = 0.8,
    balanced_transverse: bool = True,
    basis: Literal["random", "coordinate"] = "random",
    seed: int = 0,
) -> MultiscaleDataset:
    """Generate one exact, orthogonal-tube, or ambient-noise dataset.

    Parameters
    ----------
    model:
        ``exact``, ``orthogonal_tubes``, or ``ambient_noise``.
    n_clusters:
        Number of distinct central directions and energy levels.
    columns_per_cluster:
        Number of differently scaled columns associated with each direction.
    condition_number:
        Ratio between the largest and smallest prescribed central energies.
    ambient_dim:
        Row dimension of ``X``.  The minimum is ``n_clusters`` for ``exact``
        and ``ambient_noise``, and ``n_clusters * tube_dim`` for orthogonal
        tubes.
    tube_dim:
        Dimension of each mutually orthogonal tube subspace, including its
        central direction.
    eta:
        Maximum transverse-to-central norm ratio for each column.
    amplitude_spread:
        Standard deviation of lognormal raw column magnitudes before each
        cluster is normalized to its requested energy.
    balanced_transverse:
        In the orthogonal-tube model, center transverse deviations so that the
        declared central direction is an exact covariance eigenvector.
    """
    if model not in {"exact", "orthogonal_tubes", "ambient_noise"}:
        raise ValueError(f"unknown model: {model}")
    if eta < 0:
        raise ValueError("eta must be nonnegative")
    if tube_dim < 1:
        raise ValueError("tube_dim must be positive")

    rng = np.random.default_rng(seed)
    energies = log_spaced_energies(n_clusters, high_energy, condition_number)
    amplitudes = _cluster_amplitudes(
        energies, columns_per_cluster, rng, amplitude_spread
    )

    required_dim = n_clusters * tube_dim if model == "orthogonal_tubes" else n_clusters
    if ambient_dim is None:
        ambient_dim = required_dim
    if ambient_dim < required_dim:
        raise ValueError(
            f"ambient_dim={ambient_dim} is too small for model={model}; "
            f"it must be at least {required_dim}"
        )

    total_basis_dim = (
        n_clusters * tube_dim if model == "orthogonal_tubes" else n_clusters
    )
    Q = _orthonormal_columns(ambient_dim, total_basis_dim, rng, basis)

    if model == "orthogonal_tubes":
        tube_bases = np.empty(
            (n_clusters, ambient_dim, tube_dim), dtype=np.float64
        )
        central_directions = np.empty((ambient_dim, n_clusters), dtype=np.float64)
        for i in range(n_clusters):
            block = Q[:, i * tube_dim : (i + 1) * tube_dim]
            tube_bases[i] = block
            central_directions[:, i] = block[:, 0]
    else:
        tube_bases = None
        central_directions = Q[:, :n_clusters]

    n_columns = n_clusters * columns_per_cluster
    X = np.empty((ambient_dim, n_columns), dtype=np.float64)
    labels = np.repeat(np.arange(n_clusters, dtype=np.int64), columns_per_cluster)

    for i in range(n_clusters):
        start = i * columns_per_cluster
        stop = start + columns_per_cluster
        a_i = amplitudes[i]
        u_i = central_directions[:, i]

        if model == "exact" or eta == 0:
            X[:, start:stop] = u_i[:, None] * a_i[None, :]
            continue

        if model == "orthogonal_tubes":
            V_i = tube_bases[i, :, 1:]
            z_i = _transverse_coordinates(
                tube_dim - 1, a_i, rng, balanced_transverse
            )
            transverse = V_i @ z_i
        else:
            # Arbitrary ambient perturbations: only remove the component along
            # this cluster's own center.  The result may point toward another
            # cluster and deliberately tests cross-cluster leakage.
            transverse = rng.standard_normal((ambient_dim, columns_per_cluster))
            transverse -= u_i[:, None] * (u_i @ transverse)[None, :]
            norms = np.linalg.norm(transverse, axis=0, keepdims=True)
            transverse /= np.maximum(norms, np.finfo(np.float64).tiny)
            transverse *= rng.random((1, columns_per_cluster))

        X[:, start:stop] = (
            u_i[:, None] * a_i[None, :]
            + eta * transverse * a_i[None, :]
        )

    metadata: dict[str, Any] = {
        "model": model,
        "seed": seed,
        "ambient_dim": ambient_dim,
        "n_columns": n_columns,
        "n_clusters": n_clusters,
        "columns_per_cluster": columns_per_cluster,
        "high_energy": high_energy,
        "condition_number": condition_number,
        "tube_dim": tube_dim if model == "orthogonal_tubes" else None,
        "eta": eta if model != "exact" else 0.0,
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


if __name__ == "__main__":
    main()
