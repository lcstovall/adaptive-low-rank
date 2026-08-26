from pathlib import Path
from scipy.io import loadmat
from sklearn.datasets import fetch_openml
import numpy as np
from PIL import Image
import graphlearning as gl


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


def load_dataset(name):
    """Load a supported data set in matrix form.

    Parameters
    ----------
    name : str
        Data-set identifier. Supported identifiers are ``interactions``,
        ``mnist``, ``mnistT``, ``yearprediction``, ``coil20``, ``cfar10``,
        ``cfar10T``, and ``new_data``. Synthetic datasets are loaded from
        generated files named ``data/<name>.npz``.

    Returns
    -------
    np.ndarray
        Data matrix. Transposed variants have samples in columns; the
        untransposed variants retain samples in rows as loaded.

    Raises
    ------
    ValueError
        If ``name`` is not a supported identifier.
    """

    root = Path(__file__).resolve().parents[2]

    if name == "interactions":
        data = loadmat(root / "data" / "interactions.mat")["B"]
        return data

    elif name.startswith("poly") or name.startswith("exp"):
        path = root / "data" / f"{name}.npz"
        if not path.exists():
            raise FileNotFoundError(
                f"Synthetic dataset '{name}' was not found at {path}. "
                "Run scripts/generate_synthetic_datasets.py first."
            )
        with np.load(path) as data:
            return data["X"]

    elif name == "cluster_expansion":
        data = np.load(root / "data" / "cluster_expansion_M.npy")
        indices = np.random.default_rng().choice(data.shape[0], size=5000, replace=False)
        return data[indices, :]

    elif name == "mnistT":
        mnist = fetch_openml("mnist_784", as_frame=False, parser="auto")
        data = mnist["data"].astype(np.float64)
        return data.T

    elif name == "yearprediction":
        data = np.loadtxt(root / "data" / "YearPredictionMSD.txt", delimiter=",")
        return data[:, 1:].T

    elif name == "coil20":
        path = root / "data" / "coil-20-proc"

        images = []
        for file in sorted(path.glob("*.png")):
            image = np.asarray(Image.open(file), dtype=float)
            images.append(image.ravel())

        data = np.asarray(images).T

        return data

    elif name == "cfar10T":
        data, labels = gl.datasets.load("cifar10", metric="simclr")
        return data.T

    else:
        raise ValueError(f"Unknown dataset '{name}'")
