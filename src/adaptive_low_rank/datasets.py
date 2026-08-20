from pathlib import Path
from scipy.io import loadmat
from sklearn.datasets import fetch_openml
import numpy as np
from PIL import Image
import graphlearning as gl


def load_dataset(name):
    """Load a supported data set in matrix form.

    Parameters
    ----------
    name : str
        Data-set identifier. Supported identifiers are ``interactions``,
        ``mnist``, ``mnistT``, ``yearprediction``, ``coil20``, ``cfar10``,
        and ``cfar10T``.

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

    elif name == "mnistT":
        mnist = fetch_openml("mnist_784", as_frame=False, parser="auto")
        data = mnist["data"].astype(np.float64)
        return data.T

    elif name == "mnist":
        mnist = fetch_openml("mnist_784", as_frame=False, parser="auto")
        data = mnist["data"].astype(np.float64)
        return data[:1000, :]

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

    elif name == "cfar10":
        data, labels = gl.datasets.load("cifar10", metric="simclr")
        return data[:1000, :]

    else:
        raise ValueError(f"Unknown dataset '{name}'")
