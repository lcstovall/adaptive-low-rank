from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import loadmat
from sklearn.datasets import fetch_openml

from adaptive_low_rank.synthetic import ensure_synthetic_dataset


def load_dataset(name=None, config=None):
    """Load a supported data set in matrix form.

    Parameters
    ----------
    name : str, optional
        Data-set identifier. Built-in identifiers are ``interactions``,
        ``mnist``, ``yearprediction``, ``coil20``, and ``cifar10``. Legacy
        identifiers ``mnistT`` and ``cfar10T`` are also accepted. Other names
        are loaded from ``data/<name>.npz`` when that archive exists. Ignored
        when ``config`` is given.
    config : dict, optional
        Parsed experiment YAML; its ``dataset`` field is used as the name. If
        it sets ``synthetic: true``, ``data/<dataset>.npz`` is generated when
        missing and then loaded.

    Returns
    -------
    np.ndarray
        Data matrix in the loader-specific orientation. ``mnist``, ``mnistT``,
        ``cifar10``, ``cfar10T``, and ``yearprediction`` have samples in
        columns; ``coil20`` has images in rows. Other loaders preserve the
        orientation of their source or generated matrix.

    Raises
    ------
    ValueError
        If ``name`` is not a supported identifier, or neither ``name`` nor
        ``config`` is given.
    """

    root = Path(__file__).resolve().parents[2]

    if config is not None:
        name = config["dataset"]
        synthetic_path = ensure_synthetic_dataset(config)
        if synthetic_path is not None:
            with np.load(synthetic_path) as data:
                return data["X"]
    if name is None:
        raise ValueError("Provide a dataset name or a config.")

    if name == "interactions":
        data = loadmat(root / "data" / "interactions.mat")["B"]
        return data

    elif (root / "data" / f"{name}.npz").exists():
        with np.load(root / "data" / f"{name}.npz") as data:
            return data["X"]

    elif name in ("mnist", "mnistT"):
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

        data = np.asarray(images)

        return data

    elif name in ("cifar10", "cfar10T"):
        data_path = root / "data" / "cifar10_simclr.npz"
        with np.load(data_path) as archive:
            data = archive["data"]
        return data.T

    else:
        raise ValueError(f"Unknown dataset '{name}'")
