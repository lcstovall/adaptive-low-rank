from pathlib import Path
from scipy.io import loadmat
from sklearn.datasets import fetch_openml
import numpy as np
import pandas as pd
from PIL import Image
import graphlearning as gl


def load_dataset(name):

    root = Path(__file__).resolve().parents[2]

    if name == "interactions":
        return loadmat(root / "data" / "interactions.mat")["B"]

    elif name == "mnist":
        mnist = fetch_openml('mnist_784', as_frame=False, parser='auto')
        X = mnist['data'].astype(np.float64)
        return X.T

    elif name == "yearprediction":
        data = np.loadtxt(root / "data" / "YearPredictionMSD.txt", delimiter=",")
        return data[:, 1:].T
    
    elif name == "coil20":
        path = root / "data" / "coil-20-proc"

        images = []
        for file in sorted(path.glob("*.png")):
            image = np.asarray(Image.open(file), dtype=float)
            images.append(image.ravel())

        X = np.asarray(images).T
        
        return X

    elif name == "cfar10":
        data, labels = gl.datasets.load("cifar10", metric="simclr")
        return data.T

    else:
        raise ValueError(f"Unknown dataset '{name}'")
