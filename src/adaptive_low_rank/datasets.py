from pathlib import Path
from scipy.io import loadmat
from sklearn.datasets import fetch_openml
import numpy as np


def load_dataset(name):

    root = Path(__file__).resolve().parents[2]

    if name == "interactions":
        return loadmat(root / "data" / "interactions.mat")["B"]


    if name == "mnist":
        mnist = fetch_openml('mnist_784', as_frame=False, parser='auto')
        X = mnist['data'].astype(np.float64)
        return X
    
    raise ValueError(f"Unknown dataset '{name}'")