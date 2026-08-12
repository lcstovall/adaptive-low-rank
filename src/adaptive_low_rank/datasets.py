from pathlib import Path
from scipy.io import loadmat
from sklearn.datasets import fetch_openml
import numpy as np
import pandas as pd
from PIL import Image


def load_dataset(name):

    root = Path(__file__).resolve().parents[2]

    if name == "interactions":
        return loadmat(root / "data" / "interactions.mat")["B"]

    elif name == "mnist":
        mnist = fetch_openml('mnist_784', as_frame=False, parser='auto')
        X = mnist['data'].astype(np.float64)
        return X

    elif name == "yearprediction":
        data = np.loadtxt(root / "data" / "YearPredictionMSD.txt", delimiter=",")
        return data[:, 1:]

    elif name == "movielens":
        ratings = pd.read_csv(root / "data" / "rating.csv")

        user_ids = ratings["userId"].unique()
        movie_ids = ratings["movieId"].unique()

        user_map = {user_id: i for i, user_id in enumerate(user_ids)}
        movie_map = {movie_id: i for i, movie_id in enumerate(movie_ids)}

        X = np.zeros(
            (len(user_ids), len(movie_ids)),
            dtype=float,
        )

        rows = ratings["userId"].map(user_map).to_numpy()
        cols = ratings["movieId"].map(movie_map).to_numpy()

        X[rows, cols] = ratings["rating"].to_numpy()

        return X

    elif name == "coil20":
        path = root / "data" / "coil-20-proc"

        images = []
        for file in sorted(path.glob("*.png")):
            image = np.asarray(Image.open(file), dtype=float)
            images.append(image.ravel())

        X = np.asarray(images).T
        
        return X

    else:
        raise ValueError(f"Unknown dataset '{name}'")
