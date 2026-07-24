from pathlib import Path
from scipy.io import loadmat


def load_dataset(name):

    root = Path(__file__).resolve().parents[2]

    if name == "interactions":
        return loadmat(root / "data" / "interactions.mat")["B"]

    raise ValueError(f"Unknown dataset '{name}'")