import hashlib
import json
import platform
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
from numpy.linalg import norm

from adaptive_low_rank.registry import ALGORITHMS


def experiment_metadata(X, repository_root):
    """Record the input matrix, environment, and repository revision."""
    matrix = np.asarray(X)
    description = {"shape": list(matrix.shape), "dtype": str(matrix.dtype)}
    digest = hashlib.sha256(json.dumps(description, sort_keys=True).encode("utf-8"))
    chunks = np.nditer(
        matrix,
        flags=["external_loop", "buffered", "zerosize_ok"],
        op_flags=["readonly"],
        order="C",
        buffersize=131072,
    )
    for chunk in chunks:
        digest.update(chunk.tobytes())
    description["sha256"] = digest.hexdigest()

    packages = {}
    for name in (
        "adaptive_low_rank",
        "numpy",
        "scipy",
        "matplotlib",
        "pandas",
        "PyYAML",
        "scikit-learn",
        "pillow",
    ):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "not installed"

    root = Path(repository_root)
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root, check=True, capture_output=True, text=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root, check=True, capture_output=True, text=True,
        ).stdout.strip())
        git = {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError) as error:
        git = {"commit": None, "dirty": None, "error": str(error)}

    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "matrix": description,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "git": git,
    }


def benchmark(X, runs, V=None):
    """Execute a collection of configured benchmark runs.

    Parameters
    ----------
    X : np.ndarray
        Input matrix with shape ``(d, n)``.

    runs : list[dict]
        Run specifications. Each dictionary must contain ``algorithm`` and
        ``k`` and may contain algorithm-specific keyword arguments.

    V : np.ndarray | None
        Truncated right singular vectors used for alpha computation when
        ``compute_alpha=True``.

    Returns
    -------
    list of dict
        One result dictionary per run, containing the algorithm name,
        parameters, initial squared Frobenius norm, and ``AlgorithmResult``.
    """
    results = []

    X_fro_norm = norm(X, ord="fro")

    for run in runs:
        params = deepcopy(run)

        algorithm_name = params.pop("algorithm")
        k = params.pop("k")

        Algorithm = ALGORITHMS[algorithm_name]
        algorithm = Algorithm()

        result = algorithm.select_columns(X, k=k, V=V, **params)

        results.append(
            {
                "algorithm": algorithm_name,
                "parameters": {"k": k, **params},
                "init_res": X_fro_norm,
                "result": result,
            }
        )

    return results
