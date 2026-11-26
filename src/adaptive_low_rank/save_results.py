import json
import pickle
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def save_results(results, config_path, output_dir, metadata=None):
    """Serialize benchmark results and write a tabular summary.

    Parameters
    ----------
    results : list
        Output of :func:`benchmark`.
    config_path : Path
        Path to the YAML configuration used for the experiment.
    output_dir : Path
        New destination directory. Existing directories are not overwritten.
    metadata : dict, optional
        Experiment provenance to write as ``metadata.json``.

    Notes
    -----
    The function writes ``results.pkl``, a copy of the configuration as
    ``config.yml``, and a ``summary.csv`` file.
    """
    output_dir = Path(output_dir)
    config_path = Path(config_path)

    output_dir.mkdir(parents=True, exist_ok=False)

    with open(output_dir / "results.pkl", "wb") as f:
        pickle.dump(results, f)

    shutil.copy2(config_path, output_dir / "config.yml")

    if metadata is not None:
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
            f.write("\n")

    save_gains = any(
        hasattr(run["result"], "gains_bm")
        and run["result"].gains_bm is not None
        and not np.all(np.isnan(run["result"].gains_bm))
        for run in results
    )

    rows = []

    for run in results:
        result = run["result"]

        row = {
            "algorithm": run["algorithm"],
            **run["parameters"],
            "final_residual": result.residuals[-1],
        }

        if result.runtimes.size:
            row["runtime"] = result.runtimes[-1]

        if save_gains:
            row["final_gains_bm"] = result.gains_bm[-1]
            row["final_gains_as"] = result.gains_as[-1]

        rows.append(row)

    pd.DataFrame(rows).to_csv(output_dir / "summary.csv", index=False)
