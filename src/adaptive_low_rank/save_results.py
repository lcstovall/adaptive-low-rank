from pathlib import Path
import pickle
import shutil
import pandas as pd
import numpy as np


def save_results(results, config_path, output_dir):
    """Serialize benchmark results and write a tabular summary.

    Parameters
    ----------
    results : list
        Output of :func:`benchmark`.
    config_path : Path
        Path to the YAML configuration used for the experiment.
    output_dir : Path
        Destination directory. It is created when necessary.

    Notes
    -----
    The function writes ``results.pkl``, a copy of the configuration as
    ``config.yml``, and a ``summary.csv`` file.
    """
    output_dir = Path(output_dir)
    config_path = Path(config_path)

    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "results.pkl", "wb") as f:
        pickle.dump(results, f)

    shutil.copy2(config_path, output_dir / "config.yml")

    save_alpha = any(
        hasattr(run["result"], "alphas")
        and run["result"].alphas is not None
        and not np.all(np.isnan(run["result"].alphas))
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

        if save_alpha:
            row["final_alpha"] = result.alphas[-1]

        rows.append(row)

    pd.DataFrame(rows).to_csv(output_dir / "summary.csv", index=False)
    output_dir.mkdir(parents=True, exist_ok=True)
