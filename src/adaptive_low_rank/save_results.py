from pathlib import Path
import pickle
import shutil
import pandas as pd


def save_results(results, experiment, config_path, output_dir):
    """
    Save benchmark results.

    Parameters
    ----------
    results : list
        Output of benchmark().
    experiment : dict
        Parsed experiment configuration.
    config_path : Path
        Path to the YAML config used.
    output_dir : Path
        Directory where results should be saved.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save complete results
    with open(output_dir / "results.pkl", "wb") as f:
        pickle.dump(results, f)

    # Save config used
    shutil.copy(config_path, output_dir / "config.yml")

    # Save summary csv
    rows = []
    for run in results:
        row = {
            "algorithm": run["algorithm"]
        }
        row.update(run["parameters"])
        result = run["result"]
        row["final_residual"] = result.residuals[-1]
        row["runtime"] = result.runtimes[-1]

        rows.append(row)

    df = pd.DataFrame(rows)

    df.to_csv(output_dir / "summary.csv", index=False)