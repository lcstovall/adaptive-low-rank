import argparse
from pathlib import Path

import yaml

from adaptive_low_rank.benchmark import benchmark, experiment_metadata
from adaptive_low_rank.datasets import load_dataset
from adaptive_low_rank.run_generator import generate_runs
from adaptive_low_rank.save_results import save_results

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument(
    "config",
    nargs="?",
    default="interactions.yml",
    help="Experiment YAML file (default: interactions.yml)",
)
args = parser.parse_args()
config_path = ROOT / "configs" / args.config


with open(config_path) as f:
    experiment = yaml.safe_load(f)

output_dir = ROOT / "results" / experiment["dataset"]
if output_dir.exists():
    raise FileExistsError(
        f"Results already exist at {output_dir}; move them before running again."
    )

X = load_dataset(config=experiment)
metadata = experiment_metadata(X, ROOT)

runs = generate_runs(experiment)

results = benchmark(X=X, runs=runs)

save_results(results, config_path, output_dir, metadata=metadata)
