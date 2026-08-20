import argparse
from pathlib import Path
import yaml
from adaptive_low_rank.datasets import load_dataset
from adaptive_low_rank.algorithms import LowRankAlgorithm
from adaptive_low_rank.benchmark import benchmark
from adaptive_low_rank.save_results import save_results
from adaptive_low_rank.run_generator import generate_runs
from adaptive_low_rank.plotting import *


# Load config
ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument(
    "config",
    nargs="?",
    default="interactions_residual.yml",
    help="Experiment YAML file (default: interactions_residual.yml)",
)
args = parser.parse_args()
config_path = ROOT / "configs" / args.config


with open(config_path) as f:
    experiment = yaml.safe_load(f)

# Load dataset
X = load_dataset(experiment["dataset"])

# Compute V if needed
V = None
if "r" in experiment:
    V = LowRankAlgorithm.compute_v(X, experiment["r"])
runs = generate_runs(experiment)

results = benchmark(
    X=X,
    runs=runs,
    V=V,
)

# Save
output_dir = ROOT / "results" / experiment["name"]

save_results(
    results,
    config_path,
    output_dir,
)