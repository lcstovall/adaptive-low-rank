from pathlib import Path
import yaml
from adaptive_low_rank.datasets import load_dataset
from adaptive_low_rank.algorithms import LowRankAlgorithm
from adaptive_low_rank.benchmark import benchmark
from adaptive_low_rank.save_results import save_results
from adaptive_low_rank.run_generator import generate_runs
from adaptive_low_rank.plotting import plot_alphas


# ---------------------------------------------------------
# Load config
# ---------------------------------------------------------


ROOT = Path(__file__).resolve().parents[1]

config_path = ROOT / "configs" / "interactions3.yml"

with open(config_path) as f:
    experiment = yaml.safe_load(f)

# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------

X = load_dataset(experiment["dataset"])

# ---------------------------------------------------------
# Compute V only if r is specified
# ---------------------------------------------------------

# Compute V once for the entire experiment if r is provided.
V = None
if "r" in experiment:
    V = LowRankAlgorithm.compute_v(X, experiment["r"])

runs = generate_runs(experiment)

results = benchmark(
    X=X,
    runs=runs,
    V=V,
)

# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

output_dir = ROOT / "results" / experiment["name"]

save_results(
    results,
    config_path,
    output_dir,
)

plot_alphas(results, output_dir)