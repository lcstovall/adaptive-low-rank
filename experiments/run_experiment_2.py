from pathlib import Path
import yaml
from adaptive_low_rank.datasets import load_dataset
from adaptive_low_rank.algorithms import LowRankAlgorithm
from adaptive_low_rank.benchmark import benchmark
from adaptive_low_rank.save_results import save_results
from adaptive_low_rank.run_generator import generate_runs
from adaptive_low_rank.plotting import plot_residuals


# ---------------------------------------------------------
# Load config
# ---------------------------------------------------------


ROOT = Path(__file__).resolve().parents[1]

config_path = ROOT / "configs" / "interactions2.yml"

with open(config_path) as f:
    experiment = yaml.safe_load(f)

# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------

X = load_dataset(experiment["dataset"])

runs = generate_runs(experiment)

results = benchmark(
    X=X,
    runs=runs,
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

plot_residuals(results, output_dir)