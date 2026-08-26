# Adaptive Low-Rank Matrix Approximation

A Python package for benchmarking adaptive algorithms for low-rank matrix approximation via column subset selection.

This repository provides a common framework for implementing, comparing, and evaluating algorithms that iteratively select informative columns of a matrix. It includes experiment management, plotting utilities, and reproducible benchmarking through YAML configuration files.

---

## Features

- Multiple column selection algorithms:
  - Adaptive Sampling
  - Batch Max
  - Greedy
  - Greedy++
  - Random Selection
- Configurable experiments using YAML
- Automatic benchmarking across parameter grids
- Publication-quality plots of
  - normalized residuals
  - alpha values (when computed)
- Automatic saving of
  - raw benchmark results
  - summary CSV
  - experiment configuration
- Easily extensible algorithm interface

---

## Installation

Clone the repository

```bash
git clone https://github.com/lcstovall/adaptive-low-rank.git
cd adaptive-low-rank
```

Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the package

```bash
pip install -e .
```

To run the plotting notebooks as well, install the notebook dependencies:

```bash
pip install -e ".[notebooks]"
```

Alternatively, install the complete requirements file into the virtual environment:

```bash
pip install -r requirements.txt
```

## Plotting Optimal Residual Curves

To compare the shape of the optimal low-rank residual curves across the
configured data sets, run:

```bash
python3 scripts/plot_optimal_residuals.py
```

The script excludes `mnist`, `cfar10`, `interactions_alpha`, and
`yearprediction_alpha`. It normalizes
the horizontal axis by each data set's maximum rank and the residual by its
initial Frobenius norm, then writes
`figures/optimal_residuals_normalized.png`. Use `--yscale log` to emphasize
small residual differences.

## Generating a Synthetic SVD Dataset

Create a reproducible `2500 x 2500` matrix whose first 350 singular values
follow an exponential decay with rate `0.1` per rank and whose remaining
singular values are flat. Its columns contain 75 informative columns, 500
correlated columns, and a weak-column tail scaled by `0.005`:

```bash
python3 scripts/generate_synthetic_dataset.py
```

The output is `data/synthetic_exp4.npz`. The generator is also available as
`generate_synthetic_svd_dataset` in `adaptive_low_rank.datasets`, and the
deterministic default profile can be loaded with `load_dataset("synthetic_exp4")`.

---

## Repository Structure

```
adaptive-low-rank/
│
├── configs/                # Experiment YAML files
├── data/                   # Datasets
├── results/                # Saved experiment outputs
├── scripts/
│   └── run_experiment.py   # Main experiment runner
│
├── src/
│   └── adaptive_low_rank/
│       ├── algorithms.py
│       ├── benchmark.py
│       ├── datasets.py
│       ├── plotting.py
│       ├── registry.py
│       ├── results.py
│       ├── run_generator.py
│       └── save_results.py
│
├── pyproject.toml
└── README.md
```

---

## Running an Experiment

Experiments are specified using YAML.

Example:

```yaml
dataset: interactions
name: interactions_residual

r: 50

algorithms:

  adaptive:
    k: [120]
    random_state: [0,1,2,3,4]

  batch_max:
    k: [120]
    random_state: [0,1,2,3,4]
    n_candidates: [10]

  greedy:
    k: [120]

  random:
    k: [120]
    random_state: [0,1,2,3,4]
```

Run the experiment with

```bash
python scripts/run_experiment.py configs/interactions.yml
```

or simply

```bash
python scripts/run_experiment.py
```

to use the default configuration.

Run every YAML configuration in `configs/` with

```bash
python scripts/run_all_experiments.py
```

Use `--continue-on-error` to run remaining configurations after a failure, or
`--pattern '*.yaml'` to select a different filename pattern.

Generate synthetic datasets from configurations whose filenames start with
`exp` or `poly` with

```bash
python scripts/generate_synthetic_datasets.py
```

Synthetic configurations must provide `decay_type` (`exp` or `poly`) and
`decay_param`. The optional `n`, `d`, and `random_state` fields default to
`2000`, `2000`, and `0`. Generated matrices are saved as `data/<config>.npz`;
use `--force` to regenerate an existing file.

---

## Experiment Configuration

Each experiment consists of

- a dataset
- an output name
- optional experiment-wide parameters
- algorithm-specific parameter grids

Parameters that are lists are expanded into every combination automatically.

Example

```yaml
batch_max:

  k: [50,100]
  random_state: [0,1,2]
  n_candidates: [5,10]
```

produces

```
2 × 3 × 2 = 12
```

benchmark runs.

---

## Alpha Computation

Adaptive Sampling and Batch Max can optionally compute the theoretical alpha value.

Simply include

```yaml
r: 50
```

at the top level of the experiment configuration.

The truncated right singular vectors are computed **once** and reused for every run.

If `r` is omitted, alpha values are not computed.

---

## Output

Each experiment creates a results directory containing

```
results.pkl
summary.csv
config.yml
residuals.pdf
residuals.png
alphas.pdf
alphas.png
```

Runtime measurement is disabled by default. Set `compute_runtime: true` for
an algorithm run when runtime data is needed. `summary.csv` contains

- algorithm
- parameters
- final residual
- runtime (if computed)
- final alpha (if available)

---

## Adding a New Algorithm

Create a subclass of `LowRankAlgorithm`.

Implement

```python
select_index(...)
```

which returns

```python
(index, alpha)
```

where `alpha` should be `None` if the algorithm does not compute one.

Then register the algorithm in

```python
registry.py
```

For example

```python
ALGORITHMS = {
    ...
    "my_algorithm": MyAlgorithm,
}
```

It will automatically work with the benchmarking framework.

---

## Dependencies

- numpy
- scipy
- matplotlib
- pandas
- pyyaml

Install everything with

```bash
pip install -e .
```

---

## Authors

Luke Stovall  
Kevin Miller

---

## License

This project is licensed under the MIT License.