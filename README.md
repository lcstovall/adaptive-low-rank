# Adaptive Low-Rank Matrix Approximation

A Python package for benchmarking adaptive algorithms for low-rank matrix
approximation via column subset selection.

This repository provides a common framework for implementing, comparing,
and evaluating algorithms that iteratively select informative columns of a
matrix. It includes experiment management, plotting utilities, and
configurable benchmarking through YAML configuration files.

---

## Features

- Multiple column selection algorithms:
  - Adaptive Sampling
  - Batch-Max
  - Greedy
  - Greedy++
  - Random Selection
  - Sequential Selection
- Configurable experiments using YAML
- Automatic benchmarking across parameter grids
- Publication-quality plots of
  - normalized residuals
  - alpha values (when computed)
  - theory and empirical residual curves
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

Alternatively, install the package with notebook dependencies via the
requirements file (it simply runs `pip install -e ".[notebooks]"`):

```bash
pip install -r requirements.txt
```

## Notebooks

Run an experiment first; each notebook reads from `results/<dataset>/` and
writes figures to `figures/`.

| Notebook | Purpose |
| --- | --- |
| `plotting_residuals.ipynb` | Normalized residual curves |
| `plotting_alphas.ipynb` | Batch-Max alpha trajectories |
| `plotting_synthetic.ipynb` | Residual plots for synthetic datasets |
| `plotting_theory.ipynb` | Theory vs. empirical residual curves |
| `runtime_tables.ipynb` | Runtime-scaling tables |

### Plotting Theory Curves

Use `notebooks/plotting_theory.ipynb` to compare theoretical bounds,
empirical residuals, and optimal rank-k residuals. The notebook loads each
experiment's saved results and configuration, calls the reusable calculations
in `adaptive_low_rank.theory`, and creates the figures with
`adaptive_low_rank.plotting.plot_theory_curves`.

### Generating Synthetic Datasets

Synthetic datasets are built automatically the first time an experiment needs
them: `scripts/run_experiment.py` passes the experiment YAML to
`load_dataset`, which generates `data/<dataset>.npz` if it is missing before
the algorithms run. A config is synthetic only if it sets `synthetic: true`;
`dataset:` is the name of the generated file, and neither it nor the YAML
filename affects generation. `dataset_type` is `decay` (default) or
`multiscale`, which is the default when `model` is set.

To regenerate a dataset after changing its config, delete its `.npz` file or
call `ensure_synthetic_dataset(config, force=True)`.

For example, `configs/exp.yml` produces `data/exp.npz`.

The generators live in `adaptive_low_rank.synthetic`
(`generate_multiscale_dataset`, `ensure_synthetic_dataset`) and
`adaptive_low_rank.datasets` (`generate_synthetic_dataset`). A generated
dataset can be loaded with `load_dataset("exp")`.

---

## Repository Structure

```
adaptive-low-rank/
│
├── configs/                # Experiment YAML files
├── data/                   # Datasets
├── figures/                # Generated figures
├── notebooks/              # Analysis and plotting notebooks
├── results/                # Saved experiment outputs
├── scripts/
│   ├── run_all_experiments.py
│   ├── run_experiment.py   # Main experiment runner
│   └── runtime_scaling.py
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
│       ├── save_results.py
│       ├── synthetic.py
│       └── theory.py
│
├── LICENSE
├── pyproject.toml
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Running an Experiment

Experiments are specified using YAML.

Example:

```yaml
dataset: interactions

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
python scripts/run_experiment.py interactions.yml
```

or simply

```bash
python scripts/run_experiment.py
```

to use the default configuration (`interactions.yml`). The argument is the
name of a file inside `configs/`, not a path.

Run every YAML configuration in `configs/` with

```bash
python scripts/run_all_experiments.py
```

Use `--continue-on-error` to run remaining configurations after a failure, or
`--pattern 'interactions*.yml'` to select a subset of configuration files.

Synthetic datasets are generated automatically when an experiment runs and
`data/<dataset>.npz` is missing. For decay-based datasets, set
`synthetic: true` and provide `decay_type` (`exp` or `poly`) and
`decay_param`. The optional `n`, `d`, and `random_state` fields default to
`2000`, `2000`, and `0`. Model-based multiscale datasets use model-specific
configuration instead; see `configs/orthogonal_tubes_1.yml` and
`configs/orthogonal_tubes_2.yml`. Generated matrices are saved as
`data/<dataset>.npz`.

---

## Experiment Configuration

Each experiment consists of

- a dataset identifier
- optional experiment-wide parameters
- algorithm-specific parameter grids

Parameters that are lists are expanded into every combination automatically.

Algorithm names used as keys under `algorithms:`:

| Key | Algorithm |
| --- | --- |
| `adaptive` | Adaptive Sampling |
| `batch_max` | Batch-Max |
| `greedy` | Greedy |
| `greedy_pp` | Greedy++ |
| `random` | Random Selection |
| `sequential` | Sequential Selection |

Example

```yaml
dataset: interactions
algorithms:
  batch_max:
    k: [50, 100]
    random_state: [0, 1, 2]
    n_candidates: [5, 10]
```

produces

```
2 × 3 × 2 = 12
```

benchmark runs.

---

## Alpha Diagnostics

Batch-Max can record per-iteration gain diagnostics used to calculate the
alpha curves. Enable them on the `batch_max` algorithm entry:

```yaml
algorithms:
  batch_max:
    k: [120]
    n_candidates: [10, 50]
    compute_alpha: true
```

When Batch-Max diagnostics are available, `summary.csv` includes the
`final_gains_bm` and `final_gains_as` values; the alpha plotting helper derives
alpha as the ratio of the averaged gains minus one.

---

## Output

Running an experiment saves these files under `results/<dataset>/`:

```
results.pkl
summary.csv
config.yml
```

`summary.csv` contains one row per run with the algorithm, its parameters,
`final_residual`, and `runtime` when recorded.

The experiment runner does not generate plots. Use the plotting notebooks or
plotting helpers separately to create figures.

---

## Runtime Scaling

```bash
python scripts/runtime_scaling.py
```

benchmarks Adaptive, Batch-Max, Greedy, and Greedy++ on synthetic matrices of
increasing size and saves `results/runtime_scaling/runtime_scaling.pkl`.
`notebooks/runtime_tables.ipynb` turns it into mean and standard-deviation
CSV tables under `results/runtime_scaling/tables/`.

---

## License

See [LICENSE](LICENSE).

Runtime measurement is disabled by default. Set `compute_runtime: true` for
an algorithm run when runtime data is needed. `summary.csv` contains the
algorithm and run parameters, plus:

- `final_residual`
- `runtime`, if computed
- `final_gains_bm` and `final_gains_as`, if Batch-Max diagnostics are available

---

## Adding a New Algorithm

Create a subclass of `LowRankAlgorithm`.

Implement the following method (with NumPy imported as `np`):

```python
def select_index(
    self,
    R: np.ndarray,
    k: int,
    random_state: np.random.RandomState,
    n_candidates: int | None = None,
    V: np.ndarray | None = None,
    compute_alpha: bool = False,
) -> tuple[int, tuple[float, float] | None]:
    ...
```

which returns

```python
(index, gains)
```

where `index` is the selected column and `gains` is either a
`(batch_max_gain, adaptive_sampling_gain)` tuple or `None` when the algorithm
does not compute gain diagnostics. See `LowRankAlgorithm.select_index` for the
full signature.

Import the class and add it to the `ALGORITHMS` mapping in `registry.py`.

For example, add this entry:

```python
"my_algorithm": MyAlgorithm,
```

It will automatically work with the benchmarking framework.

---

## Dependencies

The package dependencies are declared in `pyproject.toml`: NumPy, SciPy,
Matplotlib, pandas, PyYAML, scikit-learn, Pillow, and GraphLearning. Install
the optional notebook tools with `pip install -e ".[notebooks]"`.

Install the core package dependencies with

```bash
pip install -e .
```

---

## Authors

Luke Stovall  
Kevin Miller

---

## License

This project is licensed under the MIT License. The project authors are listed
above; the copyright notice names Luke Stovall as the copyright holder. See
[LICENSE](./LICENSE) for the license terms.
