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

## Notebooks

Run an experiment first. The plotting notebooks read from
`results/<dataset>/` and write figures to `figures/`. The runtime tables
notebook reads the runtime-scaling results and writes CSV tables under
`results/runtime_scaling/tables/`.

| Notebook | Purpose |
| --- | --- |
| `plotting_residuals.ipynb` | Normalized residual curves |
| `plotting_alphas.ipynb` | Batch-Max alpha trajectories |
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
`load_dataset`, which generates `data/<dataset>.npz` if it is missing. Cached
archives are checked against the generation parameters; an archive with
missing or mismatched metadata raises an error instead of being reused. A
config is synthetic only if it sets `synthetic: true`;
`dataset:` is the name of the generated file, and neither it nor the YAML
filename affects generation. `dataset_type` is `decay` (default) or
`multiscale`, which is the default when `model` is set.

To regenerate a dataset after changing its config, or to replace an older
archive without generation metadata, delete its `.npz` file or call
`ensure_synthetic_dataset(config, force=True)`.

For example, `configs/poly.yml` produces `data/poly.npz`.

The generators live in `adaptive_low_rank.synthetic`
(`generate_multiscale_dataset`, `ensure_synthetic_dataset`) and
`adaptive_low_rank.datasets` (`generate_synthetic_dataset`). A generated
dataset can be loaded with `load_dataset("poly")`.

#### Tight Frame Union Model

The **tight frame union** model generates union-of-subspaces datasets where each
subspace is populated by a tight frame. This creates structured datasets with
known energy hierarchies perfect for testing column selection algorithms.

```python
from adaptive_low_rank.synthetic import generate_multiscale_dataset

ds = generate_multiscale_dataset(
    model='tight_frame_union',
    n_clusters=5,            # 5 orthogonal subspaces
    columns_per_cluster=10,  # 10 frame vectors per subspace
    tube_dim=4,              # 4D subspaces
    decay_rate='polynomial', # Energy decay pattern
    seed=0
)
```

**Key parameters:**
- `n_clusters`: Number of tight frames (subspaces)
- `columns_per_cluster`: Vectors per frame
- `tube_dim`: Dimension of each subspace
- `high_energy`: Highest frame level (default: 1.0)
- `condition_number`: Ratio of highest to lowest frame level (default: 1e4)
- `decay_rate`: Energy decay pattern (`'linear'`, `'polynomial'`, or `'exponential'`)

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
`2000`, `2000`, and `0`. Generated matrices are saved as `data/<dataset>.npz`.

---

## Experiment Configuration

Each experiment consists of

- a dataset identifier
- optional experiment-wide parameters
- algorithm-specific parameter grids

Parameters that are lists are expanded into every combination automatically.
Algorithms other than Random raise an error if they select a column with
negligible residual energy rather than recording a no-op selection. This can
occur when the requested rank exceeds the matrix's numerical rank.

Greedy++ samples candidates uniformly without replacement within each
iteration, but can sample the same column again in later iterations. If the
winning candidate has negligible residual energy, the run raises an error
without resampling, even when usable columns remain outside the sampled batch.

Random Selection samples uniformly without replacement from all column indices,
including zero or dependent columns. Each residual is recomputed from the
original matrix using a reorthogonalized basis of the selected columns. A
selected column whose distance from that basis is at or below
`eps * max(X.shape) * ||X||_F` is treated as numerically dependent; such
selections still count toward `k`. Thus `k` cannot exceed the number of columns,
and the numerical span may have fewer than `k` directions.

Algorithm names used as keys under `algorithms:`:

| Key | Algorithm |
| --- | --- |
| `adaptive` | Adaptive Sampling |
| `batch_max` | Batch-Max |
| `greedy` | Greedy |
| `greedy_pp` | Greedy++ |
| `random` | Random Selection |

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
metadata.json
```

The runner refuses to overwrite an existing output directory. Move or archive
the previous directory before rerunning. `metadata.json` records the matrix
shape, dtype, and SHA-256 (over a JSON shape/dtype header and C-order matrix
bytes), Python and dependency versions, platform, Git revision, and worktree
dirty status. Archive the exact source changes separately when the worktree is
dirty. Preserve the output directory and input dataset alongside any published
results.
`results.pkl` uses Python pickle; only load files from trusted sources.

`summary.csv` contains one row per run with the algorithm, its parameters,
`final_residual`, and `runtime` when recorded.

The experiment runner does not generate plots. Use the plotting notebooks or
plotting helpers separately to create figures.

Runtime measurement is disabled by default. Set `compute_runtime: true` for
an algorithm run when runtime data is needed. `summary.csv` contains the
algorithm and run parameters, plus:

- `final_residual`
- `runtime`, if computed
- `final_gains_bm` and `final_gains_as`, if Batch-Max diagnostics are available

---

## Runtime Scaling

```bash
python scripts/runtime_scaling.py
```

benchmarks Adaptive, Batch-Max, and Greedy on synthetic matrices of
increasing size and saves `results/runtime_scaling/runtime_scaling.pkl`.
`notebooks/runtime_tables.ipynb` turns it into mean and standard-deviation
CSV tables under `results/runtime_scaling/tables/`.

Run a small validation sweep with `python scripts/runtime_scaling.py --smoke-test`.
The default `--threads=1` is verified and recorded in `environment.json`.
On Apple Accelerate, the benchmark uses its native per-thread single-threaded
mode and restores the previous mode afterward; other thread counts are not
supported there. For other BLAS backends, `threadpoolctl` must detect and
verify the requested thread count or the benchmark stops.
The runtime benchmark refuses to overwrite a nonempty output directory. Archive
older results in `results/runtime_scaling/` before running the current benchmark;
the notebook expects the current results schema.

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
Matplotlib, pandas, PyYAML, scikit-learn, and Pillow. Install the optional
notebook tools with `pip install -e ".[notebooks]"`.

Install the core package dependencies with

```bash
pip install -e .
```

Run the regression tests with:

```bash
python -m unittest discover -s tests -v
```

Continuous integration runs these tests on Python 3.10 through 3.14.

---

## Authors

Luke Stovall  
Kevin Miller

---

## License

This project is licensed under the MIT License. The project authors are listed
above; the copyright notice names Luke Stovall as the copyright holder. See
[LICENSE](./LICENSE) for the license terms.
