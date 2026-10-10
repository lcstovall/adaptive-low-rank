## Datasets

The experiment configurations use the dataset identifiers below. Each loader
returns a NumPy matrix. `mnist`, `cifar10`, and `yearprediction` put features
in rows and samples in columns. `coil20` returns one flattened image per row.
The other loaders retain the orientation of their source or generated matrix.

| Identifier | Source and loading behavior | Local input |
| --- | --- | --- |
| `interactions` | Loads the `B` matrix from the MATLAB file. | `data/interactions.mat` |
| `mnist` | Fetches `mnist_784` from OpenML with scikit-learn and transposes the data. | OpenML access; the data may be cached locally by scikit-learn. |
| `cifar10` | Loads the local CIFAR-10 feature archive and transposes the data. | `data/cifar10_simclr.npz` |
| `coil20` | Reads every PNG in sorted order, flattens each image, and returns one image per row. | `data/coil-20-proc/` |
| `yearprediction` | Reads the comma-separated file, removes the first column (the target), and transposes the feature matrix. | `data/YearPredictionMSD.txt` |
| Any config with `synthetic: true` (e.g. `exp`, `poly`) | Generates the matrix `X` from the config if missing, then loads it from a compressed NumPy archive. | `data/<identifier>.npz` |

Only `data/interactions.mat` is tracked as a dataset input. The other local
inputs in the table are not included in a clean clone.

### Data availability

The following dataset assets are tracked in Git:

- `interactions.mat`

These loader-required inputs are not tracked and must be obtained or created
separately before running the corresponding experiments:

- `data/cifar10_simclr.npz` for `cifar10`
- `data/coil-20-proc/` containing the COIL-20 image files for `coil20`
- `data/YearPredictionMSD.txt` for `yearprediction`

The `mnist` loader fetches `mnist_784` from
[OpenML](https://www.openml.org/d/554), so its first run may require network
access. Synthetic datasets are generated from their experiment
configurations when needed; their `.npz` archives are not committed.

Keep the filenames and directory layout shown in the table for local inputs.
Follow the respective terms and attribution requirements when obtaining
external datasets:

- [COIL-20](http://www.cs.columbia.edu/CAVE/software/softlib/coil-20.php)
- [YearPredictionMSD](https://archive.ics.uci.edu/dataset/203/yearpredictionmsd)
- [MNIST on OpenML](https://www.openml.org/d/554)
- [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)

### Reproducibility notes

- Synthetic matrices are generated from the matching YAML configuration using
	polynomial or exponential singular-value decay and normalized to unit
	Frobenius norm.
- The loader removes the YearPredictionMSD target column before returning the
	feature matrix. It does not normalize the real-valued datasets.

### Generating synthetic datasets

The synthetic archives are described by the matching files in `configs/` and
are generated automatically the first time an experiment using them runs. To
regenerate one, delete its `.npz` file.