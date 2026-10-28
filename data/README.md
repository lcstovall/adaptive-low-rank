## Datasets

The experiment configurations use the dataset identifiers below. Each loader
returns a NumPy matrix. `mnist`, `cifar10`, and `yearprediction` put features
in rows and samples in columns. `coil20` returns one flattened image per row.
The other loaders retain the orientation of their source or generated matrix.

- `interactions` loads the `B` matrix from the MATLAB file.
  - Local input: `data/interactions.mat`
- `mnist` fetches `mnist_784` from OpenML with scikit-learn and transposes
  the data. The data may be cached locally by scikit-learn.
- `cifar10` loads the local CIFAR-10 feature archive and transposes the data.
  - Local input: `data/cifar10_simclr.npz`
- `coil20` reads every PNG in sorted order, flattens each image, and returns
  one image per row.
  - Local input: `data/coil-20-proc/`
- `yearprediction` reads the comma-separated file, removes the first column
  (the target), and transposes the feature matrix.
  - Local input: `data/YearPredictionMSD.txt`
- Any config with `synthetic: true` (e.g. `poly`) generates the matrix `X`
  from the config if missing, then loads it from a compressed NumPy archive.
  - Local input: `data/<identifier>.npz`

Only `data/interactions.mat` is tracked as a dataset input. The other local
inputs above are not included in a clean clone.

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

Keep the filenames and directory layout shown above for local inputs.
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