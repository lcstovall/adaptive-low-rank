## Datasets

The experiment configurations use the dataset identifiers below. Each loader
returns a NumPy matrix; where noted, samples are transposed into columns.
The transposed datasets have features in rows and samples in columns; the other datasets retain the orientation of their source file.

| Identifier | Source and loading behavior | Local input |
| --- | --- | --- |
| `interactions` | Loads the `B` matrix from the MATLAB file. | `data/interactions.mat` |
| `mnistT` | Fetches `mnist_784` from OpenML with scikit-learn and transposes the data. | OpenML access; the data may be cached locally by scikit-learn. |
| `cfar10T` | Loads CIFAR-10 through GraphLearning using the `simclr` metric, then transposes the data. | GraphLearning's CIFAR-10 data/cache. |
| `coil20` | Reads every PNG in sorted order, flattens each image, and transposes the resulting matrix. | `data/coil-20-proc/` |
| `yearprediction` | Reads the comma-separated file, removes the first column (the target), and transposes the feature matrix. | `data/YearPredictionMSD.txt` |
| `cluster_expansion` | Loads the matrix and randomly selects 5,000 rows without replacement. | `data/cluster_expansion_M.npy` |
| `exp005`, `exp01`, `exp1`, `poly2`, `poly3` | Loads the generated matrix `X` from a compressed NumPy archive. | `data/<identifier>.npz` |

The `.mat`, `.npy`, `.npz`, and COIL-20 files needed by the repository are
already included. The external datasets used by the loaders are MNIST via
OpenML, CIFAR-10 via GraphLearning, COIL-20, and YearPredictionMSD.

### Data availability

The following inputs are included in this repository:

- `interactions.mat`
- `cluster_expansion_M.npy`
- `coil-20-proc/`
- `YearPredictionMSD.txt`
- The generated synthetic archives (`exp*.npz` and `poly*.npz`)

MNIST is fetched from [OpenML](https://www.openml.org/d/554) when
`mnistT` is loaded. CIFAR-10 is fetched or loaded from GraphLearning's data
source and cache when `cfar10T` is loaded. Both loaders therefore may require
network access the first time they run. The local CIFAR-10 `.npz` files are
supporting data files and are not opened directly by `datasets.py`.

For the datasets included in the repository, retain the filenames and
directory layout shown in the table. For the external datasets, follow their
respective terms and attribution requirements:

- [COIL-20](http://www.cs.columbia.edu/CAVE/software/softlib/coil-20.php)
- [YearPredictionMSD](https://archive.ics.uci.edu/dataset/203/yearpredictionmsd)
- [MNIST on OpenML](https://www.openml.org/d/554)
- [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)

### Reproducibility notes

- `cluster_expansion` draws a new random sample of 5,000 rows on each load;
	its result is not deterministic because the loader does not set a seed.
- Synthetic matrices are generated from the matching YAML configuration using
	polynomial or exponential singular-value decay and normalized to unit
	Frobenius norm.
- The loader removes the YearPredictionMSD target column before returning the
	feature matrix. It does not normalize the real-valued datasets.

### Generating synthetic datasets

The synthetic archives are described by the matching files in `configs/` and
can be generated or regenerated from the repository root with:

```bash
```bash
python3 scripts/generate_synthetic_datasets.py
```

By default, existing archives are kept. Use `--force` to regenerate them.