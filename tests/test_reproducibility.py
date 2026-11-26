import json
import runpy
import shutil
import tempfile
import unittest
from functools import partial
from pathlib import Path
from unittest.mock import patch

import numpy as np

from adaptive_low_rank.benchmark import experiment_metadata
from adaptive_low_rank.save_results import save_results
from adaptive_low_rank.synthetic import ensure_synthetic_dataset


class SyntheticCacheTests(unittest.TestCase):
    def test_cache_matches_generation_config_not_algorithm_config(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = {
                "synthetic": True,
                "dataset": "example",
                "decay_type": "poly",
                "decay_param": 1,
                "n": 4,
                "d": 4,
                "algorithms": {"greedy": {"k": 1}},
            }
            path = ensure_synthetic_dataset(config, data_dir=temp_dir)
            with np.load(path) as archive:
                original = archive["X"]
            config["algorithms"]["greedy"]["k"] = 2
            self.assertEqual(ensure_synthetic_dataset(config, data_dir=temp_dir), path)
            config["decay_param"] = 2
            with self.assertRaisesRegex(ValueError, "regenerate"):
                ensure_synthetic_dataset(config, data_dir=temp_dir)
            ensure_synthetic_dataset(config, force=True, data_dir=temp_dir)
            with np.load(path) as archive:
                self.assertFalse(np.allclose(original, archive["X"]))

    def test_legacy_cache_requires_regeneration(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "legacy.npz"
            np.savez_compressed(path, X=np.eye(2))
            config = {
                "synthetic": True,
                "dataset": "legacy",
                "decay_type": "poly",
                "decay_param": 1,
            }
            with self.assertRaisesRegex(ValueError, "regenerate"):
                ensure_synthetic_dataset(config, data_dir=temp_dir)

    def test_multiscale_cache_detects_changed_seed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = {
                "synthetic": True,
                "dataset": "frames",
                "model": "tight_frame_union",
                "n_clusters": 2,
                "columns_per_cluster": 2,
                "tube_dim": 2,
                "seed": 0,
            }
            ensure_synthetic_dataset(config, data_dir=temp_dir)
            self.assertIsNotNone(ensure_synthetic_dataset(config, data_dir=temp_dir))
            config["seed"] = 1
            with self.assertRaisesRegex(ValueError, "regenerate"):
                ensure_synthetic_dataset(config, data_dir=temp_dir)


class ExperimentResultsTests(unittest.TestCase):
    def test_existing_results_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = root / "config.yml"
            config.write_text("dataset: example\n")
            output = root / "results"
            output.mkdir()
            prior = output / "results.pkl"
            prior.write_bytes(b"existing results")
            with self.assertRaises(FileExistsError):
                save_results([], config, output)
            self.assertEqual(prior.read_bytes(), b"existing results")

    def test_metadata_identifies_matrix_and_is_saved(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = root / "config.yml"
            config.write_text("dataset: example\n")
            matrix = np.arange(6.0).reshape(2, 3)
            metadata = experiment_metadata(matrix, root)
            self.assertEqual(metadata["matrix"]["shape"], [2, 3])
            self.assertEqual(
                metadata["matrix"]["sha256"],
                experiment_metadata(matrix.copy(), root)["matrix"]["sha256"],
            )
            self.assertNotEqual(
                metadata["matrix"]["sha256"],
                experiment_metadata(matrix.T, root)["matrix"]["sha256"],
            )
            self.assertEqual(
                metadata["matrix"]["sha256"],
                experiment_metadata(np.asfortranarray(matrix), root)["matrix"]["sha256"],
            )
            save_results([], config, root / "results", metadata=metadata)
            saved = json.loads((root / "results" / "metadata.json").read_text())
            self.assertEqual(saved, metadata)

    def test_runner_records_metadata_and_refuses_a_second_run(self):
        source = Path(__file__).resolve().parents[1] / "scripts" / "run_experiment.py"
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "scripts").mkdir()
            (root / "configs").mkdir()
            script = root / "scripts" / "run_experiment.py"
            shutil.copyfile(source, script)
            (root / "configs" / "small.yml").write_text(
                "dataset: small\nsynthetic: true\ndecay_type: poly\n"
                "decay_param: 1\nn: 4\nd: 4\n"
                "algorithms:\n  greedy:\n    k: [1]\n"
            )
            with patch("sys.argv", [str(script), "small.yml"]), patch(
                "adaptive_low_rank.datasets.ensure_synthetic_dataset",
                wraps=partial(ensure_synthetic_dataset, data_dir=root / "data"),
            ) as ensure:
                runpy.run_path(str(script), run_name="__main__")
                ensure.assert_called_once()
                metadata = json.loads(
                    (root / "results" / "small" / "metadata.json").read_text()
                )
                self.assertEqual(metadata["matrix"]["shape"], [4, 4])
                self.assertTrue((root / "data" / "small.npz").is_file())
                ensure.reset_mock()
                with self.assertRaises(FileExistsError):
                    runpy.run_path(str(script), run_name="__main__")
                ensure.assert_not_called()


if __name__ == "__main__":
    unittest.main()
