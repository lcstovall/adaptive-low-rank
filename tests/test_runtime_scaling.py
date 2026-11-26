import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from scripts import runtime_scaling
from adaptive_low_rank.plotting import runtime_table


class RuntimeScalingPathTests(unittest.TestCase):
    def test_default_output_path(self):
        script = Path(__file__).resolve().parents[1] / "scripts" / "runtime_scaling.py"
        with tempfile.TemporaryDirectory() as temp_dir:
            subprocess.run(
                [
                    sys.executable, str(script), "--n-values", "10",
                    "--dimensions", "10", "--fixed-d", "10", "--fixed-n", "10",
                    "--rank", "3", "--k", "2", "--repeats", "1",
                ],
                cwd=temp_dir,
                check=True,
                capture_output=True,
                text=True,
            )
            root = Path(temp_dir) / "results"
            self.assertTrue((root / "runtime_scaling" / "runtime_scaling.pkl").is_file())
            self.assertFalse((root / "runtime_scaling_v2").exists())
            environment = json.loads((root / "runtime_scaling" / "environment.json").read_text())
            control = environment["blas_thread_control"]
            self.assertEqual(control["requested_threads"], 1)
            self.assertIn(control["verification"], ("BLASGetThreading", "threadpoolctl"))
            if control["verification"] == "BLASGetThreading":
                self.assertEqual(control["effective_mode"], "single-threaded")
            else:
                self.assertTrue(control["threadpoolctl_blas_pools"])
                self.assertTrue(all(pool["num_threads"] == 1 for pool in control["threadpoolctl_blas_pools"]))


class RuntimeScalingThreadTests(unittest.TestCase):
    def test_accelerate_restores_mode_after_failure(self):
        library = MagicMock()
        library.BLASGetThreading.side_effect = [0, 1, 0]
        library.BLASSetThreading.return_value = 0
        config = {"Build Dependencies": {"blas": {"name": "accelerate"}}}
        with (
            patch.object(runtime_scaling.np, "show_config", return_value=config),
            patch.object(runtime_scaling.sys, "platform", "darwin"),
            patch.object(runtime_scaling, "threadpool_limits", return_value=nullcontext()),
            patch.object(runtime_scaling, "threadpool_info", return_value=[]),
            patch.object(runtime_scaling.ctypes, "CDLL", return_value=library),
        ):
            with self.assertRaisesRegex(ValueError, "benchmark failure"):
                with runtime_scaling.verified_blas_threads(1) as control:
                    self.assertEqual(control["effective_mode"], "single-threaded")
                    raise ValueError("benchmark failure")
            self.assertEqual(library.BLASSetThreading.call_args_list[0].args, (1,))
            self.assertEqual(library.BLASSetThreading.call_args_list[1].args, (0,))
            with self.assertRaisesRegex(ValueError, "only supports verified"):
                with runtime_scaling.verified_blas_threads(2):
                    self.fail("Unsupported thread count was accepted")

    def test_unverifiable_blas_pool_fails(self):
        config = {"Build Dependencies": {"blas": {"name": "openblas"}}}
        with (
            patch.object(runtime_scaling.np, "show_config", return_value=config),
            patch.object(runtime_scaling, "threadpool_limits", return_value=nullcontext()),
            patch.object(runtime_scaling, "threadpool_info", return_value=[]),
        ):
            with self.assertRaisesRegex(RuntimeError, "No BLAS thread pool detected"):
                with runtime_scaling.verified_blas_threads(1):
                    self.fail("Unverifiable thread count was accepted")

    def test_accelerate_mode_change_fails_and_restores(self):
        library = MagicMock()
        library.BLASGetThreading.side_effect = [0, 1, 0, 0]
        library.BLASSetThreading.return_value = 0
        config = {"Build Dependencies": {"blas": {"name": "accelerate"}}}
        with (
            patch.object(runtime_scaling.np, "show_config", return_value=config),
            patch.object(runtime_scaling.sys, "platform", "darwin"),
            patch.object(runtime_scaling, "threadpool_limits", return_value=nullcontext()),
            patch.object(runtime_scaling, "threadpool_info", return_value=[]),
            patch.object(runtime_scaling.ctypes, "CDLL", return_value=library),
        ):
            with self.assertRaisesRegex(RuntimeError, "changed during the benchmark"):
                with runtime_scaling.verified_blas_threads(1):
                    pass
            self.assertEqual(library.BLASSetThreading.call_args_list[1].args, (0,))

    def test_wrong_reported_thread_count_fails(self):
        config = {"Build Dependencies": {"blas": {"name": "openblas"}}}
        pools = [{"user_api": "blas", "num_threads": 2}]
        with (
            patch.object(runtime_scaling.np, "show_config", return_value=config),
            patch.object(runtime_scaling, "threadpool_limits", return_value=nullcontext()),
            patch.object(runtime_scaling, "threadpool_info", return_value=pools),
        ):
            with self.assertRaisesRegex(RuntimeError, "could not verify"):
                with runtime_scaling.verified_blas_threads(1):
                    self.fail("Wrong thread count was accepted")


class RuntimeTableTests(unittest.TestCase):
    def test_absolute_times_and_paired_ratios_for_both_sweeps(self):
        results = {
            "algorithms": ["Greedy", "Adaptive", "BatchMax"],
            "dimensions": [10, 20],
            "n_values": [100, 200],
            "runtimes": np.array([
                [[[5, 4], [15, 20]], [[6, 20], [12, 32]]],
                [[[1, 2], [3, 4]], [[2, 4], [4, 8]]],
                [[[4, 8], [6, 12]], [[4, 12], [12, 16]]],
            ], dtype=float),
        }
        columns = [
            "ARS time (s)", "BatchMax time (s)", "Greedy time (s)",
            "BatchMax/ARS", "Greedy/ARS",
        ]
        mean, std = runtime_table(results, fixed_d=10, include_absolute=True)
        self.assertEqual(list(mean.columns), columns)
        self.assertEqual(list(mean.index), [100, 200])
        np.testing.assert_allclose(mean.loc[100], [2, 5, 10, 3, 5])
        np.testing.assert_allclose(std.loc[100], [1, 1, 5, 1, 0])
        np.testing.assert_allclose(mean.loc[200], [3, 10, 12, 3.5, 3.5])
        np.testing.assert_allclose(std.loc[200], [1, 2, 8, 0.5, 1.5])

        mean, std = runtime_table(results, fixed_n=100, include_absolute=True)
        self.assertEqual(list(mean.columns), columns)
        self.assertEqual(list(mean.index), [10, 20])
        np.testing.assert_allclose(mean.loc[20], [3, 8, 9, 2.5, 3])
        np.testing.assert_allclose(std.loc[20], [1, 4, 3, 0.5, 0])

        legacy_mean, legacy_std = runtime_table(results, fixed_d=10)
        self.assertEqual(list(legacy_mean.columns), ["BatchMax", "Greedy"])
        np.testing.assert_allclose(legacy_mean.loc[100], [3, 5])
        np.testing.assert_allclose(legacy_std.loc[100], [1, 0])


if __name__ == "__main__":
    unittest.main()
