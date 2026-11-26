import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import MagicMock, patch

from scripts import runtime_scaling


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


if __name__ == "__main__":
    unittest.main()
