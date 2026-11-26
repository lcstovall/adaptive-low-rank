import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


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


if __name__ == "__main__":
    unittest.main()
