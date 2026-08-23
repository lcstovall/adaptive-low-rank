"""Run every experiment configuration in the configs directory."""

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"
RUNNER = Path(__file__).with_name("run_experiment.py")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run all YAML experiment configurations."
    )
    parser.add_argument(
        "--pattern",
        default=None,
        help="Config filename pattern (default: all .yml and .yaml files).",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Run remaining configurations after a failure.",
    )
    args = parser.parse_args()

    if args.pattern is None:
        configs = sorted(
            set(CONFIG_DIR.glob("*.yml")) | set(CONFIG_DIR.glob("*.yaml"))
        )
    else:
        configs = sorted(CONFIG_DIR.glob(args.pattern))
    if not configs:
        parser.error(f"No configuration files matched {args.pattern!r}")

    failures = []
    for config_path in configs:
        print(f"\n=== Running {config_path.name} ===", flush=True)
        completed = subprocess.run(
            [sys.executable, str(RUNNER), config_path.name],
            cwd=ROOT,
        )
        if completed.returncode == 0:
            print(f"=== Completed {config_path.name} ===", flush=True)
            continue

        failures.append(config_path.name)
        print(
            f"=== Failed {config_path.name} "
            f"(exit code {completed.returncode}) ===",
            flush=True,
        )
        if not args.continue_on_error:
            return completed.returncode

    if failures:
        print("\nFailed configurations:", flush=True)
        for config_name in failures:
            print(f"- {config_name}", flush=True)
        return 1

    print(f"\nCompleted {len(configs)} experiment(s).", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())