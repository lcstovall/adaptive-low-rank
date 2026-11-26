"""Paired, fixed-work runtime sweeps for adaptive-low-rank.

Run from the repository root after installing the package and threadpoolctl.
Uses the existing select_index/_project_out kernels with a timing-only driver;
the package's general-purpose select_columns method is not changed.
"""
from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import inspect
import json
import os
import pickle
import platform
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

import adaptive_low_rank.algorithms as algorithm_module
from adaptive_low_rank.algorithms import Adaptive, BatchMax, Greedy


ALGORITHMS = {"Adaptive": Adaptive, "BatchMax": BatchMax, "Greedy": Greedy}
REFERENCE_COMMIT = "13ce70703ba24e26aef016670d1181c30acc39bc"


def generate_svd_dataset(n, d, rank=200, random_state=0):
    """Rank-r matrix with orthonormal factors and ||X||_F^2 = n.

    Singular-value *shape* is linear from 1 to 0.1; a common scale factor
    enforces unit average squared column norm. Generation is never timed.
    """
    if not 1 <= rank <= min(d, n):
        raise ValueError("Require 1 <= rank <= min(d, n).")
    rng = np.random.RandomState(random_state)
    U, _ = np.linalg.qr(rng.standard_normal((d, rank)), mode="reduced")
    V, _ = np.linalg.qr(rng.standard_normal((n, rank)), mode="reduced")
    weights = np.linspace(1.0, 0.1, rank)
    weights *= np.sqrt(n) / np.linalg.norm(weights)
    return (U * weights) @ V.T


def seed_for(base, d, n, trial, stream):
    """Stable independent streams for data, selection, and method order."""
    return int(np.random.SeedSequence([base, d, n, trial, stream]).generate_state(1)[0])


@dataclass
class RunResult:
    seconds: float
    completed_steps: int
    indices: list[int]
    final_residual_norm: float
    selected_rank: int
    smallest_selected_singular_value: float
    rank_tolerance: float
    minimum_pivot_energy: float
    stop_reason: str | None

    def valid(self, k):
        return (self.stop_reason is None and self.completed_steps == k
                and self.selected_rank == k and len(set(self.indices)) == k
                and np.isfinite(self.final_residual_norm))


def time_selection(algorithm, X, k, batch_size, random_state,
                   mean_column_energy, pivot_rtol=1e-12):
    """Time copy + RNG setup + selection/scoring + real projection updates.

    Do not compute residual histories, alpha, or SVD inside the timer.
    A selected direction below tolerance stops the run; it is never replaced
    by an idle iteration or a silently resampled pivot. Incomplete runs fail
    the benchmark rather than entering the timing summaries.
    """
    start = time.perf_counter()
    R = np.array(X, dtype=np.float64, order="C", copy=True)
    rng = np.random.RandomState(random_state)
    indices = []
    seen = set()
    minimum_energy = float("inf")
    reason = None
    for _ in range(k):
        idx, _ = algorithm.select_index(
            R, k, rng, n_candidates=batch_size, V=None, compute_alpha=False
        )
        idx = int(idx)
        if not 0 <= idx < R.shape[1]:
            reason = "invalid selected index"
            break
        if idx in seen:
            reason = "repeated selected index"
            break
        column = R[:, idx]
        energy = float(column @ column)
        if (not np.isfinite(energy) or energy == 0
                or energy <= pivot_rtol * mean_column_energy):
            reason = "selected residual direction is below projection tolerance"
            break
        updated = algorithm._project_out(R, column)
        if updated is R:
            reason = "projection kernel skipped the update"
            break
        R = updated
        indices.append(idx)
        seen.add(idx)
        minimum_energy = min(minimum_energy, energy)
    elapsed = time.perf_counter() - start

    # Validation and the single final residual diagnostic are outside timing.
    residual_norm = float(np.linalg.norm(R, "fro"))
    del R
    singular_values = np.linalg.svd(X[:, indices], compute_uv=False)
    threshold = (np.finfo(float).eps * max(X.shape[0], len(indices))
                 * singular_values[0]) if indices else 0.0
    numerical_rank = int(np.count_nonzero(singular_values > threshold))
    return RunResult(
        seconds=elapsed, completed_steps=len(indices), indices=indices,
        final_residual_norm=residual_norm, selected_rank=numerical_rank,
        smallest_selected_singular_value=float(singular_values[-1]) if indices else 0.0,
        rank_tolerance=float(threshold),
        minimum_pivot_energy=minimum_energy if indices else 0.0,
        stop_reason=reason,
    )


def atomic_pickle(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


def sweep_results(measured, dimensions, n_values, repeats):
    """Preserve the original runtime_table helper's array convention."""
    times = np.empty((len(ALGORITHMS), len(dimensions), repeats, len(n_values)))
    for ai, name in enumerate(ALGORITHMS):
        for di, d in enumerate(dimensions):
            for ni, n in enumerate(n_values):
                times[ai, di, :, ni] = measured[(d, n)][name]
    return dict(algorithms=list(ALGORITHMS), dimensions=list(dimensions),
                n_values=list(n_values), runtimes=times, mean_runtimes=times.mean(axis=2))


def write_summary(path, sweep, varying):
    """Mean/std of absolute seconds and paired trial-wise runtime ratios.

    ddof=0 matches adaptive_low_rank.plotting.runtime_table.
    """
    fields = [varying]
    for name in ALGORITHMS:
        fields += [name + "_seconds_mean", name + "_seconds_std"]
    for name in ("BatchMax", "Greedy"):
        fields += [name + "_over_ARS_mean", name + "_over_ARS_std"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        sizes = sweep["n_values"] if varying == "n" else sweep["dimensions"]
        for i, size in enumerate(sizes):
            samples = (sweep["runtimes"][:, 0, :, i] if varying == "n"
                       else sweep["runtimes"][:, i, :, 0])
            row = {varying: size}
            for ai, name in enumerate(ALGORITHMS):
                row[name + "_seconds_mean"] = float(samples[ai].mean())
                row[name + "_seconds_std"] = float(samples[ai].std(ddof=0))
            for ai, name in enumerate(ALGORITHMS):
                if name == "Adaptive":
                    continue
                ratios = samples[ai] / samples[0]
                row[name + "_over_ARS_mean"] = float(ratios.mean())
                row[name + "_over_ARS_std"] = float(ratios.std(ddof=0))
            writer.writerow(row)


@contextmanager
def verified_blas_threads(threads):
    blas_name = np.show_config(mode="dicts")["Build Dependencies"]["blas"]["name"]
    accelerate = sys.platform == "darwin" and blas_name.lower() == "accelerate"
    if accelerate and threads != 1:
        raise ValueError("Apple Accelerate only supports verified --threads=1.")

    with threadpool_limits(limits=threads, user_api="blas"):
        pools = [pool for pool in threadpool_info() if pool["user_api"] == "blas"]
        if any(pool["num_threads"] != threads for pool in pools):
            raise RuntimeError("threadpoolctl could not verify the requested BLAS thread count.")
        if not accelerate and not pools:
            raise RuntimeError(f"No BLAS thread pool detected for NumPy backend {blas_name!r}.")

        control = dict(
            numpy_blas=blas_name, requested_threads=threads,
            threadpoolctl_blas_pools=pools,
            verification="threadpoolctl",
        )
        if not accelerate:
            yield control
            return

        try:
            library = ctypes.CDLL("/System/Library/Frameworks/Accelerate.framework/Accelerate")
            get_mode = library.BLASGetThreading
            set_mode = library.BLASSetThreading
        except (OSError, AttributeError) as exc:
            raise RuntimeError("Apple Accelerate's thread-control API is unavailable.") from exc
        get_mode.argtypes = []
        get_mode.restype = ctypes.c_uint
        set_mode.argtypes = [ctypes.c_uint]
        set_mode.restype = ctypes.c_int
        previous_mode = get_mode()
        if previous_mode not in (0, 1):
            raise RuntimeError(f"Unknown Apple Accelerate threading mode: {previous_mode}.")
        try:
            if set_mode(1) != 0 or get_mode() != 1:
                raise RuntimeError("Could not verify single-threaded Apple Accelerate.")
            control.update(verification="BLASGetThreading", effective_mode="single-threaded")
            yield control
            if get_mode() != 1:
                raise RuntimeError("Apple Accelerate threading mode changed during the benchmark.")
        finally:
            if set_mode(previous_mode) != 0 or get_mode() != previous_mode:
                raise RuntimeError("Could not restore the Apple Accelerate threading mode.")


def environment(args, thread_control):
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], check=True,
                                capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], check=True,
                                    capture_output=True, text=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    source = inspect.getsource(algorithm_module).encode()
    return dict(
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        python=sys.version, numpy=np.__version__, platform=platform.platform(),
        machine=platform.machine(), processor=platform.processor(), cpu_count=os.cpu_count(),
        threadpools=threadpool_info(), blas_thread_control=thread_control,
        git_commit=commit, git_worktree_dirty=dirty,
        algorithm_source_sha256=hashlib.sha256(source).hexdigest(),
        reviewed_reference_commit=REFERENCE_COMMIT, std_ddof=0,
        timing_includes="residual copy, RNG setup, pivot selection/scoring, projection updates, pivot checks",
        timing_excludes="generation, warmup, final residual norm, selected-column SVD, file I/O",
    )


def run_benchmark(args):
    if args.smoke_test:
        args.n_values, args.dimensions = [80, 120], [40, 64]
        args.fixed_d, args.fixed_n, args.rank, args.k, args.repeats = 64, 80, 30, 10, 2
    if args.output_dir is None:
        args.output_dir = Path("results/runtime_scaling_smoke" if args.smoke_test
                               else "results/runtime_scaling")
    if min(args.repeats, args.threads, args.batch_size) < 1 or not 0 < args.k < args.rank:
        raise ValueError("Require repeats/threads/batch-size >= 1 and 0 < k < rank.")
    if not 0 <= args.seed < 2**32 or not np.isfinite(args.pivot_rtol) or args.pivot_rtol <= 0:
        raise ValueError("Require a nonnegative 32-bit seed and finite positive pivot-rtol.")
    if len(set(args.n_values)) != len(args.n_values) or len(set(args.dimensions)) != len(args.dimensions):
        raise ValueError("Dimension lists must not contain duplicates.")
    # Two sweeps, not their Cartesian product; measure the intersection once.
    configs = list(dict.fromkeys([(args.fixed_d, n) for n in args.n_values]
                                + [(d, args.fixed_n) for d in args.dimensions]))
    if any(min(d, n) < args.rank for d, n in configs):
        raise ValueError("Every configuration must have d,n >= rank.")
    out = args.output_dir
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} is not empty; choose a new --output-dir.")
    out.mkdir(parents=True, exist_ok=True)
    measured, records = {}, []
    with verified_blas_threads(args.threads) as thread_control:
        (out / "environment.json").write_text(json.dumps(environment(args, thread_control), indent=2) + "\n")
        script_source = Path(__file__).read_text()
        (out / "runtime_scaling_used.py").write_text(script_source)
        for d, n in configs:
            measured[(d, n)] = {name: [] for name in ALGORITHMS}
            for trial in range(args.repeats):
                data_seed = seed_for(args.seed, d, n, trial, 0)
                algorithm_seed = seed_for(args.seed, d, n, trial, 1)
                order_seed = seed_for(args.seed, d, n, trial, 2)
                X = generate_svd_dataset(n, d, args.rank, data_seed)
                mean_energy = float(np.einsum("ij,ij->", X, X)) / n
                if trial == 0:
                    # One excluded full-work warm-up per method/configuration.
                    for name in ALGORITHMS:
                        warm = time_selection(ALGORITHMS[name](), X, args.k, args.batch_size,
                                              algorithm_seed, mean_energy, args.pivot_rtol)
                        if not warm.valid(args.k):
                            (out / "failure.json").write_text(json.dumps(
                                dict(phase="warmup", d=d, n=n, method=name,
                                     data_seed=data_seed, algorithm_seed=algorithm_seed,
                                     result=asdict(warm)), indent=2))
                            raise RuntimeError(f"Warm-up failed validation for {name}, d={d}, n={n}.")
                order = np.random.RandomState(order_seed).permutation(list(ALGORITHMS)).tolist()
                for position, name in enumerate(order):
                    result = time_selection(ALGORITHMS[name](), X, args.k, args.batch_size,
                                            algorithm_seed, mean_energy, args.pivot_rtol)
                    record = dict(d=d, n=n, trial=trial, method=name,
                                  data_seed=data_seed, algorithm_seed=algorithm_seed,
                                  order_seed=order_seed, order_position=position,
                                  valid=result.valid(args.k), **asdict(result))
                    records.append(record)
                    # Preserve completed samples even if a later run fails.
                    atomic_pickle(out / "checkpoint.pkl", dict(records=records))
                    if not record["valid"]:
                        (out / "failure.json").write_text(json.dumps(record, indent=2))
                        raise RuntimeError(f"Invalid run: {name}, d={d}, n={n}; see failure.json.")
                    measured[(d, n)][name].append(result.seconds)
                    print(f"d={d:5d} n={n:7d} trial={trial+1:2d}/{args.repeats} "
                          f"{name:8s} {result.seconds:.4f}s ({result.completed_steps} updates)", flush=True)
                del X
    columns = sweep_results(measured, [args.fixed_d], args.n_values, args.repeats)
    rows = sweep_results(measured, args.dimensions, [args.fixed_n], args.repeats)
    for name, sweep, varying in [("column_scaling", columns, "n"), ("row_scaling", rows, "d")]:
        atomic_pickle(out / (name + ".pkl"), sweep)
        write_summary(out / (name + "_summary.csv"), sweep, varying)
    atomic_pickle(out / "runtime_scaling.pkl", dict(
        schema_version=2, smoke_test=args.smoke_test, column_sweep=columns,
        row_sweep=rows, records=records))
    with (out / "raw_trials.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        for record in records:
            writer.writerow({**record, "indices": json.dumps(record["indices"])})
    print(f"Validated results and tables saved under {out}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-values", nargs="+", type=int, default=[1000, 5000, 10000, 50000, 100000])
    parser.add_argument("--dimensions", nargs="+", type=int, default=[250, 500, 1000, 2000, 4000])
    parser.add_argument("--fixed-d", type=int, default=1000)
    parser.add_argument("--fixed-n", type=int, default=10000)
    parser.add_argument("--rank", type=int, default=200)
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pivot-rtol", type=float, default=1e-12)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--smoke-test", action="store_true", help="Use a tiny grid, rank=30, k=10, repeats=2.")
    return parser.parse_args()


if __name__ == "__main__":
    run_benchmark(parse_args())