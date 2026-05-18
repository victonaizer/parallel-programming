#!/usr/bin/env python3
"""Run cluster-style MPI matrix multiplication experiments."""
import argparse
import csv
import os
import pathlib
import subprocess
import sys

DEFAULT_SIZES = [200, 400, 800, 1200, 1600, 2000]
DEFAULT_PROCS = [1, 2, 4, 8, 16, 32, 40]
ROOT = pathlib.Path(__file__).resolve().parent.parent
BIN = ROOT / "matrix_mul_mpi_super"
DATA = ROOT / "data"
RESULTS = ROOT / "results"


def parse_int_list(value):
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def run(size, nprocs, oversubscribe):
    fa, fb = DATA / "matrix_a.txt", DATA / "matrix_b.txt"
    fc = DATA / "result.txt"
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/generate_matrix.py"), "-n", str(size), "-d", str(DATA)],
        check=True,
        capture_output=True,
    )
    cmd = ["mpirun"]
    if oversubscribe:
        cmd.append("--oversubscribe")
    cmd += ["-np", str(nprocs), str(BIN), str(fa), str(fb), str(fc)]
    out = subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/verify.py"), str(fa), str(fb), str(fc)],
        check=True,
        capture_output=True,
    )
    parts = out.stdout.strip().split(",")
    return {
        "n": int(parts[0]),
        "procs": int(parts[1]),
        "flops": int(parts[2]),
        "time_sec": float(parts[3]),
        "mflops": float(parts[4]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default=",".join(map(str, DEFAULT_SIZES)))
    ap.add_argument("--procs", default=",".join(map(str, DEFAULT_PROCS)))
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--output", default="experiments.csv")
    ap.add_argument("--oversubscribe", action="store_true", default=bool(os.getenv("MPI_OVERSUBSCRIBE")))
    args = ap.parse_args()

    sizes = parse_int_list(args.sizes)
    procs = parse_int_list(args.procs)

    RESULTS.mkdir(exist_ok=True)
    csv_path = RESULTS / args.output
    rows = []
    for n in sizes:
        for p in procs:
            for trial in range(1, args.repeats + 1):
                print(f"n={n}  procs={p}  trial {trial}/{args.repeats} ... ", end="", flush=True)
                r = run(n, p, args.oversubscribe)
                r["trial"] = trial
                rows.append(r)
                print(f"{r['time_sec']:.4f}s  {r['mflops']:.1f} MFLOPS")

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["n", "procs", "trial", "flops", "time_sec", "mflops"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved: {csv_path}")


if __name__ == "__main__":
    main()
