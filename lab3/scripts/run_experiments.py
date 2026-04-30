#!/usr/bin/env python3
"""Run MPI matrix multiplication experiments across sizes and process counts."""
import subprocess, pathlib, csv, sys

SIZES = [200, 400, 800, 1200, 1600, 2000]
PROCS = [1, 2, 4, 8]
REPEATS = 3
ROOT = pathlib.Path(__file__).resolve().parent.parent
BIN = ROOT / "matrix_mul_mpi"
DATA = ROOT / "data"
RESULTS = ROOT / "results"


def run(size, nprocs):
    fa, fb = DATA / "matrix_a.txt", DATA / "matrix_b.txt"
    fc = DATA / "result.txt"
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/generate_matrix.py"), "-n", str(size), "-d", str(DATA)],
        check=True, capture_output=True,
    )
    out = subprocess.run(
        ["mpirun", "--oversubscribe", "-np", str(nprocs), str(BIN), str(fa), str(fb), str(fc)],
        check=True, capture_output=True, text=True,
    )
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/verify.py"), str(fa), str(fb), str(fc)],
        check=True, capture_output=True,
    )
    parts = out.stdout.strip().split(",")
    return {"n": int(parts[0]), "procs": int(parts[1]), "flops": int(parts[2]),
            "time_sec": float(parts[3]), "mflops": float(parts[4])}


def main():
    RESULTS.mkdir(exist_ok=True)
    csv_path = RESULTS / "experiments.csv"
    rows = []
    for n in SIZES:
        for p in PROCS:
            for trial in range(1, REPEATS + 1):
                print(f"n={n}  procs={p}  trial {trial}/{REPEATS} ... ", end="", flush=True)
                r = run(n, p)
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
