#!/usr/bin/env python3
"""Run matrix multiplication experiments for various sizes and collect timing data."""
import subprocess, pathlib, csv, sys

SIZES = [100, 200, 400, 600, 800, 1000, 1200]
REPEATS = 3
ROOT = pathlib.Path(__file__).resolve().parent.parent
BIN = ROOT / "matrix_mul"
DATA = ROOT / "data"
RESULTS = ROOT / "results"


def run(size):
    fa, fb = DATA / "matrix_a.txt", DATA / "matrix_b.txt"
    fc = DATA / "result.txt"
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/generate_matrix.py"), "-n", str(size), "-d", str(DATA)],
        check=True, capture_output=True,
    )
    out = subprocess.run(
        [str(BIN), str(fa), str(fb), str(fc)],
        check=True, capture_output=True, text=True,
    )
    # verify
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/verify.py"), str(fa), str(fb), str(fc)],
        check=True, capture_output=True,
    )
    parts = out.stdout.strip().split(",")
    return {"n": int(parts[0]), "flops": int(parts[1]),
            "time_sec": float(parts[2]), "mflops": float(parts[3])}


def main():
    RESULTS.mkdir(exist_ok=True)
    csv_path = RESULTS / "experiments.csv"
    rows = []
    for n in SIZES:
        for trial in range(1, REPEATS + 1):
            print(f"n={n}  trial {trial}/{REPEATS} ... ", end="", flush=True)
            r = run(n)
            r["trial"] = trial
            rows.append(r)
            print(f"{r['time_sec']:.4f}s  {r['mflops']:.1f} MFLOPS")

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["n", "trial", "flops", "time_sec", "mflops"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved: {csv_path}")


if __name__ == "__main__":
    main()
