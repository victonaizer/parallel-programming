#!/usr/bin/env python3
import argparse, random, pathlib


def gen(n, lo, hi):
    return [[random.randint(lo, hi) for _ in range(n)] for _ in range(n)]


def dump(matrix, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(f"{len(matrix)}\n")
        f.writelines(" ".join(map(str, row)) + "\n" for row in matrix)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=500)
    ap.add_argument("-d", "--dir", type=str, default="data")
    ap.add_argument("--lo", type=int, default=-50)
    ap.add_argument("--hi", type=int, default=50)
    args = ap.parse_args()

    out = pathlib.Path(args.dir)
    dump(gen(args.n, args.lo, args.hi), out / "matrix_a.txt")
    dump(gen(args.n, args.lo, args.hi), out / "matrix_b.txt")
    print(f"OK: {args.n}x{args.n} -> {out}")
