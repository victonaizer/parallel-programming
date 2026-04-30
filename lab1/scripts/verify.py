#!/usr/bin/env python3
import sys, numpy as np


def load(path):
    with open(path) as f:
        n = int(f.readline())
        rows = [list(map(int, f.readline().split())) for _ in range(n)]
    return np.array(rows, dtype=np.int64)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(f"Usage: {sys.argv[0]} A B C")
    a, b, c = (load(p) for p in sys.argv[1:])
    expected = a @ b
    if np.array_equal(c, expected):
        print(f"OK  {a.shape[0]}x{a.shape[0]}")
    else:
        d = np.abs(c - expected)
        print(f"FAIL  max_diff={d.max()}  errors={np.count_nonzero(d)}/{c.size}")
        sys.exit(1)
