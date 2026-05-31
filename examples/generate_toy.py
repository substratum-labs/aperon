#!/usr/bin/env python3
"""Generate a tiny Aperon HNTR/HNTQ dataset for CLI quickstarts."""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np


VERSION = 4


def write_matrix(path: Path, magic: bytes, matrix: np.ndarray) -> None:
    matrix = np.asarray(matrix, dtype="<f4")
    with path.open("wb") as f:
        f.write(magic)
        f.write(struct.pack("<III", VERSION, matrix.shape[0], matrix.shape[1]))
        f.write(matrix.tobytes(order="C"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("tmp/aperon-toy"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)

    centers = np.array(
        [
            [-4.0, -4.0, 0.0, 0.0],
            [4.0, -4.0, 0.0, 0.0],
            [-4.0, 4.0, 0.0, 0.0],
            [4.0, 4.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    vectors = np.repeat(centers, 16, axis=0)
    vectors += rng.normal(0.0, 0.18, size=vectors.shape).astype(np.float32)
    queries = centers + rng.normal(0.0, 0.08, size=centers.shape).astype(np.float32)

    write_matrix(args.out / "vectors.hntr", b"HNTR", vectors)
    write_matrix(args.out / "queries.hntq", b"HNTQ", queries)

    print(f"wrote {args.out / 'vectors.hntr'}")
    print(f"wrote {args.out / 'queries.hntq'}")


if __name__ == "__main__":
    main()
