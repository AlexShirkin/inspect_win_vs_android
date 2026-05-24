#!/usr/bin/env python
"""Collect qnn_executor_runner output_*.raw files into embeddings.npz."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-dir", type=Path, required=True)
    parser.add_argument("--out-npz", type=Path, required=True)
    parser.add_argument("--embedding-dim", type=int, default=128)
    args = parser.parse_args()

    pattern = re.compile(r"^output_(\d+)_0\.raw$")
    indexed: list[tuple[int, Path]] = []
    for path in args.outputs_dir.iterdir():
        match = pattern.match(path.name)
        if match:
            indexed.append((int(match.group(1)), path))

    if not indexed:
        raise FileNotFoundError(f"No output_*_0.raw in {args.outputs_dir}")

    indexed.sort(key=lambda x: x[0])
    embeddings = np.stack(
        [
            np.fromfile(path, dtype=np.float32).reshape(args.embedding_dim)
            for _, path in indexed
        ],
        axis=0,
    )
    args.out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out_npz, embeddings=embeddings.astype(np.float32))
    print(f"Saved {embeddings.shape} -> {args.out_npz}")


if __name__ == "__main__":
    main()
