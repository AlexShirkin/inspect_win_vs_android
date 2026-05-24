"""Standalone loader run with smart_avatar_dev (numpy 2.x)."""

import sys
from pathlib import Path

import numpy as np


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("Usage: load_android_npz.py INPUT.npz OUTPUT.npy")

    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])
    with np.load(src, allow_pickle=True) as archive:
        np.save(dst, archive["embeddings"])


if __name__ == "__main__":
    main()
