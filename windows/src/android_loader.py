"""Load Android npz files saved with numpy 2.x (requires smart_avatar_dev)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np


def _find_dev_python() -> Path:
    candidates = [
        Path(r"C:\Users\Alex\miniconda3\envs\smart_avatar_dev\python.exe"),
        Path.home() / "miniconda3" / "envs" / "smart_avatar_dev" / "python.exe",
        Path.home() / "anaconda3" / "envs" / "smart_avatar_dev" / "python.exe",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "smart_avatar_dev env not found; needed to read numpy-2.x Android npz"
    )


def load_android_embeddings(path: Path) -> np.ndarray:
    """Load embeddings array from Android npz via numpy-2.x interpreter."""
    loader = Path(__file__).resolve().parent / "load_android_npz.py"
    dev_python = _find_dev_python()
    tmp = path.with_name(path.stem + "._android_tmp.npy")

    proc = subprocess.run(
        [str(dev_python), str(loader), str(path), str(tmp)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Failed to load Android npz via {dev_python}:\n{proc.stderr or proc.stdout}"
        )

    embeddings = np.load(tmp).astype(np.float32)
    tmp.unlink(missing_ok=True)
    return embeddings
