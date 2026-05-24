"""Load datasets and Android reference embeddings."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"


def load_merged_three_npz(path: Path) -> Dict[str, Any]:
    """Load mocopi/meta_quest dict stored as pickled object in arr_0."""
    with np.load(path, allow_pickle=True) as archive:
        if "arr_0" in archive:
            return archive["arr_0"].item()
        return {key: archive[key] for key in archive.files}


def load_android_embeddings(path: Path) -> Dict[str, np.ndarray]:
    """Load Android ExecuTorch embeddings (requires numpy 2.x)."""
    with np.load(path, allow_pickle=True) as archive:
        return {key: archive[key] for key in archive.files}


def get_poses(db: Dict[str, Any]) -> np.ndarray:
    poses = np.asarray(db["poses"], dtype=np.float32)
    if poses.ndim != 4:
        raise ValueError(f"Expected poses [N,T,J,D], got {poses.shape}")
    return poses


def get_metadata(db: Dict[str, Any]) -> Dict[str, np.ndarray]:
    keys = ("pose2group", "pose2person", "pose_deleted", "labels", "persons")
    return {key: np.asarray(db[key]) for key in keys if key in db}
