"""Experiment definitions for Windows PyTorch runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
WEIGHTS_DIR = REPO_ROOT / "weights"
RESULTS_DIR = REPO_ROOT / "results" / "windows"


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    data_path: Path
    weights_path: Path
    frames_num: int
    joints_num: int
    dim_num: int
    android_embeddings_path: Optional[Path] = None
    task: str = "boxing"  # boxing | avatar_library
    description: str = ""


EXPERIMENTS: List[ExperimentConfig] = [
    ExperimentConfig(
        name="mocopi_body",
        data_path=DATA_DIR / "mocopi_body_db_30_12_2d_proj_merged_three.npz",
        weights_path=WEIGHTS_DIR / "sequential_250_128_mocopi_body_2d_proj_30_12_77999.pth",
        frames_num=30,
        joints_num=12,
        dim_num=2,
        android_embeddings_path=DATA_DIR / "android_upper_embeddings_fixed.npz",
        task="boxing",
        description="Boxing upper body (2D projection, 12 joints)",
    ),
    ExperimentConfig(
        name="mocopi_legs",
        data_path=DATA_DIR / "mocopi_legs_db_30_11_2d_proj_merged_three.npz",
        weights_path=WEIGHTS_DIR / "sequential_250_128_mocopi_legs_2d_proj_30_11_15999.pth",
        frames_num=30,
        joints_num=11,
        dim_num=2,
        android_embeddings_path=DATA_DIR / "android_lower_embeddings_fixed.npz",
        task="boxing",
        description="Boxing lower body (2D projection, 11 joints)",
    ),
    ExperimentConfig(
        name="meta_quest_avatar",
        data_path=DATA_DIR / "meta_quest_db_one_phase_41_corrected_labels.npz",
        weights_path=WEIGHTS_DIR / "sequential_250_128_meta_quest_30_13_mobile_refactor_63499.pth",
        frames_num=30,
        joints_num=13,
        dim_num=3,
        android_embeddings_path=None,
        task="avatar_library",
        description="Avatar motion library — одна модель (13 joints 3D), без отдельной legs-модели",
    ),
]
