#!/usr/bin/env python
"""Write qnn_executor_runner .raw inputs + input_list.txt from experiment dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
WINDOWS_SRC = REPO_ROOT / "windows" / "src"
if str(WINDOWS_SRC) not in sys.path:
    sys.path.insert(0, str(WINDOWS_SRC))

from configs import EXPERIMENTS  # noqa: E402
from data_io import get_poses, load_merged_three_npz  # noqa: E402
from model_utils import normalize_sequence, sequence_to_tensor  # noqa: E402


def pose_to_runner_raw(
    pose: np.ndarray,
    frames_num: int,
    joints_num: int,
    dim_num: int,
    normalize: bool = True,
) -> np.ndarray:
    seq = normalize_sequence(pose) if normalize else pose.astype(np.float32)
    tensor = sequence_to_tensor(
        seq, frames_num, joints_num, dim_num, torch.device("cpu"), precision="fp32"
    )
    return tensor.numpy().astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, choices=[e.name for e in EXPERIMENTS])
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory for .raw + input_list.txt")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--indices",
        type=str,
        default=None,
        help="Comma-separated pose indices only (e.g. 279,831,441 for legs worst)",
    )
    parser.add_argument("--no-normalize", action="store_true")
    args = parser.parse_args()

    cfg = next(e for e in EXPERIMENTS if e.name == args.experiment)
    poses = get_poses(load_merged_three_npz(cfg.data_path))

    if args.indices:
        indices = [int(x.strip()) for x in args.indices.split(",") if x.strip()]
    elif args.max_samples is not None:
        indices = list(range(min(len(poses), args.max_samples)))
    else:
        indices = list(range(len(poses)))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    for out_idx, pose_idx in enumerate(indices):
        raw = pose_to_runner_raw(
            poses[pose_idx],
            cfg.frames_num,
            cfg.joints_num,
            cfg.dim_num,
            normalize=not args.no_normalize,
        )
        name = f"input_{out_idx}_0.raw"
        raw.tofile(args.out_dir / name)
        lines.append(name)

    (args.out_dir / "input_list.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    index_map = args.out_dir / "input_index_map.txt"
    index_map.write_text(
        "\n".join(f"{out_idx} {pose_idx}" for out_idx, pose_idx in enumerate(indices)) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(indices)} inputs to {args.out_dir} (feature bytes={raw.nbytes})")
    print(f"Pose index map: {index_map}")


if __name__ == "__main__":
    main()
