#!/usr/bin/env python
"""
Run FP32 / FP16-simulated PyTorch embedding experiments on Windows.

Usage (conda env smart_avatar):
  python windows/run_all.py
  python windows/run_all.py --experiment mocopi_body
  python windows/run_all.py --skip-fp32   # reuse saved fp32 embeddings
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

WINDOWS_DIR = Path(__file__).resolve().parent
SRC_DIR = WINDOWS_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from configs import EXPERIMENTS, ExperimentConfig, RESULTS_DIR  # noqa: E402
from data_io import get_metadata, get_poses, load_merged_three_npz  # noqa: E402
from metrics import compare_embeddings, format_report  # noqa: E402
from model_utils import embed_batch  # noqa: E402


def save_embeddings(path: Path, embeddings: np.ndarray, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, embeddings=embeddings, **meta)


def load_embeddings(path: Path) -> np.ndarray:
    with np.load(path) as archive:
        return archive["embeddings"]


from android_loader import load_android_embeddings  # noqa: E402


def run_experiment(
    cfg: ExperimentConfig,
    device: torch.device,
    skip_fp32: bool = False,
    skip_fp16: bool = False,
) -> dict:
    print(f"\n{'=' * 72}\nExperiment: {cfg.name}\n{cfg.description}\n{'=' * 72}")

    out_dir = RESULTS_DIR / cfg.name
    out_dir.mkdir(parents=True, exist_ok=True)

    db = load_merged_three_npz(cfg.data_path)
    poses = get_poses(db)
    meta = get_metadata(db)
    groups = meta.get("pose2group")

    print(f"Loaded {len(poses)} poses, shape={poses.shape}")

    fp32_path = out_dir / "embeddings_fp32.npz"
    fp16_path = out_dir / "embeddings_fp16.npz"

    if skip_fp32 and fp32_path.exists():
        print("Reusing saved FP32 embeddings")
        emb_fp32 = load_embeddings(fp32_path)
    elif not skip_fp32:
        print("Running FP32 inference...")
        emb_fp32 = embed_batch(
            poses,
            cfg.weights_path,
            cfg.frames_num,
            cfg.joints_num,
            cfg.dim_num,
            device=device,
            precision="fp32",
        )
        save_embeddings(fp32_path, emb_fp32, meta)
        print(f"Saved {fp32_path}")
    else:
        raise FileNotFoundError(f"FP32 embeddings not found: {fp32_path}")

    if skip_fp16 and fp16_path.exists():
        print("Reusing saved FP16-sim embeddings")
        emb_fp16 = load_embeddings(fp16_path)
    elif not skip_fp16:
        print("Running FP16-simulated inference (model.half(), input half)...")
        emb_fp16 = embed_batch(
            poses,
            cfg.weights_path,
            cfg.frames_num,
            cfg.joints_num,
            cfg.dim_num,
            device=device,
            precision="fp16",
        )
        save_embeddings(fp16_path, emb_fp16, meta)
        print(f"Saved {fp16_path}")
    else:
        raise FileNotFoundError(f"FP16 embeddings not found: {fp16_path}")

    reports = {}
    cmp_fp16 = compare_embeddings(
        emb_fp32, emb_fp16, name=f"{cfg.name}: FP32 vs FP16-sim", groups=groups
    )
    reports["fp32_vs_fp16"] = cmp_fp16.to_dict()
    report_text = format_report(cmp_fp16)

    emb_android = None
    if cfg.android_embeddings_path is not None:
        print(f"Loading Android embeddings from {cfg.android_embeddings_path}")
        emb_android = load_android_embeddings(cfg.android_embeddings_path)
        if emb_android.shape != emb_fp32.shape:
            raise ValueError(
                f"Android shape {emb_android.shape} != FP32 shape {emb_fp32.shape}"
            )

        cmp_android_fp32 = compare_embeddings(
            emb_fp32,
            emb_android,
            name=f"{cfg.name}: FP32 vs Android ExecuTorch",
            groups=groups,
        )
        cmp_android_fp16 = compare_embeddings(
            emb_fp16,
            emb_android,
            name=f"{cfg.name}: FP16-sim vs Android ExecuTorch",
            groups=groups,
        )
        cmp_fp32_fp16_android = compare_embeddings(
            emb_fp32,
            emb_fp16,
            name=f"{cfg.name}: FP32 vs FP16 (with android context)",
            groups=groups,
        )
        reports["fp32_vs_android"] = cmp_android_fp32.to_dict()
        reports["fp16_vs_android"] = cmp_android_fp16.to_dict()

        report_text += "\n\n" + format_report(cmp_android_fp32)
        report_text += "\n\n" + format_report(cmp_android_fp16)

        # Which reference is closer to Android?
        cos_fp32 = cmp_android_fp32.cosine_similarity_mean
        cos_fp16 = cmp_android_fp16.cosine_similarity_mean
        reports["android_closer_to"] = "fp32" if cos_fp32 >= cos_fp16 else "fp16_sim"
        reports["android_cosine_fp32_mean"] = cos_fp32
        reports["android_cosine_fp16_mean"] = cos_fp16

    summary_path = out_dir / "comparison_report.txt"
    summary_path.write_text(report_text, encoding="utf-8")

    json_path = out_dir / "comparison_report.json"
    payload = {
        "experiment": cfg.name,
        "task": cfg.task,
        "description": cfg.description,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "n_samples": int(len(poses)),
        "config": {
            "frames_num": cfg.frames_num,
            "joints_num": cfg.joints_num,
            "dim_num": cfg.dim_num,
            "weights": str(cfg.weights_path.name),
            "data": str(cfg.data_path.name),
        },
        "reports": reports,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(report_text)
    print(f"\nSaved reports to {summary_path} and {json_path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Windows FP32/FP16 embedding experiments")
    parser.add_argument(
        "--experiment",
        choices=[e.name for e in EXPERIMENTS],
        nargs="*",
        help="Run only selected experiments (default: all)",
    )
    parser.add_argument("--skip-fp32", action="store_true", help="Reuse saved FP32 npz")
    parser.add_argument("--skip-fp16", action="store_true", help="Reuse saved FP16 npz")
    parser.add_argument(
        "--device",
        default=None,
        help="torch device (default: cpu for fp32, cuda for fp16 if available)",
    )
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else None
    selected = EXPERIMENTS
    if args.experiment:
        names = set(args.experiment)
        selected = [e for e in EXPERIMENTS if e.name in names]

    all_results = []
    for cfg in selected:
        all_results.append(
            run_experiment(
                cfg,
                device=device,
                skip_fp32=args.skip_fp32,
                skip_fp16=args.skip_fp16,
            )
        )

    # Cross-experiment summary
    summary_dir = RESULTS_DIR / "_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_lines = ["# Windows Experiment Summary\n"]
    for result in all_results:
        name = result["experiment"]
        fp16 = result["reports"]["fp32_vs_fp16"]
        summary_lines.append(f"## {name} ({result['task']})")
        summary_lines.append(
            f"- FP32 vs FP16 cos mean={fp16['cosine_similarity_mean']:.6f}, "
            f"min={fp16['cosine_similarity_min']:.6f}, "
            f"rel_l2 mean={fp16['relative_l2_mean']:.6e}, "
            f"retrieval@10 overlap={fp16['retrieval_overlap_at_k_mean']:.4f}"
        )
        if "fp32_vs_android" in result["reports"]:
            a32 = result["reports"]["fp32_vs_android"]
            a16 = result["reports"]["fp16_vs_android"]
            summary_lines.append(
                f"- Android vs FP32 cos mean={a32['cosine_similarity_mean']:.6f}, "
                f"retrieval overlap={a32['retrieval_overlap_at_k_mean']:.4f}"
            )
            summary_lines.append(
                f"- Android vs FP16-sim cos mean={a16['cosine_similarity_mean']:.6f}, "
                f"retrieval overlap={a16['retrieval_overlap_at_k_mean']:.4f}"
            )
            summary_lines.append(
                f"- Android closer to: {result['reports'].get('android_closer_to', 'n/a')}"
            )
        summary_lines.append("")

    summary_md = summary_dir / "SUMMARY.md"
    summary_md.write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"\nCross-experiment summary: {summary_md}")


if __name__ == "__main__":
    main()
