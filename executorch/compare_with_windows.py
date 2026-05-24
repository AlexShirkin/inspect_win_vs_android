#!/usr/bin/env python
"""
Compare Android ExecuTorch embeddings with Windows reference outputs.

Run on the ExecuTorch machine after syncing results/windows/ from Windows.

Usage:
  python executorch/compare_with_windows.py --experiment mocopi_body
  python executorch/compare_with_windows.py --all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
WINDOWS_SRC = REPO_ROOT / "windows" / "src"
if str(WINDOWS_SRC) not in sys.path:
    sys.path.insert(0, str(WINDOWS_SRC))

from configs import EXPERIMENTS  # noqa: E402
from metrics import compare_embeddings, format_report  # noqa: E402


def load_npz_embeddings(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=True) as archive:
        return archive["embeddings"].astype(np.float32)


def load_android(path: Path) -> tuple[np.ndarray, dict]:
    with np.load(path, allow_pickle=True) as archive:
        emb = archive["embeddings"].astype(np.float32)
        meta = {}
        if "pose2group" in archive:
            meta["pose2group"] = archive["pose2group"]
    return emb, meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=[e.name for e in EXPERIMENTS])
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if not args.all and not args.experiment:
        parser.error("Specify --experiment NAME or --all")

    selected = EXPERIMENTS if args.all else [e for e in EXPERIMENTS if e.name == args.experiment]

    out_root = REPO_ROOT / "results" / "executorch"
    out_root.mkdir(parents=True, exist_ok=True)

    for cfg in selected:
        win_dir = REPO_ROOT / "results" / "windows" / cfg.name
        fp32_path = win_dir / "embeddings_fp32.npz"
        fp16_path = win_dir / "embeddings_fp16.npz"

        if not fp32_path.exists():
            print(f"SKIP {cfg.name}: missing {fp32_path} (run windows/run_all.py first)")
            continue

        emb_fp32 = load_npz_embeddings(fp32_path)
        emb_fp16 = load_npz_embeddings(fp16_path) if fp16_path.exists() else None

        android_path = cfg.android_embeddings_path
        if android_path is None:
            print(f"SKIP {cfg.name}: no Android embeddings path configured")
            continue

        emb_android, meta = load_android(android_path)
        groups = meta.get("pose2group")

        reports = {}
        text_parts = []

        r32 = compare_embeddings(
            emb_fp32, emb_android, name=f"{cfg.name}: Windows FP32 vs Android", groups=groups
        )
        reports["fp32_vs_android"] = r32.to_dict()
        text_parts.append(format_report(r32))

        if emb_fp16 is not None:
            r16 = compare_embeddings(
                emb_fp16, emb_android, name=f"{cfg.name}: Windows FP16-sim vs Android", groups=groups
            )
            reports["fp16_vs_android"] = r16.to_dict()
            text_parts.append(format_report(r16))

        exp_out = out_root / cfg.name
        exp_out.mkdir(parents=True, exist_ok=True)
        (exp_out / "comparison_report.txt").write_text("\n\n".join(text_parts), encoding="utf-8")
        (exp_out / "comparison_report.json").write_text(
            json.dumps({"experiment": cfg.name, "reports": reports}, indent=2),
            encoding="utf-8",
        )
        print(f"Wrote {exp_out / 'comparison_report.txt'}")


if __name__ == "__main__":
    main()
