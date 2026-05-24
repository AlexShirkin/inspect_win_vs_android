#!/usr/bin/env python
"""Compare PTE runner embeddings (x86 QNN or device) with Windows FP32 reference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
WINDOWS_SRC = REPO_ROOT / "windows" / "src"
if str(WINDOWS_SRC) not in sys.path:
    sys.path.insert(0, str(WINDOWS_SRC))

from configs import EXPERIMENTS, RESULTS_DIR  # noqa: E402
from data_io import get_metadata, load_merged_three_npz  # noqa: E402
from metrics import compare_embeddings, format_report  # noqa: E402


def load_embeddings(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=True) as archive:
        return archive["embeddings"].astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, choices=[e.name for e in EXPERIMENTS])
    parser.add_argument("--pte-embeddings", type=Path, required=True)
    parser.add_argument(
        "--label",
        default="PTE QNN",
        help="Label in report (e.g. 'PTE QNN x86', 'PTE XNNPACK')",
    )
    args = parser.parse_args()

    cfg = next(e for e in EXPERIMENTS if e.name == args.experiment)
    fp32_path = RESULTS_DIR / cfg.name / "embeddings_fp32.npz"
    if not fp32_path.exists():
        raise FileNotFoundError(f"Missing Windows reference: {fp32_path}")

    emb_ref = load_embeddings(fp32_path)
    emb_pte = load_embeddings(args.pte_embeddings)
    if emb_ref.shape != emb_pte.shape:
        raise ValueError(f"Shape mismatch ref={emb_ref.shape} pte={emb_pte.shape}")

    db = load_merged_three_npz(cfg.data_path)
    groups = get_metadata(db).get("pose2group")

    report = compare_embeddings(
        emb_ref,
        emb_pte,
        name=f"{cfg.name}: Windows FP32 vs {args.label}",
        groups=groups,
    )

    out_dir = REPO_ROOT / "results" / "executorch" / cfg.name
    out_dir.mkdir(parents=True, exist_ok=True)
    txt = format_report(report)
    (out_dir / "pte_comparison_report.txt").write_text(txt, encoding="utf-8")
    (out_dir / "pte_comparison_report.json").write_text(
        json.dumps(
            {
                "experiment": cfg.name,
                "pte_label": args.label,
                "pte_embeddings": str(args.pte_embeddings),
                "report": report.to_dict(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(txt)
    print(f"\nWrote {out_dir / 'pte_comparison_report.txt'}")


if __name__ == "__main__":
    main()
