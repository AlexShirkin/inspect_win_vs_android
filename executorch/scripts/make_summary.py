#!/usr/bin/env python
"""Aggregate results/executorch reports into _summary/SUMMARY.md."""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EXEC_RESULTS = REPO_ROOT / "results" / "executorch"


def main() -> None:
    lines = ["# ExecuTorch Experiment Summary", ""]

    for exp_dir in sorted(EXEC_RESULTS.iterdir()):
        if not exp_dir.is_dir() or exp_dir.name.startswith("_"):
            continue
        name = exp_dir.name
        lines.append(f"## {name}")

        android_cmp = exp_dir / "comparison_report.json"
        if android_cmp.exists():
            data = json.loads(android_cmp.read_text(encoding="utf-8"))
            for key, label in (
                ("fp32_vs_android", "Android vs Windows FP32"),
                ("fp16_vs_android", "Android vs Windows FP16-sim"),
            ):
                if key in data.get("reports", {}):
                    r = data["reports"][key]
                    lines.append(
                        f"- {label}: cos_mean={r['cosine_similarity_mean']:.6f}, "
                        f"cos_min={r['cosine_similarity_min']:.6f}, "
                        f"rel_l2={r['relative_l2_mean']:.6e}, "
                        f"retrieval@10={r['retrieval_overlap_at_k_mean']:.4f}"
                    )

        pte_cmp = exp_dir / "pte_comparison_report.json"
        if pte_cmp.exists():
            data = json.loads(pte_cmp.read_text(encoding="utf-8"))
            r = data["report"]
            label = data.get("pte_label", "PTE")
            lines.append(
                f"- {label} vs FP32: cos_mean={r['cosine_similarity_mean']:.6f}, "
                f"cos_min={r['cosine_similarity_min']:.6f}, "
                f"rel_l2={r['relative_l2_mean']:.6e}, "
                f"retrieval@10={r['retrieval_overlap_at_k_mean']:.4f}"
            )

        layer_report = exp_dir / "layer_gaps_report.txt"
        if layer_report.exists():
            lines.append(f"- layer debug: `{layer_report.relative_to(REPO_ROOT)}`")

        lines.append("")

    out_dir = EXEC_RESULTS / "_summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "SUMMARY.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
