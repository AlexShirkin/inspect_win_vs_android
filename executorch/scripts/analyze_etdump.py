#!/usr/bin/env python
"""
Summarize ETDump + optional ETRecord for layer-wise debug (run in Docker conda env).

After export with dump_intermediate_outputs=True and runner:
  qnn_executor_runner ... --dump_intermediate_outputs --etdump_path etdump.etdp

Example:
  conda run -n executorch python executorch/scripts/analyze_etdump.py \
    --etdump results/executorch/mocopi_legs/etdump/etdump.etdp \
    --etrecord /path/to/etrecord.bin \
    --out-dir results/executorch/mocopi_legs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--etdump", type=Path, required=True)
    parser.add_argument("--etrecord", type=Path, default=None)
    parser.add_argument("--debug-buffer", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    from executorch.devtools import Inspector

    inspector = Inspector(
        etdump_path=str(args.etdump),
        etrecord=str(args.etrecord) if args.etrecord else None,
        debug_buffer_path=str(args.debug_buffer) if args.debug_buffer else None,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    layers: list[dict] = []
    report_lines = ["# ETDump layer summary", ""]

    for block in inspector.event_blocks:
        if block.name != "Execute":
            continue
        for event in block.events:
            if event.perf_data is not None and getattr(event, "is_delegated_op", False):
                continue
            entry = {
                "name": event.name,
                "has_debug_data": event.debug_data is not None,
            }
            if event.debug_data is not None:
                try:
                    shapes = []
                    for item in event.debug_data:
                        if hasattr(item, "shape"):
                            shapes.append(list(item.shape))
                        elif hasattr(item, "sizes"):
                            shapes.append(list(item.sizes))
                    entry["output_shapes"] = shapes
                except Exception as exc:  # noqa: BLE001
                    entry["parse_error"] = str(exc)
            layers.append(entry)
            report_lines.append(f"## {event.name}")
            report_lines.append(f"- debug_data: {entry['has_debug_data']}")
            if "output_shapes" in entry:
                report_lines.append(f"- shapes: {entry['output_shapes']}")
            report_lines.append("")

    gaps_path = args.out_dir / "layer_gaps.json"
    gaps_path.write_text(json.dumps({"layers": layers}, indent=2), encoding="utf-8")

    report_path = args.out_dir / "layer_gaps_report.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"Wrote {gaps_path} ({len(layers)} execute events)")
    print(f"Wrote {report_path}")
    print(
        "\nNext: compare worst Android samples with Windows FP32 per-layer using "
        "qnn_intermediate_output_inspector.py or Inspector.compare_results if you "
        "have reference tensors."
    )


if __name__ == "__main__":
    main()
