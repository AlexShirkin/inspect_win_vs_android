#!/usr/bin/env python
"""
Export mmd_nca_net to XNNPACK CPU .pte for control experiment (run in Docker conda env).

Edit MmdNcaNetModel in examples/models/mmd_nca_net/model.py (weights, joints) before running.

Example:
  cd /workspace/workspace/executorch
  conda run -n executorch python \
    /workspace/inspect_win_vs_android/executorch/scripts/export_xnnpack_pte.py \
    --executorch-root /workspace/workspace/executorch \
    --out-dir /workspace/workspace/executorch/deeplab_v3 \
    --name mmd_nca_body_xnnpack_fp32
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--executorch-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--name", default="mmd_nca_xnnpack_fp32")
    parser.add_argument("--etrecord", type=Path, default=None)
    args = parser.parse_args()

    root = args.executorch_root.resolve()
    if str(root.parent) not in sys.path:
        sys.path.insert(0, str(root.parent))

    import torch
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.examples.models.mmd_nca_net import MmdNcaNetModel
    from executorch.exir import EdgeCompileConfig, ExecutorchBackendConfig, to_edge_transform_and_lower
    from executorch.extension.export_util.utils import save_pte_program

    if args.etrecord is not None:
        from executorch.devtools import generate_etrecord

    instance = MmdNcaNetModel()
    model = instance.get_eager_model().eval()
    example_inputs = instance.get_example_inputs()

    ep = torch.export.export(model, example_inputs)
    edge = to_edge_transform_and_lower(
        ep,
        partitioner=[XnnpackPartitioner()],
        compile_config=EdgeCompileConfig(_check_ir_validity=False, _skip_dim_order=True),
    )
    edge_copy = copy.deepcopy(edge)
    exec_prog = edge.to_executorch(config=ExecutorchBackendConfig(extract_delegate_segments=False))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.etrecord is not None:
        generate_etrecord(str(args.etrecord), edge_copy, exec_prog)
        print(f"ETRecord: {args.etrecord}")

    pte_path = save_pte_program(exec_prog, args.name, str(args.out_dir))
    print(f"PTE: {pte_path}")


if __name__ == "__main__":
    main()
