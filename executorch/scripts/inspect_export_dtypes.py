#!/usr/bin/env python
"""
Print dtypes in the export graph for mmd_nca_net (run in ExecuTorch conda env).

Example (inside Docker, after editing model.py weights/joints manually):
  cd /workspace/workspace/executorch
  conda run -n executorch python \
    /workspace/inspect_win_vs_android/executorch/scripts/inspect_export_dtypes.py \
    --executorch-root /workspace/workspace/executorch
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--executorch-root",
        type=Path,
        required=True,
        help="Path to executorch tree (examples.models importable)",
    )
    args = parser.parse_args()

    root = args.executorch_root.resolve()
    parent = root.parent
    if str(parent) not in sys.path:
        sys.path.insert(0, str(parent))

    import torch
    from executorch.examples.models.mmd_nca_net import MmdNcaNetModel

    instance = MmdNcaNetModel()
    model = instance.get_eager_model().eval()
    inputs = instance.get_example_inputs()

    print("=== Eager model parameter dtypes ===")
    param_dtypes = Counter(p.dtype for p in model.parameters())
    for dtype, count in sorted(param_dtypes.items(), key=lambda x: str(x[0])):
        print(f"  {dtype}: {count}")

    print("\n=== Example input ===")
    print(f"  shape={tuple(inputs[0].shape)} dtype={inputs[0].dtype}")

    exported = torch.export.export(model, inputs)
    gm = exported.module()

    print("\n=== Exported graph: placeholder / output dtypes ===")
    for node in gm.graph.nodes:
        if node.op in ("placeholder", "output"):
            print(f"  {node.op}: {node.name} -> {[getattr(t, 'dtype', t) for t in node.args if hasattr(t, 'dtype')]}")

    print("\n=== Exported graph: call_function dtypes (sample) ===")
    dtype_ops: Counter[str] = Counter()
    cast_nodes = []
    for node in gm.graph.nodes:
        if node.op == "call_function":
            target = str(node.target)
            if "to" in target.lower() or "dtype" in target.lower():
                cast_nodes.append((node.name, target, node.args))
            if hasattr(node, "meta") and "tensor_meta" in node.meta:
                tm = node.meta["tensor_meta"]
                if hasattr(tm, "dtype"):
                    dtype_ops[str(tm.dtype)] += 1

    if cast_nodes:
        print("  Explicit cast / to() nodes:")
        for name, target, nargs in cast_nodes[:30]:
            print(f"    {name}: {target} args={nargs}")
        if len(cast_nodes) > 30:
            print(f"    ... and {len(cast_nodes) - 30} more")
    else:
        print("  No explicit to()/dtype nodes in FX graph (good).")

    print("\n=== Tensor meta dtype histogram (call_function) ===")
    for dtype, count in dtype_ops.most_common():
        print(f"  {dtype}: {count}")

    print("\nChecklist:")
    print("  [ ] weights and trace input same dtype (fp32 export vs fp16 HTP)")
    print("  [ ] no unexpected fp16↔fp32 between GRU/BN/Softmax")
    print("  [ ] build_executorch_binary use_fp16=True when not quantizing (QNN default)")


if __name__ == "__main__":
    main()
