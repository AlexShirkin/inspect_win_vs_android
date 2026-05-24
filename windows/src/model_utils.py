"""Model loading and embedding inference (FP32 / simulated FP16)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mmd_nca_net_mobile import MMD_NCA_Mobile_Net  # noqa: E402


Precision = Literal["fp32", "fp16"]


def normalize_sequence(sequence: np.ndarray) -> np.ndarray:
    """Per-sequence min-max normalization used in production code."""
    seq_min = sequence.min()
    seq_max = sequence.max()
    denom = seq_max - seq_min
    if denom == 0:
        return np.zeros_like(sequence, dtype=np.float32)
    return ((sequence - seq_min) / denom).astype(np.float32)


def resolve_device(precision: Precision, device: Optional[torch.device] = None) -> torch.device:
    """FP16 full-model simulation needs CUDA (BN/GRU lack CPU Half kernels in torch 1.10)."""
    if device is not None:
        return device
    if precision == "fp16":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError(
            "FP16 simulation requires CUDA on this PyTorch build. "
            "Use --device cuda or run on a machine with GPU."
        )
    return torch.device("cpu")


def build_model(
    weights_path: Path,
    frames_num: int,
    joints_num: int,
    dim_num: int,
    device: torch.device,
    precision: Precision = "fp32",
) -> MMD_NCA_Mobile_Net:
    model = MMD_NCA_Mobile_Net(device, frames_num, joints_num, dim_num)
    model.load_weights(str(weights_path), device)
    model = model.to(device)
    if precision == "fp16":
        model = model.half()
    return model


def sequence_to_tensor(
    sequence: np.ndarray,
    frames_num: int,
    joints_num: int,
    dim_num: int,
    device: torch.device,
    precision: Precision = "fp32",
) -> torch.Tensor:
    feature_dim = joints_num * dim_num
    tensor = (
        torch.as_tensor(sequence, dtype=torch.float32)
        .reshape(1, frames_num, feature_dim)
        .permute(1, 0, 2)
        .to(device)
    )
    if precision == "fp16":
        tensor = tensor.half()
    return tensor


@torch.no_grad()
def embed_sequence(
    model: MMD_NCA_Mobile_Net,
    sequence: np.ndarray,
    frames_num: int,
    joints_num: int,
    dim_num: int,
    device: torch.device,
    precision: Precision = "fp32",
    normalize: bool = True,
) -> np.ndarray:
    seq = normalize_sequence(sequence) if normalize else sequence.astype(np.float32)
    tensor = sequence_to_tensor(
        seq, frames_num, joints_num, dim_num, device, precision=precision
    )
    embedding = model.A_LSTM.gen_embedding(tensor)
    return embedding.float().cpu().numpy().squeeze()


@torch.no_grad()
def embed_batch(
    poses: np.ndarray,
    weights_path: Path,
    frames_num: int,
    joints_num: int,
    dim_num: int,
    device: Optional[torch.device] = None,
    precision: Precision = "fp32",
    normalize: bool = True,
    batch_log_every: int = 100,
) -> np.ndarray:
    device = resolve_device(precision, device)

    model = build_model(
        weights_path, frames_num, joints_num, dim_num, device, precision=precision
    )
    embeddings = np.zeros((len(poses), 128), dtype=np.float32)

    for idx, pose in enumerate(poses):
        embeddings[idx] = embed_sequence(
            model,
            pose,
            frames_num,
            joints_num,
            dim_num,
            device,
            precision=precision,
            normalize=normalize,
        )
        if batch_log_every and (idx + 1) % batch_log_every == 0:
            print(f"  embedded {idx + 1}/{len(poses)} ({precision})")

    return embeddings
