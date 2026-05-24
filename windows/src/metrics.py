"""Embedding comparison metrics for FP32 vs FP16 vs Android."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(norms, eps)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_n = _l2_normalize(a)
    b_n = _l2_normalize(b)
    return np.sum(a_n * b_n, axis=1)


def pairwise_cosine_matrix(embeddings: np.ndarray) -> np.ndarray:
    emb = _l2_normalize(embeddings)
    return emb @ emb.T


def pairwise_l2_matrix(embeddings: np.ndarray) -> np.ndarray:
    sq = np.sum(embeddings ** 2, axis=1, keepdims=True)
    dist2 = sq + sq.T - 2.0 * (embeddings @ embeddings.T)
    return np.sqrt(np.maximum(dist2, 0.0))


def matrix_upper_triangle_values(matrix: np.ndarray) -> np.ndarray:
    idx = np.triu_indices(matrix.shape[0], k=1)
    return matrix[idx]


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    rx = x.argsort().argsort().astype(np.float64)
    ry = y.argsort().argsort().astype(np.float64)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    if denom == 0:
        return float("nan")
    return float((rx * ry).sum() / denom)


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    x -= x.mean()
    y -= y.mean()
    denom = np.sqrt((x ** 2).sum() * (y ** 2).sum())
    if denom == 0:
        return float("nan")
    return float((x * y).sum() / denom)


def topk_neighbors(sim_matrix: np.ndarray, k: int = 10) -> np.ndarray:
    """Return indices of top-k neighbors excluding self."""
    n = sim_matrix.shape[0]
    k = min(k, n - 1)
    neighbors = np.argsort(-sim_matrix, axis=1)[:, 1 : k + 1]
    return neighbors


def retrieval_overlap(
    ref_neighbors: np.ndarray,
    cmp_neighbors: np.ndarray,
) -> Dict[str, float]:
    k = ref_neighbors.shape[1]
    overlaps = [
        len(set(ref_neighbors[i]).intersection(cmp_neighbors[i])) / k
        for i in range(ref_neighbors.shape[0])
    ]
    overlaps = np.asarray(overlaps, dtype=np.float64)
    return {
        "retrieval_overlap_at_k_mean": float(overlaps.mean()),
        "retrieval_overlap_at_k_min": float(overlaps.min()),
        "retrieval_overlap_at_k_p05": float(np.percentile(overlaps, 5)),
        "retrieval_overlap_at_k_p50": float(np.percentile(overlaps, 50)),
        "retrieval_overlap_at_k_p95": float(np.percentile(overlaps, 95)),
    }


def group_retrieval_accuracy(
    embeddings: np.ndarray,
    groups: np.ndarray,
    k: int = 10,
) -> float:
    sim = pairwise_cosine_matrix(embeddings)
    neighbors = topk_neighbors(sim, k=k)
    hits = []
    for i in range(len(groups)):
        neighbor_groups = groups[neighbors[i]]
        hits.append(np.mean(neighbor_groups == groups[i]))
    return float(np.mean(hits))


@dataclass
class ComparisonReport:
    name: str
    n_samples: int
    # per-sample embedding element diffs
    abs_diff_mean: float
    abs_diff_max: float
    abs_diff_p95: float
    abs_diff_p99: float
  # vector-level
    l2_distance_mean: float
    l2_distance_max: float
    l2_distance_p95: float
    l2_distance_p99: float
    relative_l2_mean: float
    relative_l2_max: float
    cosine_similarity_mean: float
    cosine_similarity_min: float
    cosine_similarity_p01: float
    cosine_similarity_p05: float
    # pairwise structure
    cosine_matrix_spearman: float
    cosine_matrix_pearson: float
    l2_matrix_spearman: float
    l2_matrix_pearson: float
    # retrieval
    retrieval_overlap_at_k_mean: float
    retrieval_overlap_at_k_min: float
    retrieval_overlap_at_k_p05: float
    retrieval_overlap_at_k_p50: float
    retrieval_overlap_at_k_p95: float
    ref_group_retrieval_at_10: Optional[float] = None
    cmp_group_retrieval_at_10: Optional[float] = None
    group_retrieval_delta: Optional[float] = None
    worst_samples_by_cosine: Optional[List[int]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compare_embeddings(
    reference: np.ndarray,
    candidate: np.ndarray,
    name: str,
    groups: Optional[np.ndarray] = None,
    k: int = 10,
    worst_n: int = 20,
) -> ComparisonReport:
    if reference.shape != candidate.shape:
        raise ValueError(f"Shape mismatch: {reference.shape} vs {candidate.shape}")

    diff = np.abs(reference - candidate)
    l2 = np.linalg.norm(reference - candidate, axis=1)
    ref_norm = np.linalg.norm(reference, axis=1)
    rel_l2 = l2 / np.maximum(ref_norm, 1e-12)
    cos = cosine_similarity(reference, candidate)

    ref_cos_mat = pairwise_cosine_matrix(reference)
    cmp_cos_mat = pairwise_cosine_matrix(candidate)
    ref_l2_mat = pairwise_l2_matrix(reference)
    cmp_l2_mat = pairwise_l2_matrix(candidate)

    ref_tri = matrix_upper_triangle_values(ref_cos_mat)
    cmp_tri = matrix_upper_triangle_values(cmp_cos_mat)
    ref_l2_tri = matrix_upper_triangle_values(ref_l2_mat)
    cmp_l2_tri = matrix_upper_triangle_values(cmp_l2_mat)

    ref_neighbors = topk_neighbors(ref_cos_mat, k=k)
    cmp_neighbors = topk_neighbors(cmp_cos_mat, k=k)
    overlap = retrieval_overlap(ref_neighbors, cmp_neighbors)

    ref_group_acc = None
    cmp_group_acc = None
    group_delta = None
    if groups is not None:
        ref_group_acc = group_retrieval_accuracy(reference, groups, k=k)
        cmp_group_acc = group_retrieval_accuracy(candidate, groups, k=k)
        group_delta = cmp_group_acc - ref_group_acc

    worst = np.argsort(cos)[:worst_n].tolist()

    return ComparisonReport(
        name=name,
        n_samples=int(reference.shape[0]),
        abs_diff_mean=float(diff.mean()),
        abs_diff_max=float(diff.max()),
        abs_diff_p95=float(np.percentile(diff, 95)),
        abs_diff_p99=float(np.percentile(diff, 99)),
        l2_distance_mean=float(l2.mean()),
        l2_distance_max=float(l2.max()),
        l2_distance_p95=float(np.percentile(l2, 95)),
        l2_distance_p99=float(np.percentile(l2, 99)),
        relative_l2_mean=float(rel_l2.mean()),
        relative_l2_max=float(rel_l2.max()),
        cosine_similarity_mean=float(cos.mean()),
        cosine_similarity_min=float(cos.min()),
        cosine_similarity_p01=float(np.percentile(cos, 1)),
        cosine_similarity_p05=float(np.percentile(cos, 5)),
        cosine_matrix_spearman=spearman_corr(ref_tri, cmp_tri),
        cosine_matrix_pearson=pearson_corr(ref_tri, cmp_tri),
        l2_matrix_spearman=spearman_corr(ref_l2_tri, cmp_l2_tri),
        l2_matrix_pearson=pearson_corr(ref_l2_tri, cmp_l2_tri),
        retrieval_overlap_at_k_mean=overlap["retrieval_overlap_at_k_mean"],
        retrieval_overlap_at_k_min=overlap["retrieval_overlap_at_k_min"],
        retrieval_overlap_at_k_p05=overlap["retrieval_overlap_at_k_p05"],
        retrieval_overlap_at_k_p50=overlap["retrieval_overlap_at_k_p50"],
        retrieval_overlap_at_k_p95=overlap["retrieval_overlap_at_k_p95"],
        ref_group_retrieval_at_10=ref_group_acc,
        cmp_group_retrieval_at_10=cmp_group_acc,
        group_retrieval_delta=group_delta,
        worst_samples_by_cosine=worst,
    )


def format_report(report: ComparisonReport) -> str:
    lines = [
        f"=== {report.name} ===",
        f"samples: {report.n_samples}",
        "",
        "Per-element |ref - cand|:",
        f"  mean={report.abs_diff_mean:.6e}  max={report.abs_diff_max:.6e}  p95={report.abs_diff_p95:.6e}  p99={report.abs_diff_p99:.6e}",
        "",
        "Per-vector L2 distance:",
        f"  mean={report.l2_distance_mean:.6e}  max={report.l2_distance_max:.6e}  p95={report.l2_distance_p95:.6e}  p99={report.l2_distance_p99:.6e}",
        "",
        "Relative L2 (||diff|| / ||ref||):",
        f"  mean={report.relative_l2_mean:.6e}  max={report.relative_l2_max:.6e}",
        "",
        "Cosine similarity (ref vs cand, per sample):",
        f"  mean={report.cosine_similarity_mean:.6f}  min={report.cosine_similarity_min:.6f}  p01={report.cosine_similarity_p01:.6f}  p05={report.cosine_similarity_p05:.6f}",
        "",
        "Pairwise structure preservation:",
        f"  cosine matrix: spearman={report.cosine_matrix_spearman:.6f}  pearson={report.cosine_matrix_pearson:.6f}",
        f"  L2 matrix:     spearman={report.l2_matrix_spearman:.6f}  pearson={report.l2_matrix_pearson:.6f}",
        "",
        f"Retrieval overlap@10 (ref neighbors vs cand neighbors):",
        f"  mean={report.retrieval_overlap_at_k_mean:.4f}  min={report.retrieval_overlap_at_k_min:.4f}  p05={report.retrieval_overlap_at_k_p05:.4f}  p50={report.retrieval_overlap_at_k_p50:.4f}  p95={report.retrieval_overlap_at_k_p95:.4f}",
    ]
    if report.ref_group_retrieval_at_10 is not None:
        lines.extend(
            [
                "",
                "Group retrieval@10 (same pose2group in top-10):",
                f"  ref={report.ref_group_retrieval_at_10:.4f}  cand={report.cmp_group_retrieval_at_10:.4f}  delta={report.group_retrieval_delta:+.4f}",
            ]
        )
    if report.worst_samples_by_cosine:
        lines.extend(["", f"Worst {len(report.worst_samples_by_cosine)} samples by cosine: {report.worst_samples_by_cosine}"])
    return "\n".join(lines)
