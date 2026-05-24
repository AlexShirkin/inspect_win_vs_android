#!/usr/bin/env python
"""Cross-experiment analysis: read comparison JSON and print structured findings."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "windows" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from configs import EXPERIMENTS, RESULTS_DIR  # noqa: E402
from metrics import compare_embeddings, pairwise_cosine_matrix  # noqa: E402


def load_embeddings(path: Path) -> np.ndarray:
    with np.load(path) as archive:
        return archive["embeddings"]


def ranking_inversions(ref: np.ndarray, cand: np.ndarray, groups: np.ndarray) -> dict:
    """Count how often nearest same-group neighbor changes rank."""
    sim_ref = pairwise_cosine_matrix(ref)
    sim_cand = pairwise_cosine_matrix(cand)
    inversions = 0
    total = 0
    for i in range(len(groups)):
        same = np.where(groups == groups[i])[0]
        same = same[same != i]
        if len(same) == 0:
            continue
        ref_best = same[np.argmax(sim_ref[i, same])]
        cand_best = same[np.argmax(sim_cand[i, same])]
        if ref_best != cand_best:
            inversions += 1
        total += 1
    return {
        "same_group_nearest_changed": inversions,
        "same_group_nearest_total": total,
        "same_group_nearest_change_rate": inversions / max(total, 1),
    }


def main() -> None:
    rows = []
    boxing_parts = {}

    for cfg in EXPERIMENTS:
        out_dir = RESULTS_DIR / cfg.name
        json_path = out_dir / "comparison_report.json"
        if not json_path.exists():
            continue
        data = json.loads(json_path.read_text(encoding="utf-8"))
        fp16 = data["reports"]["fp32_vs_fp16"]
        row = {
            "name": cfg.name,
            "task": cfg.task,
            "n": data["n_samples"],
            "fp16_cos_min": fp16["cosine_similarity_min"],
            "fp16_rel_l2": fp16["relative_l2_mean"],
            "fp16_retrieval": fp16["retrieval_overlap_at_k_mean"],
            "fp16_group_delta": fp16.get("group_retrieval_delta"),
        }
        if "fp32_vs_android" in data["reports"]:
            a = data["reports"]["fp32_vs_android"]
            row.update(
                {
                    "android_cos_min": a["cosine_similarity_min"],
                    "android_cos_mean": a["cosine_similarity_mean"],
                    "android_rel_l2": a["relative_l2_mean"],
                    "android_retrieval": a["retrieval_overlap_at_k_mean"],
                    "android_group_delta": a.get("group_retrieval_delta"),
                }
            )
        rows.append(row)

        if cfg.name in ("mocopi_body", "mocopi_legs"):
            fp32 = load_embeddings(out_dir / "embeddings_fp32.npz")
            boxing_parts[cfg.name] = fp32
            if cfg.android_embeddings_path:
                from android_loader import load_android_embeddings

                boxing_parts[f"{cfg.name}_android"] = load_android_embeddings(
                    cfg.android_embeddings_path
                )

    lines = [
        "# Windows Analysis Findings",
        "",
        "## 0. Контекст деплоя (важно)",
        "",
        "| Задача | Pipeline на Android |",
        "|---|---|",
        "| **meta_quest (avatar library)** | **Одна** модель, туловище/скелет 13×3D. Отдельной legs-модели **нет**. |",
        "| **boxing (correct/incorrect)** | **Две** модели: upper (12×2D) + lower (11×2D), часто concat эмбеддингов. |",
        "",
        "Сравнивать meta_quest с boxing «в целом» некорректно. Ближайший аналог meta_quest на boxing-данных — "
        "**только mocopi_body** (upper). Legs — дополнительный источник drift, которого в avatar library не было.",
        "",
        "## 1. FP32 vs FP16-sim (PyTorch CUDA half)",
        "",
        "Если проблема только в FP16, все модели должны показывать одинаковый drift.",
        "",
        "| Experiment | cos_min | rel_l2 mean | retrieval@10 | group_retrieval delta |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['name']} | {r['fp16_cos_min']:.6f} | {r['fp16_rel_l2']:.4e} | "
            f"{r['fp16_retrieval']:.4f} | {r.get('fp16_group_delta', 0):+.4f} |"
        )

    lines.extend(
        [
            "",
            "**Вывод:** FP16-sim drift минимален для всех трёх моделей (cos_min > 0.99968). "
            "Разница «boxing vs avatar» в PyTorch FP16 **не объясняет** разное поведение на Android.",
            "",
            "## 2. Android ExecuTorch vs Windows FP32",
            "",
            "**meta_quest на Android не измерялся** (нет дампа embeddings). Ниже — только boxing-модели.",
            "",
            "| Experiment | cos_mean | cos_min | rel_l2 mean | retrieval@10 | group delta |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for r in rows:
        if "android_cos_mean" not in r:
            continue
        lines.append(
            f"| {r['name']} | {r['android_cos_mean']:.6f} | {r['android_cos_min']:.6f} | "
            f"{r['android_rel_l2']:.4e} | {r['android_retrieval']:.4f} | "
            f"{r.get('android_group_delta', 0):+.4f} |"
        )

    body_row = next((r for r in rows if r["name"] == "mocopi_body"), None)
    legs_row = next((r for r in rows if r["name"] == "mocopi_legs"), None)
    meta_row = next((r for r in rows if r["name"] == "meta_quest_avatar"), None)

    lines.extend(
        [
            "",
            "**Паттерн с учётом single-model meta_quest:**",
            "- **mocopi_body** (аналог meta_quest по роли «одна upper-модель»): Android rel_l2 ~1%, group delta +0.001 — "
            "**скорее всего похоже на то, что вы видели с meta_quest**.",
            "- **mocopi_legs**: rel_l2 ~3.5% — **~3.4× хуже body**. Этого компонента в avatar library **не было**.",
            "- Различие «meta_quest OK / boxing broken» объясняется не типом задачи, а **dual-model pipeline + legs drift**, "
            "а не тем, что avatar-модель «особенная».",
            "",
            "Если на устройстве валидация «ломается», а здесь cosine высокий — проверьте:",
            "1. runtime preprocessing (нормализация per-sequence на Android)",
            "2. concat upper+lower embedding перед классификатором (legs тянет вниз)",
            "3. что `*_fixed.npz` — это уже исправленный дамп, а не оригинальный broken output",
            "",
        ]
    )

    if body_row and meta_row and legs_row:
        lines.extend(
            [
                "### Сводка: meta_quest vs boxing-компоненты",
                "",
                "| | meta_quest (FP32↔FP16) | mocopi_body Android | mocopi_legs Android |",
                "|---|---:|---:|---:|",
                f"| rel_l2 / drift | {meta_row['fp16_rel_l2']:.4e} | {body_row['android_rel_l2']:.4e} | {legs_row['android_rel_l2']:.4e} |",
                f"| retrieval@10 | {meta_row['fp16_retrieval']:.4f} | {body_row['android_retrieval']:.4f} | {legs_row['android_retrieval']:.4f} |",
                f"| group delta | {meta_row.get('fp16_group_delta', 0):+.4f} | {body_row.get('android_group_delta', 0):+.4f} | {legs_row.get('android_group_delta', 0):+.4f} |",
                "",
                "meta_quest по FP16-стабильности близок к body; legs — выброс, которого в avatar pipeline не было.",
                "",
            ]
        )

    if "mocopi_body" in boxing_parts and "mocopi_legs" in boxing_parts:
        body = boxing_parts["mocopi_body"]
        legs = boxing_parts["mocopi_legs"]
        combined_fp32 = np.concatenate([body, legs], axis=1)

        if "mocopi_body_android" in boxing_parts and "mocopi_legs_android" in boxing_parts:
            combined_android = np.concatenate(
                [boxing_parts["mocopi_body_android"], boxing_parts["mocopi_legs_android"]],
                axis=1,
            )
            cmp_combined = compare_embeddings(
                combined_fp32,
                combined_android,
                name="boxing combined [body|legs] FP32 vs Android",
            )
            lines.extend(
                [
                    "## 3. Combined boxing embedding (concat upper + lower)",
                    "",
                    f"- cosine mean: {cmp_combined.cosine_similarity_mean:.6f}",
                    f"- cosine min: {cmp_combined.cosine_similarity_min:.6f}",
                    f"- relative L2 mean: {cmp_combined.relative_l2_mean:.4e}",
                    f"- retrieval overlap@10: {cmp_combined.retrieval_overlap_at_k_mean:.4f}",
                    "",
                    "Combined embedding — **только boxing** (avatar library так не работает). "
                    "Legs drift доминирует в concat.",
                    "",
                ]
            )

    body_dir = RESULTS_DIR / "mocopi_body"
    legs_dir = RESULTS_DIR / "mocopi_legs"
    inversion_lines = ["## 4. Same-group nearest neighbor stability (Android vs FP32)", ""]

    for part_name, label in (("mocopi_body", "upper (аналог meta_quest)"), ("mocopi_legs", "lower (только boxing)")):
        part_dir = RESULTS_DIR / part_name
        android_key = f"{part_name}_android"
        if not (part_dir / "embeddings_fp32.npz").exists() or android_key not in boxing_parts:
            continue
        with np.load(part_dir / "embeddings_fp32.npz") as archive:
            groups = archive["pose2group"] if "pose2group" in archive else None
        if groups is None:
            continue
        inv = ranking_inversions(boxing_parts[part_name], boxing_parts[android_key], groups)
        inversion_lines.append(
            f"- **{label}:** {inv['same_group_nearest_changed']}/{inv['same_group_nearest_total']} "
            f"({100 * inv['same_group_nearest_change_rate']:.1f}%) сменили ближайшего same-group соседа"
        )

    if len(inversion_lines) > 2:
        inversion_lines.extend(
            [
                "",
                "Upper (~аналог meta_quest single-model) должен быть стабильнее lower. "
                "Даже при высоком cosine смена nearest neighbor ломает metric learning.",
                "",
            ]
        )
        lines.extend(inversion_lines)

    lines.extend(
        [
            "## 5. Рекомендации для ExecuTorch-коллеги",
            "",
            "1. Layer-wise ETDump на **mocopi_legs** — там max Android drift; meta_quest legs не использовал.",
            "2. Для sanity check: body-only inference на Android (как meta_quest) — ожидаем хороший результат.",
            "3. Сравнить preprocessing: min-max per sequence на Android vs `calc_embedding`.",
            "4. XNNPACK CPU .pte vs Qualcomm — локализовать backend vs export.",
            "5. Worst samples legs: [279, 831, 441, 559, 53].",
            "",
        ]
    )

    out_path = RESULTS_DIR / "_summary" / "FINDINGS.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
