# ExecuTorch Experiment Findings

## Главный вывод (после PTE x86 batch)

**Android ≈ PTE QNN x86** при одинаковом drift от Windows FP32. Значит расхождение **не в device preprocessing**, а в **export + QNN FP16 runtime** (`use_fp16=True` в `build_executorch_binary`).

PyTorch FP16-sim на CUDA даёт drift на порядок меньше — «чистый» half в PyTorch ≠ QNN delegated FP16.

## Сводная таблица (rel_l2 mean, 833 poses)

| Сравнение | mocopi_body | mocopi_legs |
|-----------|------------:|------------:|
| FP32 vs FP16-sim (PyTorch) | **0.16%** | **0.42%** |
| FP32 vs Android | 1.02% | 3.49% |
| FP32 vs PTE QNN x86 | 1.01% | 3.47% |
| Android vs PTE (разница) | ~0.01% | ~0.02% |

| Сравнение | body retrieval@10 | legs retrieval@10 |
|-----------|------------------:|------------------:|
| FP32 vs FP16-sim | 0.9956 | 0.9849 |
| FP32 vs Android / PTE | ~0.964 / ~0.933 | ~0.934 / ~0.933 |

## mocopi_body (upper)

- Cosine mean vs FP32: Android 0.999927, PTE 0.999929 — практически идентично.
- Worst samples совпадают: 33, 630, 384, 291, 35, 38, …
- Group retrieval@10: delta +0.001 (стабильно).
- **Интерпретация:** upper-модель на QNN FP16 даёт ~1% drift — похоже на ожидаемое для meta_quest-style single model.

## mocopi_legs (lower) — основной источник проблемы boxing

- Cosine min: 0.9786 (Android и PTE).
- rel_l2 ~3.5% — **~3.4× хуже body**, при том что PyTorch FP16-sim для legs всего 0.42%.
- Worst samples (одинаковые у Android и PTE): **279, 831, 441, 559, 53**, затем 821, 545, 49, …
- Group retrieval@10: delta **−0.003** (лёгкая деградация metric learning).
- В avatar library отдельной legs-модели не было — «boxing broken / meta_quest OK» объясняется **dual-model + legs**, не «особенностью» avatar.

## ROOT_CAUSE (текущая гипотеза)

1. **Где ломается:** QNN HTP FP16 delegated graph (export), не Android app layer.
2. **Почему legs хуже:** тот же export path, но legs чувствительнее (GRU/BN/attention stack) — нужен layer-wise ETDump на worst samples.
3. **Почему cosine высокий, а валидация плохая:** retrieval@10 падает (особенно legs 0.93), nearest same-group neighbor меняется (~13.6% для lower) — см. Windows FINDINGS §4.

## Рекомендации

| Приоритет | Действие |
|-----------|----------|
| 1 | ETDump на **mocopi_legs**, samples 279, 831, 441, 559, 53 |
| 2 | Попробовать export **FP32** на HTP (`use_fp16=False`) или selective FP16 |
| 3 | XNNPACK control: если XNNPACK ≈ FP32, а QNN нет — backend, не модель |
| 4 | На продукте: sanity body-only; при concat upper+lower — legs доминирует drift |
| 5 | Выровнять checkpoint: Windows FP32 эталон `77999`, PTE build `70999` — для абсолютной сверки весов |

## Артефакты

- Android: `results/executorch/*/comparison_report.*`
- PTE x86: `results/executorch/*/pte_comparison_report.*`
- Сводка: `results/executorch/_summary/SUMMARY.md`
