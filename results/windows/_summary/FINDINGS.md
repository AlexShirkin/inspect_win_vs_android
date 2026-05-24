# Windows Analysis Findings

## 0. Контекст деплоя (важно)

| Задача | Pipeline на Android |
|---|---|
| **meta_quest (avatar library)** | **Одна** модель, туловище/скелет 13×3D. Отдельной legs-модели **нет**. |
| **boxing (correct/incorrect)** | **Две** модели: upper (12×2D) + lower (11×2D), часто concat эмбеддингов. |

Сравнивать meta_quest с boxing «в целом» некорректно. Ближайший аналог meta_quest на boxing-данных — **только mocopi_body** (upper). Legs — дополнительный источник drift, которого в avatar library не было.

## 1. FP32 vs FP16-sim (PyTorch CUDA half)

Если проблема только в FP16, все модели должны показывать одинаковый drift.

| Experiment | cos_min | rel_l2 mean | retrieval@10 | group_retrieval delta |
|---|---:|---:|---:|---:|
| mocopi_body | 0.999977 | 1.6225e-03 | 0.9956 | +0.0002 |
| mocopi_legs | 0.999680 | 4.2125e-03 | 0.9849 | +0.0012 |
| meta_quest_avatar | 0.999904 | 3.1313e-03 | 0.9826 | +0.0000 |

**Вывод:** FP16-sim drift минимален для всех трёх моделей (cos_min > 0.99968). Разница «boxing vs avatar» в PyTorch FP16 **не объясняет** разное поведение на Android.

## 2. Android ExecuTorch vs Windows FP32

**meta_quest на Android не измерялся** (нет дампа embeddings). Ниже — только boxing-модели.

| Experiment | cos_mean | cos_min | rel_l2 mean | retrieval@10 | group delta |
|---|---:|---:|---:|---:|---:|
| mocopi_body | 0.999927 | 0.984466 | 1.0209e-02 | 0.9639 | +0.0010 |
| mocopi_legs | 0.999406 | 0.978626 | 3.4913e-02 | 0.9334 | -0.0026 |

**Паттерн с учётом single-model meta_quest:**
- **mocopi_body** (аналог meta_quest по роли «одна upper-модель»): Android rel_l2 ~1%, group delta +0.001 — **скорее всего похоже на то, что вы видели с meta_quest**.
- **mocopi_legs**: rel_l2 ~3.5% — **~3.4× хуже body**. Этого компонента в avatar library **не было**.
- Различие «meta_quest OK / boxing broken» объясняется не типом задачи, а **dual-model pipeline + legs drift**, а не тем, что avatar-модель «особенная».

Если на устройстве валидация «ломается», а здесь cosine высокий — проверьте:
1. runtime preprocessing (нормализация per-sequence на Android)
2. concat upper+lower embedding перед классификатором (legs тянет вниз)
3. что `*_fixed.npz` — это уже исправленный дамп, а не оригинальный broken output

### Сводка: meta_quest vs boxing-компоненты

| | meta_quest (FP32↔FP16) | mocopi_body Android | mocopi_legs Android |
|---|---:|---:|---:|
| rel_l2 / drift | 3.1313e-03 | 1.0209e-02 | 3.4913e-02 |
| retrieval@10 | 0.9826 | 0.9639 | 0.9334 |
| group delta | +0.0000 | +0.0010 | -0.0026 |

meta_quest по FP16-стабильности близок к body; legs — выброс, которого в avatar pipeline не было.

## 3. Combined boxing embedding (concat upper + lower)

- cosine mean: 0.999703
- cosine min: 0.983934
- relative L2 mean: 2.2126e-02
- retrieval overlap@10: 0.9616

Combined embedding — **только boxing** (avatar library так не работает). Legs drift доминирует в concat.

## 4. Same-group nearest neighbor stability (Android vs FP32)

- **upper (аналог meta_quest):** 37/833 (4.4%) сменили ближайшего same-group соседа
- **lower (только boxing):** 113/833 (13.6%) сменили ближайшего same-group соседа

Upper (~аналог meta_quest single-model) должен быть стабильнее lower. Даже при высоком cosine смена nearest neighbor ломает metric learning.

## 5. Рекомендации для ExecuTorch-коллеги

1. Layer-wise ETDump на **mocopi_legs** — там max Android drift; meta_quest legs не использовал.
2. Для sanity check: body-only inference на Android (как meta_quest) — ожидаем хороший результат.
3. Сравнить preprocessing: min-max per sequence на Android vs `calc_embedding`.
4. XNNPACK CPU .pte vs Qualcomm — локализовать backend vs export.
5. Worst samples legs: [279, 831, 441, 559, 53].
