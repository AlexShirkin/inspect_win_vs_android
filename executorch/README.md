# ExecuTorch / Android experiments

Эта директория — для работы **на машине с настроенным ExecuTorch** (не на текущем Windows-компьютере).

Windows-часть уже подготовила эталоны в `results/windows/`. Ваша задача — воспроизвести и расширить сравнение на реальном `.pte` inference.

## Контекст проблемы

- **Задача boxing (mocopi body + legs):** на Windows FP32 валидация OK, на Android (ExecuTorch FP16) эмбеддинги сильно искажаются.
- **Задача avatar library (meta_quest):** на Android работало хорошо.
- Гипотеза: частичная поддержка FP16 операторов в ExecuTorch, неявные fp16↔fp32 casts, dtype mismatch при export.

## Что уже сделано на Windows

См. `results/windows/_summary/SUMMARY.md` после запуска `windows/run_all.py`.

Для каждого эксперимента сохранены:
- `embeddings_fp32.npz` — эталон
- `embeddings_fp16.npz` — симуляция PyTorch FP16
- `comparison_report.json` — метрики

Android reference embeddings (уже есть в `data/`):
- `android_upper_embeddings_fixed.npz` — body / upper
- `android_lower_embeddings_fixed.npz` — legs / lower

## Ваши задачи

### 1. Сверить Android ExecuTorch с Windows эталонами

```bash
# Скопировать repo + results/windows/ с Windows-машины
python executorch/compare_with_windows.py --experiment mocopi_body
python executorch/compare_with_windows.py --experiment mocopi_legs
```

Скрипт `compare_with_windows.py` (создать) должен:
1. Загрузить `results/windows/<exp>/embeddings_fp32.npz`
2. Загрузить Android `.npz` из `data/`
3. Посчитать те же метрики, что `windows/src/metrics.py` (можно импортировать)

### 2. Layer-wise debug (ETRecord + ETDump)

Для **boxing body** модели (худший кейс):

1. Export с `generate_etrecord=True`
2. Прогнать inference с debug → `ETDump`
3. Inspector API: `calculate_numeric_gap` — найти первый слой с большим расхождением

Сохранять в:
```
results/executorch/<experiment>/
  etrecord/
  etdump/
  layer_gaps.json
  layer_gaps_report.txt
```

### 3. Проверить dtype при export

Чеклист:
- [ ] Веса и вход трассировки одного dtype (оба fp16 или оба fp32)
- [ ] `nn.Embedding` / `BatchNorm1d` / `GRU` / `Softmax` — какой dtype в графе
- [ ] Есть ли `to(dtype=float32)` / `to(dtype=float16)` между слоями

### 4. XNNPACK CPU backend (контрольный эксперiment)

Экспортировать `.pte` для XNNPACK (CPU), прогнать те же 833 poses, сравнить с FP32.

Если XNNPACK ≈ FP32, а Qualcomm/HTP сильно отличается — проблема в backend, не в модели.

### 5. Meta Quest baseline

Прогнать `meta_quest_avatar` через ExecuTorch FP16 на Android (если есть `.pte`).

**Контекст:** meta_quest на устройстве использовал **одну** модель (без legs). Для сравнения смотрите также **mocopi_body** как upper-only аналог — не combined boxing pipeline.

## Архитектура модели

Файл: `mmd_nca_net_mobile.py`

Ключевые операторы (потенциально проблемные для FP16):
- `BatchNorm1d` (bn1, bn2, bn3, bn4, bn5)
- bidirectional `GRU` x4
- `SelfAttentiveEncoder`: Linear → Tanh → Softmax → bmm
- `F.relu` + Linear stack

Конфиги экспериментов — `windows/src/configs.py`.

## Формат данных

**Boxing (merged_three):** pickled dict в `arr_0`:
- `poses`: `[N, 30, J, 2]` — body J=12, legs J=11
- `pose2group`, `pose2person`, ...

**Avatar:** `poses`: `[530, 30, 13, 3]`

## Ожидаемые артефакты (git)

Коммитить:
- `executorch/scripts/*.py`
- `executorch/compare_with_windows.py`
- `results/executorch/**/layer_gaps*.json` (если не огромные)
- `results/executorch/**/comparison_report.*`

Не коммитить:
- большие `.pte`, `.etdump` бинарники (добавить в `.gitignore`)

## Связь с Windows-коллегой

| Windows (`windows/`) | ExecuTorch (`executorch/`) |
|---------------------|----------------------------|
| FP32 эталон | сверка с `.pte` output |
| FP16 PyTorch sim | оценка «ожидаемого» FP16 drift |
| метрики в JSON | layer-wise gap analysis |
| SUMMARY.md | итоговый ROOT_CAUSE.md |

После ваших экспериментов создайте `executorch/FINDINGS.md` с:
1. слой/оператор первого большого расхождения
2. отличие boxing vs meta_quest
3. рекомендация: fp32 export, selective fp16, или fix preprocessing
