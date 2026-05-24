# Windows PyTorch experiments (FP32 vs FP16 simulation)

Эта директория — для экспериментов на Windows **без ExecuTorch**.

## Окружение

```powershell
C:\Users\Alex\miniconda3\envs\smart_avatar\python.exe windows\run_all.py
```

- `smart_avatar` — PyTorch + numpy 1.x (инференс, сохранение эмбеддингов)
- FP16 симуляция: **CUDA** (`model.half()`); на CPU torch 1.10 не поддерживает Half для BN/GRU
- `smart_avatar_dev` — numpy 2.x (только чтение Android `.npz`, вызывается автоматически)

## Запуск

```powershell
# все эксперименты
python windows\run_all.py

# один эксперiment
python windows\run_all.py --experiment mocopi_body

# пересчитать только FP16 (FP32 уже сохранён)
python windows\run_all.py --skip-fp32
```

## Результаты

Пишутся в `results/windows/<experiment_name>/`:

| Файл | Описание |
|------|----------|
| `embeddings_fp32.npz` | эталон PyTorch FP32 |
| `embeddings_fp16.npz` | симуляция FP16 (`model.half()`, input half) |
| `comparison_report.txt` | человекочитаемые метрики |
| `comparison_report.json` | те же метрики в JSON |

Сводка по всем экспериментам: `results/windows/_summary/SUMMARY.md`

## Метрики сравнения

Для каждой пары (FP32↔FP16, FP32↔Android, FP16↔Android):

- per-element `|diff|` (mean, max, p95, p99)
- L2 distance и relative L2
- cosine similarity (mean, min, p01, p05)
- Spearman/Pearson корреляция **pairwise cosine/L2 матриц** (структура пространства)
- retrieval overlap@10 (насколько совпадают ближайшие соседи)
- group retrieval@10 по `pose2group` (delta для candidate)

## Предобработка

Используется та же min-max нормализация per-sequence, что в `mmd_nca_net_mobile.calc_embedding`.

## Структура

```
windows/
  run_all.py          # точка входа
  src/
    configs.py        # эксперименты и пути
    data_io.py        # загрузка npz
    model_utils.py    # FP32/FP16 инференс
    metrics.py        # метрики сравнения
```
