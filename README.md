# inspect_win_vs_android

Сравнение эмбеддингов metric learning моделей: **Windows PyTorch FP32** vs **FP16 simulation** vs **Android ExecuTorch**.

## Структура репозитория

```
inspect_win_vs_android/
├── mmd_nca_net_mobile.py      # архитектура модели
├── data/                      # датасеты и Android embeddings (не в git по умолчанию)
├── weights/                   # веса моделей
├── windows/                   # ← эксперименты на Windows (PyTorch)
│   ├── run_all.py
│   └── src/
├── executorch/                # ← эксперименты на машине с ExecuTorch
│   ├── README.md              # задание для коллеги
│   └── compare_with_windows.py
└── results/
    ├── windows/               # отчёты + embeddings (npz в .gitignore)
    └── executorch/            # layer-wise debug, pte comparisons
```

## Быстрый старт (Windows)

```powershell
C:\Users\Alex\miniconda3\envs\smart_avatar\python.exe windows\run_all.py
```

Подробности: [windows/README.md](windows/README.md)

## ExecuTorch (другой компьютер)

После синхронизации `results/windows/` через git:

```bash
python executorch/compare_with_windows.py --all
```

Подробности: [executorch/README.md](executorch/README.md)

## Эксперименты

| Имя | Задача | Модель | Данные |
|-----|--------|--------|--------|
| `mocopi_body` | boxing (upper) | 30×12×2 | 833 poses |
| `mocopi_legs` | boxing (lower) | 30×11×2 | 833 poses |
| `meta_quest_avatar` | motion library (single model, no legs) | 30×13×3 | 530 poses |
