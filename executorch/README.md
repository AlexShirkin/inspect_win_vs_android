# ExecuTorch / Android experiments

Работа на машине с Docker + ExecuTorch. Windows-эталоны уже в `results/windows/`.

## Быстрый старт (уже снятые Android embeddings)

Из корня `inspect_win_vs_android/`:

```bash
python executorch/compare_with_windows.py --all
python executorch/scripts/make_summary.py
```

## Docker: ваш текущий билд

Монтирование репозитория (как в основном README):

```bash
# хост: pte_qualcomm -> /workspace
docker exec -it executorch_container bash
cd /workspace/workspace/install
./build_mmd_nca_net.sh   # веса/joints вручную в model.py и mmd_nca_net.py
```

Пути внутри контейнера:

| Переменная | Значение |
|------------|----------|
| `WORKSPACE` | `/workspace/workspace` |
| `EXECUTORCH_ROOT` | `$WORKSPACE/executorch` |
| `.pte` после build | `$EXECUTORCH_ROOT/deeplab_v3/<имя из pte_filename>.pte` |

## Шаг 1 — сверка Android (готово)

```bash
cd /workspace/inspect_win_vs_android
python executorch/compare_with_windows.py --experiment mocopi_body
python executorch/compare_with_windows.py --experiment mocopi_legs
```

## Шаг 2 — PTE batch на x86 QNN runner

Проверяет **export + QNN backend** на тех же 833 poses (с нормализацией как на Windows).

```bash
cd /workspace/inspect_win_vs_android
chmod +x executorch/run_qnn_batch.sh

# body — подставьте свой .pte после build_mmd_nca_net.sh
./executorch/run_qnn_batch.sh mocopi_body \
  /workspace/workspace/executorch/deeplab_v3/mmd_nca_net_qualcomm_12_mocopi_body_front_proj_70999.pte

# legs — переключите joints/weights в model.py, пересоберите, затем:
./executorch/run_qnn_batch.sh mocopi_legs \
  /workspace/workspace/executorch/deeplab_v3/mmd_nca_net_qualcomm_11_mocopi_legs_front_proj_15999.pte
```

Артефакты: `results/executorch/<exp>/pte_run/` + `pte_comparison_report.*`

Для быстрой проверки: добавьте третий аргумент `10` (max 10 poses).

## Шаг 3 — dtypes при export

После правки `examples/models/mmd_nca_net/model.py`:

```bash
cd /workspace/workspace/executorch
conda run -n executorch python \
  /workspace/inspect_win_vs_android/executorch/scripts/inspect_export_dtypes.py \
  --executorch-root /workspace/workspace/executorch
```

## Шаг 4 — XNNPACK (контроль backend)

```bash
conda run -n executorch python \
  /workspace/inspect_win_vs_android/executorch/scripts/export_xnnpack_pte.py \
  --executorch-root /workspace/workspace/executorch \
  --out-dir /workspace/workspace/executorch/deeplab_v3 \
  --name mmd_nca_body_xnnpack_fp32
```

Запуск (путь к runner зависит от вашей cmake-сборки, часто `build-x86/backends/xnnpack/xnn_executor_runner`):

```bash
# те же input_list + .raw из pte_run, другой --model_path на xnnpack .pte
```

Если XNNPACK ≈ FP32, а QNN/Android далеко — проблема в Qualcomm FP16/backend.

## Шаг 5 — Layer-wise ETDump

1. В `build_executorch_binary(...)` передать `dump_intermediate_outputs=True` (временно в `mmd_nca_net.py`).
2. Пересобрать `.pte`.
3. Прогнать runner с `--dump_intermediate_outputs --etdump_path etdump.etdp`.
4. Анализ:

```bash
conda run -n executorch python \
  /workspace/inspect_win_vs_android/executorch/scripts/analyze_etdump.py \
  --etdump results/executorch/mocopi_legs/etdump/etdump.etdp \
  --etrecord /path/to/etrecord.bin \
  --out-dir results/executorch/mocopi_legs
```

Для ETRecord можно ориентироваться на `examples/qualcomm/scripts/export_example.py` (`generate_etrecord=True`).

## Шаг 6 — итог

```bash
python executorch/scripts/make_summary.py
# обновить executorch/FINDINGS.md по pte_comparison + layer_gaps
```

## Артефакты

| Путь | Содержимое |
|------|------------|
| `results/executorch/*/comparison_report.*` | Android vs Windows |
| `results/executorch/*/pte_comparison_report.*` | PTE runner vs Windows |
| `results/executorch/*/layer_gaps*` | ETDump summary |
| `results/executorch/_summary/SUMMARY.md` | сводка |

Не коммитить: `*.pte`, `*.etdp`, большие `pte_run/*.raw`.

## Связь с Windows

| Windows | ExecuTorch |
|---------|------------|
| `embeddings_fp32.npz` | эталон для PTE/Android |
| `FINDINGS.md` | `executorch/FINDINGS.md` |
