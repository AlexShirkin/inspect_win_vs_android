# Как сделать следующие шаги (практика)

Все команды — **внутри Docker**, mount `pte_qualcomm` → `/workspace`.

```bash
docker exec -it executorch_container bash
```

Переключение **body / legs** — вручную в двух файлах, затем `./build_mmd_nca_net.sh`:

| Файл | body | legs |
|------|------|------|
| `executorch/examples/models/mmd_nca_net/model.py` | `joints_num=12`, weights `*_body_*_77999.pth`, input `30×12×2` | `joints_num=11`, weights `*_legs_*_15999.pth`, input `30×11×2` |
| `executorch/examples/qualcomm/scripts/mmd_nca_net.py` | `ones(30,12,2)`, `pte_filename=...body...77999` | `ones(30,11,2)`, `pte_filename=...legs...15999` |

---

## A. Сверка после любого нового .pte

```bash
cd /workspace/inspect_win_vs_android

./executorch/run_qnn_batch.sh mocopi_legs \
  /workspace/workspace/executorch/deeplab_v3/mmd_nca_net_qualcomm_11_mocopi_legs_front_proj_15999.pte

python executorch/scripts/make_summary.py
```

Смотрите `results/executorch/mocopi_legs/pte_comparison_report.txt` — цель для FP32 export: **rel_l2 ближе к 0.4%** (как PyTorch FP16-sim), а не 3.5%.

---

## B. Export QNN в FP32 (главная проверка гипотезы)

Сейчас FP16 включается в `build_executorch_binary`:

```301:303:workspace/executorch/examples/qualcomm/utils.py
    backend_options = generate_htp_compiler_spec(
        use_fp16=False if quant_dtype else True
    )
```

**Временно** замените на:

```python
    backend_options = generate_htp_compiler_spec(
        use_fp16=False
    )
```

Дальше:

```bash
cd /workspace/workspace/install
./build_mmd_nca_net.sh
```

Новый `.pte` — другое имя в `pte_filename`, например `..._legs_fp32.pte`.

Снова `run_qnn_batch.sh` и сравните rel_l2. Если стало ≈ FP32 / PyTorch — **виноват HTP FP16**, можно оставить FP32 на проде или искать selective FP16.

Верните `use_fp16=True` после эксперимента, если нужен текущий прод.

---

## C. XNNPACK control (CPU, без Qualcomm)

Цель: если XNNPACK ≈ Windows FP32, а QNN нет — проблема в QNN backend.

### C.1 Export

В `model.py` выставить нужную модель (legs/body), затем:

```bash
cd /workspace/workspace/executorch
conda run -n executorch python \
  /workspace/inspect_win_vs_android/executorch/scripts/export_xnnpack_pte.py \
  --executorch-root /workspace/workspace/executorch \
  --out-dir /workspace/workspace/executorch/deeplab_v3 \
  --name mmd_nca_legs_xnnpack_fp32
```

### C.2 Runner

Нужен `xnn_executor_runner` (собирается с `EXECUTORCH_BUILD_XNNPACK=ON` в install_aot). Путь часто:

`$EXECUTORCH_ROOT/build-x86/backends/xnnpack/xnn_executor_runner`

Подготовить inputs (те же 833 poses):

```bash
cd /workspace/inspect_win_vs_android
python executorch/scripts/prepare_runner_inputs.py \
  --experiment mocopi_legs \
  --out-dir results/executorch/mocopi_legs/xnn_run

cd results/executorch/mocopi_legs/xnn_run
$EXECUTORCH_ROOT/build-x86/backends/xnnpack/xnn_executor_runner \
  --model_path /workspace/workspace/executorch/deeplab_v3/mmd_nca_legs_xnnpack_fp32.pte
```

(У `xnn_executor_runner` те же флаги `--input_list_path` / `--output_folder_path`, что у QNN.)

Собрать embeddings и сравнить:

```bash
python executorch/scripts/collect_pte_outputs.py \
  --outputs-dir results/executorch/mocopi_legs/xnn_run/outputs \
  --out-npz results/executorch/mocopi_legs/xnn_run/embeddings_xnn.npz

python executorch/scripts/compare_pte_with_windows.py \
  --experiment mocopi_legs \
  --pte-embeddings results/executorch/mocopi_legs/xnn_run/embeddings_xnn.npz \
  --label "XNNPACK fp32"
```

---

## D. ETDump на worst samples (legs)

### D.1 Export с debug

В `mmd_nca_net.py` в вызов `build_executorch_binary` добавить:

```python
    build_executorch_binary(
        ...
        dump_intermediate_outputs=True,
    )
```

Пересобрать legs `.pte` (можно `--compile_only` без полного cmake).

Опционально ETRecord — отдельный скрипт Qualcomm `export_example.py` или `generate_etrecord` по образцу в `examples/qualcomm/scripts/export_example.py`.

### D.2 Прогон только worst poses

```bash
cd /workspace/inspect_win_vs_android
WORK=results/executorch/mocopi_legs/etdump_run
mkdir -p "$WORK/outputs"

python executorch/scripts/prepare_runner_inputs.py \
  --experiment mocopi_legs \
  --out-dir "$WORK" \
  --indices 279,831,441,559,53

export LD_LIBRARY_PATH=${QNN_SDK_ROOT}/lib/x86_64-linux-clang:${BUILD_X64}/lib
cd "$WORK"
$BUILD_X64/examples/qualcomm/executor_runner/qnn_executor_runner \
  --model_path /workspace/workspace/executorch/deeplab_v3/mmd_nca_net_qualcomm_11_mocopi_legs_front_proj_15999.pte \
  --input_list_path input_list.txt \
  --output_folder_path outputs \
  --dump_intermediate_outputs \
  --etdump_path etdump.etdp \
  --debug_output_path debug_output.bin
```

### D.3 Разбор

```bash
conda run -n executorch python \
  /workspace/inspect_win_vs_android/executorch/scripts/analyze_etdump.py \
  --etdump /workspace/inspect_win_vs_android/results/executorch/mocopi_legs/etdump_run/etdump.etdp \
  --out-dir /workspace/inspect_win_vs_android/results/executorch/mocopi_legs

# детальнее — штатный inspector Qualcomm:
cd /workspace/workspace/executorch
conda run -n executorch python examples/qualcomm/qnn_intermediate_output_inspector.py \
  --etdump_path /workspace/inspect_win_vs_android/results/executorch/mocopi_legs/etdump_run/etdump.etdp
```

Ищите первый слой, где debug tensor сильно расходится с PyTorch (GRU / BatchNorm / Softmax — типичные кандидаты).

---

## E. Selective FP16 (если full FP32 слишком медленный)

1. Соберите FP32 (шаг B), сохраните `print_tabular()` графа из лога export.
2. Повторите export с `use_fp16=True` и попробуйте **не делегировать** подозрительные ops:

```bash
python -m examples.qualcomm.scripts.mmd_nca_net ... --compile_only \
  --skip_delegate_node_ops "aten._native_batch_norm_legit.no_stats,aten.gru.input"
```

Точные имена ops — из вывода графа; подбирается итеративно. После каждого варианта — `run_qnn_batch.sh`.

---

## F. Выровнять checkpoint с Windows

Windows FP32 эталон: `weights/sequential_250_128_mocopi_body_2d_proj_30_12_77999.pth`.

В `model.py` уже `77999` — хорошо. Для legs эталон Windows: `15999.pth` — должен совпадать с Android/PTE.

Пересоберите `.pte` и перезапустите `run_qnn_batch.sh`.

---

## G. На Android (после фикса export)

1. Залить новый `.pte` в приложение.
2. Снять embeddings на 833 poses → `android_*_embeddings.npz` в `inspect_win_vs_android/data/`.
3. `python executorch/compare_with_windows.py --experiment mocopi_legs`.

---

## Порядок работ (рекомендуемый)

1. **B** — FP32 QNN export + batch (быстро подтвердить гипотезу).
2. **C** — XNNPACK (отделить backend от export).
3. **D** — ETDump на legs worst (найти слой).
4. **E** — selective FP16, если нужен компромисс скорость/качество.
