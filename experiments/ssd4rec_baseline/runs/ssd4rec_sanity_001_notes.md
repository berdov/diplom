# SSD4Rec sanity 001

## Цель

Проверить, что официальный SSD4Rec обучается 5 эпох на полном KuaiRand Protocol B, использует original variable-length/bidirectional path и считает только validation.

## Окружение

Требования upstream: Python `3.10.15`, CUDA `11.8`, PyTorch `2.1.1+cu118`, RecBole `1.2.0`, mamba-ssm `2.2.2`, causal-conv1d `1.4.0`.

Фактическое окружение: Python `3.10.14`, PyTorch `2.3.0+cu118`, CUDA `11.8`, RecBole `1.2.0`, mamba-ssm `2.2.2`, causal-conv1d `1.2.2.post1`, Triton `2.3.0`.

Путь окружения: `/home/daryumin/iberdov/diplom/envs/ssd4rec`; base prefix: `/home/daryumin/iberdov/diplom/envs/tim4rec`.

Известные расхождения:

- `python`: upstream `3.10.15`, фактически `3.10.14`.
- `torch`: upstream `2.1.1+cu118`, фактически `2.3.0+cu118`.
- `causal_conv1d`: upstream `1.4.0`, фактически `1.2.2.post1`.
- `triton`: upstream `2.1.0`, фактически `2.3.0`.
- `numpy`: upstream `1.26.3`, фактически `1.26.4`.
- `scipy`: upstream `1.14.1`, фактически `1.13.1`.
- `pandas`: upstream `2.2.3`, фактически `2.2.2`.
- `pyyaml`: upstream `6.0.2`, фактически `6.0.1`.
- `tqdm`: upstream `4.67.1`, фактически `4.66.4`.
- `transformers`: upstream `4.46.3`, фактически `4.40.2`.

## Данные и Protocol B

- Dataset: полный KuaiRand Protocol B, без subset и без sampled split.
- Fingerprint: `23951 users / 7111 items / 1134420 interactions`.
- Длины историй в Protocol B min/median/mean/max: `5 / 34.0 / 47.3642 / 806`.
- `MAX_ITEM_LIST_LENGTH=50` есть в config, но при upstream `var_len=True` не является active truncation cap.
- Split: chronological leave-one-out; validation target - предпоследняя interaction.
- Test metrics не считались.
- Evaluation: `full_7111_items`; internal score tensor включает padding item `0`, он маскируется.

## Конфигурация original SSD4Rec

- `hidden_size=256`
- `num_layers=2`
- `d_state=64`
- `d_conv=4`
- `expand=2`
- `headdim=16`
- `var_len=True`
- `maskratio=0.2`
- `learning_rate=0.001`
- `train_batch_size=1024`
- `eval_batch_size=2048`
- `seed=2024`

## Variable-length mechanism

- Targets в первом train batch: `1024`.
- Flat sequence tokens: `46493`.
- Длина последовательностей min/median/mean/max: `1 / 29.0 / 45.4033 / 510`.
- `cum_item_length_shape=[1024]`.
- `item_idx_shape=[46493]`.
- Sequence registers присутствуют: `True`.
- `flip_index_is_permutation=True`.

## Bidirectional SSD

- BiSSD layers: `2`.
- Forward direction активен: `True`.
- Backward/reversed direction активен: `True`.
- Вызовы Mamba по слоям в первом forward: `{'0': 2, '1': 2}`.
- Один и тот же Mamba2 module используется для обоих направлений: `True`.
- Gradients finite: `True`.

## Обучение

| epoch | loss | HR@10 | HR@20 | HR@50 | NDCG@10 | NDCG@20 | NDCG@50 | time |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 7.166289 | 0.0851 | 0.1379 | 0.2403 | 0.0480 | 0.0613 | 0.0815 | 241.24 |
| 2 | 6.672622 | 0.0912 | 0.1455 | 0.2590 | 0.0513 | 0.0650 | 0.0874 | 117.16 |
| 3 | 6.569422 | 0.0942 | 0.1472 | 0.2686 | 0.0526 | 0.0660 | 0.0899 | 117.28 |
| 4 | 6.502672 | 0.0984 | 0.1555 | 0.2807 | 0.0546 | 0.0689 | 0.0936 | 118.10 |
| 5 | 6.448484 | 0.1008 | 0.1615 | 0.2857 | 0.0559 | 0.0712 | 0.0956 | 118.00 |

Полные validation metrics:

| epoch | HR@5 | HR@10 | HR@20 | HR@50 | Recall@5 | Recall@10 | Recall@20 | Recall@50 | NDCG@5 | NDCG@10 | NDCG@20 | NDCG@50 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.0547 | 0.0851 | 0.1379 | 0.2403 | 0.0547 | 0.0851 | 0.1379 | 0.2403 | 0.0383 | 0.0480 | 0.0613 | 0.0815 |
| 2 | 0.0562 | 0.0912 | 0.1455 | 0.2590 | 0.0562 | 0.0912 | 0.1455 | 0.2590 | 0.0402 | 0.0513 | 0.0650 | 0.0874 |
| 3 | 0.0569 | 0.0942 | 0.1472 | 0.2686 | 0.0569 | 0.0942 | 0.1472 | 0.2686 | 0.0407 | 0.0526 | 0.0660 | 0.0899 |
| 4 | 0.0625 | 0.0984 | 0.1555 | 0.2807 | 0.0625 | 0.0984 | 0.1555 | 0.2807 | 0.0430 | 0.0546 | 0.0689 | 0.0936 |
| 5 | 0.0638 | 0.1008 | 0.1615 | 0.2857 | 0.0638 | 0.1008 | 0.1615 | 0.2857 | 0.0440 | 0.0559 | 0.0712 | 0.0956 |

## Лучшая validation эпоха

- Лучшая epoch: `5`.
- Лучший validation `NDCG@10`: `0.0559`.
- Лучший validation `HR@10`: `0.1008`.

## Сравнение с ориентирами

| Источник | Split | HR@10 | HR@20 | NDCG@10 | NDCG@20 |
| --- | --- | ---: | ---: | ---: | ---: |
| Random full-ranking | validation | 0.0011 | n/a | 0.0004 | n/a |
| MostPopular full-ranking | validation | 0.0300 | n/a | 0.0168 | n/a |
| XGBoost `ltr_xgb_002` | validation | 0.0309 | n/a | 0.0150 | n/a |
| TiM4Rec `tim4rec_001` best | validation | 0.1086 | n/a | 0.0593 | n/a |
| SSD4Rec sanity best | validation | 0.1008 | 0.1615 | 0.0559 | 0.0712 |
| SSD4Rec paper v2 | paper reported | 0.1075 | 0.1731 | 0.0593 | 0.0757 |

## Ресурсы

- Slurm job: `4264406`.
- Partition: `test`.
- Node: `cn-046`.
- GPU: `NVIDIA A100-SXM4-80GB`.
- Requested GPUs: `1`; raw `SLURM_JOB_GPUS=4`, `CUDA_VISIBLE_DEVICES=0`.
- Общее время: `740.58` sec.
- Среднее время эпохи: `142.36` sec.
- TimeLimit: `00:30:00`.
- Состояние / exit code: `COMPLETED / 0:0`.
- AllocTRES: `billing=4,cpu=4,gres/gpu:a100=1,gres/gpu=1,node=1`.
- MaxRSS: `2993M`.
- Пик VRAM allocated: `3753577472` bytes.
- Пик VRAM reserved: `11087642624` bytes.

## Проблемы и исправления

- Upstream `custom_utils.py` требует runtime shim для `np.float` на NumPy >= 1.24.
- Upstream `custom_utils.py` вызывает `getLogger()` без import; wrapper добавляет `logging.getLogger`.
- Full-sort validation следует upstream semantics: маскируется item id `0`, seen history items явно не маскируются.

## Решение о полном запуске

- Pipeline готов к полному SSD4Rec run: `True`.
- Модель действительно original SSD4Rec: `True`.
- Окружение достаточно воспроизводимо для sanity: `True`.
- Exact upstream environment все еще нужен перед финальной заявкой на reproduction: `True`.
- Loss убывает за 5 эпох: `True`.
- Validation NDCG@10 растет за 5 эпох: `True`.
- Оценка wall time для 300 эпох в темпе sanity: `12.34` hours.
- Рекомендуемый GPU: `A100/type_e`.
