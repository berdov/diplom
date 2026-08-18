# Аудит TiM4Rec для KuaiRand Protocol B

## Источники

Основные источники:

- Статья: Fan et al., "TiM4Rec: An Efficient Sequential Recommendation Model Based on Time-Aware Structured State Space Duality Model", arXiv 2409.16182, https://arxiv.org/abs/2409.16182.
- Официальный репозиторий: https://github.com/AlwaysFHao/TiM4Rec.

Проверенный снимок upstream:

- commit: `8d4a6cea6a035c249a7a13999166ba41e8924abe`;
- дата коммита: `2025-08-23T11:11:42+08:00`;
- сообщение: `update official publication information in readme`;
- license: MIT License, copyright 2024 AlwaysFHao.

Структура official repo на этом snapshot: `tim4rec.py`, `ssd.py`, `run.py`, `test.py`, `environment.yaml`, `config/*.yaml`, `baseline/*`, пустые placeholder-каталоги `dataset/`, `log/`, `log_tensorboard/`, `saved/`. В Git не входят датасеты, checkpoints и training logs.

Локальная копия в `experiments/tim4rec_baseline/upstream/` взята из этого snapshot; при добавлении в наш репозиторий нормализованы только хвостовые пробелы, без изменения кода, конфигов и лицензии по смыслу.

## Ориентиры из статьи для KuaiRand

Статья описывает KuaiRand после 5-core filtering и сортировки interactions по timestamp. Опубликованный fingerprint:

| Поле | Значение |
| --- | ---: |
| users | 23 951 |
| items | 7 111 |
| interactions | 1 134 420 |
| average sequence length | 47.4 |
| max sequence length | 809 |

Опубликованные результаты TiM4Rec на KuaiRand:

| Metric | @10 | @20 | @50 |
| --- | ---: | ---: | ---: |
| Recall/Hit | 0.1109 | 0.1774 | 0.3202 |
| NDCG | 0.0611 | 0.0779 | 0.1060 |
| MRR | 0.0463 | 0.0508 | 0.0552 |

В leave-one-out evaluation с одним целевым item на пользователя `Hit@K` и `Recall@K` численно совпадают. Поэтому RecBole metric `Hit` сопоставим с paper `Recall` для этого протокола.

## Официальный KuaiRand config vs статья

Файл upstream: `config/config4kuai_64d.yaml`.

| Параметр | Статья | Официальный KuaiRand config | Статус |
| --- | --- | --- | --- |
| hidden size | 64 | 64 | совпадает |
| layers depth | 2 | 2 | совпадает |
| dropout | 0.2 for KuaiRand | 0.2 | совпадает |
| d_state | 32 | 32 | совпадает |
| d_conv | 4 | 4 | совпадает |
| expand | 2 | 2 | совпадает |
| train batch | 2048 | 2048 | совпадает |
| eval batch | 4096 | 4096 | совпадает |
| max sequence length | 50 for KuaiRand | 50 | совпадает |
| topk | 10, 20, 50 | 10, 20, 50 | совпадает |
| optimizer | Adam | adam | совпадает |
| learning rate | 0.01 | 0.001 | расхождение |
| metrics | Recall, NDCG, MRR | Hit, NDCG, MRR | совместимо для LOO |
| eval_args | sequential chronological LOO implied | не задано явно | важен default RecBole |
| `is_time` | full model is time-aware | `False` | критическое расхождение |

История upstream для `config/config4kuai_64d.yaml` содержит только первичное добавление (`102bb48 Initialize Repository`); `is_time: False` присутствовал уже в первой проверенной версии файла.

RecBole 1.2.0 sequential quick-start defaults задают:

```yaml
eval_args:
  split: {'LS': 'valid_and_test'}
  order: TO
  mode: full
repeatable: True
```

`group_by: user` приходит из общего default-конфига RecBole. В `config_kuairand.yaml` эти eval settings заданы явно, чтобы не зависеть от implicit defaults.

## Вывод по `is_time`

`is_time` в official code не является декоративным флагом:

- в `tim4rec.py` при `is_time=True` создается `layer_norm_time`, рассчитываются timestamp differences и они передаются в TiSSD layers;
- при `is_time=False` `time_diff=None`;
- в `ssd.py` при `is_time=True` создаются time-aware convolution/MLP/gate modules и используется `final_dt = dt * time_dt`;
- при `is_time=False` используется `final_dt = dt`, то есть time-aware path полностью выключен.

Статья позиционирует TiM4Rec как time-aware SSD модель. В ablation section вариант `w/o Time` описан как удаление time-aware algorithm и соответствует SSD-like варианту, а не полной TiM4Rec.

Рабочее решение для дипломной ветки:

- основной config `experiments/tim4rec_baseline/config_kuairand.yaml` ставит `is_time: True`, потому что это проверяет заявленный time-aware TiM4Rec;
- оригинальный upstream KuaiRand config с `is_time: False` сохранен без изменений в `experiments/tim4rec_baseline/upstream/config/config4kuai_64d.yaml`;
- для честного отчета после полного обучения нужно либо отдельно прогнать `is_time=False` как контрольный official-config/SSD-like run, либо явно написать, что основной run сознательно исправляет противоречие official config vs статья.

Learning rate пока оставлен `0.001` из official executable config. Статья указывает `0.01`; это второе расхождение, которое нужно закрывать отдельным sensitivity run, если метрики не приблизятся к опубликованным.

## Protocol B dataset

Используется уже подготовленный в проекте KuaiRand Protocol B:

- manifest: `outputs/data/protocol_b_manifest.json`;
- корень данных на кластере: `/home/daryumin/iberdov/diplom/data/processed/protocol_b`;
- файл RecBole: `/home/daryumin/iberdov/diplom/data/processed/protocol_b/recbole/kuairand/kuairand.inter`;
- проверенный заголовок: `user_id:token item_id:token timestamp:float`;
- проверенное число строк на cHARISMa: 1 134 421 вместе с заголовком;
- filtered fingerprint: 23 951 users, 7 111 items, 1 134 420 interactions;
- split: chronological leave-one-out, validation - предпоследний interaction, test - последний interaction;
- tie-break: `user_id`, `timestamp`, `source_row_id`;
- `MAX_ITEM_LIST_LENGTH: 50`.

Этот fingerprint совпадает с paper KuaiRand table, поэтому Protocol B остается подходящим для TiM4Rec/SSD4Rec comparison.

## Среда и GPU

Официальный README требует:

- CUDA 11.8;
- Python 3.10.14;
- PyTorch 2.3.0;
- RecBole 1.2.0;
- mamba-ssm 2.2.2;
- causal-conv1d 1.2.2 optional;
- numpy 1.26.4.

На cHARISMa доступны GPU partitions:

- `gpu-ef-quick`, лимит 3 часа, узлы с A100/H100/H200;
- `rocky`, лимит 30 дней, включает V100/A100/H100/H200;
- `test`, лимит 30 минут.

Для smoke test выбран `gpu-ef-quick` с `--constraint=type_e` и `--gres=gpu:1`, чтобы запускаться на E nodes и не занимать long partition. На этих partitions cHARISMa не принимает обычный `--mem=48G`; проектные scripts используют `--mem=0`, этот же режим применен здесь. Для full training планируется `rocky` или отдельная long GPU partition после smoke.

## Статус smoke test

Smoke test выполнен:

- Slurm job: `4260040`;
- partition: `test`, command-line override для короткой проверки; Slurm script по умолчанию остается `gpu-ef-quick`;
- constraint: `type_e`;
- node: `cn-046`;
- GPU: `NVIDIA A100-SXM4-80GB`;
- CUDA/PyTorch: CUDA 11.8, PyTorch `2.3.0+cu118`;
- Python: `3.10.14`;
- RecBole: `1.2.0`;
- mamba-ssm: `2.2.2`;
- causal-conv1d: `1.2.2.post1`;
- status: `COMPLETED`, exit code `0:0`, elapsed `00:01:18`;
- compact result: `experiments/tim4rec_baseline/runs/smoke_20260818T132855Z.json`.

Проверено:

- import `torch`;
- import `recbole`;
- import `mamba_ssm`;
- import upstream `TiM4Rec`;
- чтение `kuairand.inter`;
- создание RecBole dataset/splits;
- model init на GPU;
- один `forward` на настоящем train batch;
- compact JSON result в `experiments/tim4rec_baseline/runs/`.

Результат forward:

- `item_seq_shape`: `[2, 50]`;
- `timestamp_seq_shape`: `[2, 50]`;
- `seq_output_shape`: `[2, 64]`;
- `seq_output_dtype`: `torch.float32`;
- `seq_output_device`: `cuda:0`.

RecBole после sequential augmentation показывает `inter_num_after_recbole_processing = 1 110 469`, что равно `1 134 420 - 23 951`: первый interaction каждого пользователя не превращается в обучающий sequential пример, потому что у него нет истории. Это не изменение исходного Protocol B fingerprint.

Полное обучение на этой стадии не запускается.

## План полного запуска

1. Прогнать smoke job и зафиксировать JSON с node/GPU/CUDA/PyTorch/RecBole/mamba_ssm.
2. Запустить полное обучение `is_time=True`, `learning_rate=0.001`, `topk=[5,10,20,50]`.
3. Если полные метрики сильно расходятся со статьей, запустить проверку `learning_rate=0.01`.
4. Отдельно запустить контрольный `is_time=False` на official KuaiRand config semantics и описать как ablation/control, а не как full TiM4Rec.
5. Сравнивать со статьей по @10/@20/@50; @5 использовать только для дипломной таблицы, потому что статья @5 не публикует.
