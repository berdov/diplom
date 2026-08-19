# Аудит SSD4Rec для KuaiRand Protocol B

## Источники

Основные источники:

- Статья: Qu et al., "SSD4Rec: A Structured State Space Duality Model for Efficient Sequential Recommendation", arXiv 2409.01192, https://arxiv.org/abs/2409.01192.
- HTML v1, использованный ранее в Protocol B manifest: https://arxiv.org/html/2409.01192v1.
- HTML v2, актуальная версия arXiv на момент аудита: https://arxiv.org/html/2409.01192.
- Официальный репозиторий: https://github.com/ZhangYifeng1995/SSD4Rec.

Проверенный снимок upstream:

- commit: `bdbfe5193f3a6697bb6ee0699ab43386d80c6198`;
- дата коммита: `2025-01-16T08:48:49Z`;
- сообщение: `update datasets`;
- license: MIT License, copyright 2025 Zhang Yifeng.

Структура official repo на этом snapshot: `ssd4rec.py`, `custom_utils.py`,
`custom_trainer.py`, `main.py`, `config.yaml`, `environment.yaml`, `LICENSE`,
`README.md`, а также processed datasets в `dataset/*/*.inter`. В наш Git
скопирован только код/конфиги/license/readme; upstream datasets не коммитятся.

## Ориентиры из статьи для KuaiRand

Статья описывает 5-core filtering, сортировку interactions по timestamp и
leave-one-out split по SASRec.

Опубликованный fingerprint:

| Поле | Значение |
| --- | ---: |
| users | 23 951 |
| items | 7 111 |
| interactions | 1 134 420 |
| average sequence length | 47.4 |
| max sequence length | 809 |
| sparsity | 99.33% |

Актуальная arXiv v2 Table 4 для KuaiRand:

| Metric | SASRec* | SSD4Rec |
| --- | ---: | ---: |
| NDCG@10 | 0.0567 | 0.0593 |
| NDCG@20 | 0.0733 | 0.0757 |
| MRR@10 | 0.0426 | 0.0448 |
| MRR@20 | 0.0471 | 0.0493 |
| HR@10 | 0.1040 | 0.1075 |
| HR@20 | 0.1705 | 0.1731 |

Ранее в проекте для literature comparison использовалась arXiv v1 Table 4:

| Metric | SSD4Rec v1 |
| --- | ---: |
| NDCG@10 | 0.0602 |
| NDCG@20 | 0.0759 |
| HR@10 | 0.1076 |
| HR@20 | 0.1704 |

Для дальнейших отчетов нужно явно указывать версию статьи. Кодовый upstream
commit от `2025-01-16` ближе к arXiv v2, поэтому основной audit baseline
ориентируется на v2, а v1 оставлен как историческая ссылка Protocol B.

## Архитектура official SSD4Rec

Модель реализована в `upstream/ssd4rec.py` как `SSD4Rec(SequentialRecommender)`.
Основные элементы:

- `item_embedding` размера `n_items x hidden_size`; internal id `0` используется
  как padding/mask token;
- optional embedding normalization: dropout + LayerNorm, если `norm_embedding=True`;
- стек из `num_layers` блоков `BiSSDLayer`;
- каждый `BiSSDLayer` содержит один `Mamba2` SSD block, LayerNorm, dropout и FFN;
- forward direction обрабатывает concatenated sequence с `seq_idx=item_idx`;
- backward direction сначала переставляет токены по `flip_index`, затем снова
  вызывает тот же `Mamba2`;
- объединение: `forward_hidden_state + beta * backward_hidden_state + item_emb`,
  затем LayerNorm/dropout/FFN;
- prediction берет последний скрытый вектор каждого user segment по
  `cum_item_length - 1` и считает logits через dot product с `item_embedding.weight`;
- loss: full softmax cross-entropy по всем item ids.

Датасет и dataloaders находятся в `upstream/custom_utils.py`.

Ключевая механика:

- `SSD4RecDataset.data_augmentation()` создает history-target пары для каждого
  пользователя после сортировки по `user_id` и `timestamp`;
- при `var_len=True` history не обрезается по `MAX_ITEM_LIST_LENGTH`, поэтому
  длинные пользователи сохраняют всю историю до target;
- train `collate_fn` склеивает все histories batch в один плоский `item_id_list`;
- `item_idx` является segment register: для каждого токена хранит номер пользователя
  внутри batch;
- `cum_item_length` хранит границы сегментов;
- `flip_index` строит user-wise reverse order без перемешивания пользователей;
- train masking делает `item_id_list = item_id_list * mask_index`, где `0` является
  mask token; при `maskratio=0.2` примерно 20% history tokens заменяются на `0`;
- eval dataloader использует ту же variable-length склейку, но без masking.

Trainer находится в `upstream/custom_trainer.py`.

- `_train_epoch` ожидает tuple custom dataloader и вызывает `model.calculate_loss`.
- `_full_sort_batch_eval` считает scores по всем items и ставит `scores[:, 0] = -inf`.
- Seen history items в upstream custom eval явно не маскируются, потому что custom
  eval loader не передает `history_index`.

## Official SSD4Rec vs SSD4Rec* из TiM4Rec

Это разные объекты сравнения.

Official SSD4Rec:

- источник: `ZhangYifeng1995/SSD4Rec`;
- hidden size `256`;
- `d_state=64`, `d_conv=4`, `expand=2`, `headdim=16`;
- использует `var_len=True`, segment registers, masking и bidirectional SSD;
- published KuaiRand v2: `HR@10=0.1075`, `NDCG@10=0.0593`;
- training batch `1024`, eval batch `2048`, lr `0.001`.

`SSD4Rec*` в статье TiM4Rec:

- источник: собственная реплика авторов TiM4Rec;
- hidden size `64`;
- `d_state=32`, `d_conv=4`, `expand=2`;
- авторы TiM4Rec прямо пишут, что эта реплика не включает variable-length
  sequences и bidirectional SSD из official SSD4Rec;
- published KuaiRand: `R@10=0.1055`, `R@20=0.1717`, `R@50=0.3088`,
  `N@10=0.0588`, `N@20=0.0754`, `N@50=0.1024`;
- TiM4Rec Table 2 в HTML v2 содержит очевидную опечатку `0.5880` для `N@10`
  у `SSD4Rec*`; по масштабу соседних значений и по тексту это `0.0588`.

Следствие для диплома: официальный SSD4Rec baseline должен запускаться из
`experiments/ssd4rec_baseline`, а `SSD4Rec*` из TiM4Rec нельзя выдавать за
official SSD4Rec reproduction.

## Config для KuaiRand

Файл `experiments/ssd4rec_baseline/config_kuairand.yaml` переносит official
KuaiRand settings из upstream `config.yaml` и делает неявные RecBole defaults
явными.

| Параметр | Значение |
| --- | --- |
| dataset | `kuairand` |
| data_path | `/home/daryumin/iberdov/diplom/data/processed/protocol_b/recbole` |
| hidden_size | `256` |
| d_state | `64` |
| d_conv | `4` |
| expand | `2` |
| headdim | `16` |
| beta | `0.1` |
| maskratio | `0.2` |
| num_layers | `2` |
| dropout_prob | `0.2` |
| norm_embedding | `True` |
| var_len | `True` |
| MAX_ITEM_LIST_LENGTH | `50` |
| train_batch_size | `1024` |
| eval_batch_size | `2048` |
| learner | `adam` |
| learning_rate | `0.001` |
| epochs | `300` |
| stopping_step | `10` |
| valid_metric | `NDCG@10` |
| metrics | `NDCG`, `MRR`, `Hit` |

Явный split:

```yaml
eval_args:
  split: {'LS': 'valid_and_test'}
  order: TO
  group_by: user
  mode: full
```

## Protocol B dataset

Используется уже подготовленный в проекте KuaiRand Protocol B:

- manifest: `outputs/data/protocol_b_manifest.json`;
- корень данных на кластере: `/home/daryumin/iberdov/diplom/data/processed/protocol_b`;
- RecBole file: `/home/daryumin/iberdov/diplom/data/processed/protocol_b/recbole/kuairand/kuairand.inter`;
- заголовок: `user_id:token item_id:token timestamp:float`;
- filtered fingerprint: 23 951 users, 7 111 items, 1 134 420 interactions;
- split: chronological leave-one-out;
- `train`: все interactions кроме двух последних;
- `validation`: предпоследняя interaction;
- `test`: последняя interaction;
- tie-break: `user_id`, `timestamp`, `source_row_id`.

После SSD4Rec sequential augmentation ожидается `1 110 469` examples:
`1 134 420 - 23 951`, потому что первый interaction каждого пользователя не
может стать target с непустой history.

## Среда и GPU

Официальный SSD4Rec environment требует:

- Python `3.10.15`;
- CUDA toolkit / nvcc `11.8`;
- PyTorch `2.1.1+cu118`;
- RecBole `1.2.0`;
- mamba-ssm `2.2.2`;
- causal-conv1d `1.4.0`;
- numpy `1.26.3`;
- pandas `2.2.3`.

На кластере SSD4Rec должен запускаться из отдельного persistent env:
`/home/daryumin/iberdov/diplom/envs/ssd4rec`.

Для smoke этот путь создан как отдельный venv с
`sys.base_prefix=/home/daryumin/iberdov/diplom/envs/tim4rec`. Это не изменяет
TiM4Rec conda env, но наследует его GPU stack: Python `3.10.14`,
PyTorch `2.3.0+cu118`, RecBole `1.2.0`, mamba-ssm `2.2.2`,
causal-conv1d `1.2.2.post1`. Exact upstream environment по
`upstream/environment.yaml` остается целевым вариантом для отдельной
environment-lock задачи.

## Known issues в upstream

- `custom_utils.py` вызывает `getLogger()` без явного import. Smoke wrapper
  добавляет runtime shim `custom_utils.getLogger = logging.getLogger`.
- `custom_utils.py` ссылается на `np.float`, удаленный в новых NumPy. Smoke wrapper
  добавляет runtime shim `np.float = float`.
- Backward path в `BiSSDLayer` применяет `Mamba2` к reversed sequence, но результат
  не разворачивается обратно перед суммированием с forward path. Это поведение
  зафиксировано как upstream semantics; локально оно не менялось.
- Full-sort eval в upstream trainer маскирует только item id `0`; seen history
  items не исключаются явно.

## Статус на момент подготовки

- main уже содержит завершенный TiM4Rec run.
- Создана ветка `exp/ssd4rec-baseline`.
- Upstream snapshot сохранен локально без датасетов.
- Подготовлены config, environment notes, Slurm script и smoke test.
- Полное обучение SSD4Rec не запускалось.

## Smoke result

Успешный короткий smoke:

- JSON: `experiments/ssd4rec_baseline/runs/smoke_20260819T110252Z.json`;
- notes: `experiments/ssd4rec_baseline/runs/smoke_20260819T110252Z.md`;
- Slurm job: `4264304`;
- partition: `test`;
- constraint: `type_e`;
- node: `cn-046`;
- GPU: `NVIDIA A100-SXM4-80GB`;
- status: `COMPLETED`, exit code `0:0`, elapsed `00:03:17`;
- MaxRSS batch step: `2753448K`.

Проверено:

- import `torch`, `recbole`, `mamba_ssm`;
- чтение `kuairand.inter` из Protocol B;
- совпадение fingerprint `23951 / 7111 / 1134420`;
- `SSD4RecDataset` и custom variable-length dataloaders;
- one train batch: `1024` targets, `45156` flat sequence tokens;
- segment tensors согласованы: `cum_item_length[-1]`, `item_idx` и `flip_index`
  имеют те же `45156` tokens;
- masking работает: `9165` zero tokens, ratio `0.202963`;
- one optimizer step: loss `8.905934`, grad norm `0.957911`, parameter update
  `0.001000`;
- one full-ranking validation batch: scores shape `[2048, 7112]`, finite positive
  scores `2048`.

Полное обучение SSD4Rec этим job не запускалось.
