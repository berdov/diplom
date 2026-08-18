# TiM4Rec baseline для KuaiRand Protocol B

Цель ветки `exp/tim4rec-baseline` - подготовить воспроизводимый запуск TiM4Rec на уже зафиксированном KuaiRand Protocol B без полноценного обучения на этой стадии.

## Что зафиксировано

- Статья: https://arxiv.org/abs/2409.16182.
- Официальный репозиторий: https://github.com/AlwaysFHao/TiM4Rec.
- Снимок upstream: `8d4a6cea6a035c249a7a13999166ba41e8924abe`, дата коммита `2025-08-23T11:11:42+08:00`.
- Лицензия upstream: MIT, copyright 2024 AlwaysFHao.
- Локальная копия релевантного upstream-кода лежит в `experiments/tim4rec_baseline/upstream/`.

## Файлы

- `AUDIT.md` - аудит paper/repo/config и решение по `is_time`.
- `config_kuairand.yaml` - рабочий RecBole-конфиг для KuaiRand Protocol B.
- `environment.txt` - целевая среда и команды установки на cHARISMa.
- `smoke_test.py` - минимальная проверка imports, dataset/splits, model init и GPU forward.
- `train.py` - runner для будущего полного обучения, сейчас не запускался.
- `runs/` - только компактные smoke-результаты, без checkpoints, логов и файлов модели.

## Данные и split

Используется уже подготовленный Protocol B:

- корень данных на кластере: `/home/daryumin/iberdov/diplom/data/processed/protocol_b`;
- файл RecBole: `/home/daryumin/iberdov/diplom/data/processed/protocol_b/recbole/kuairand/kuairand.inter`;
- fingerprint: 23 951 users, 7 111 items, 1 134 420 interactions;
- split: chronological leave-one-out через RecBole sequential default `LS: valid_and_test`, `order: TO`, `mode: full`, `MAX_ITEM_LIST_LENGTH: 50`.

## Smoke test

По умолчанию Slurm script запускает только smoke test на GPU partition кластера E с `--constraint=type_e`:

```bash
sbatch slurm/tim4rec_baseline.sh
```

Полное обучение намеренно не запускалось. Для будущего запуска обучения:

```bash
sbatch --export=ALL,TIM4REC_STAGE=train slurm/tim4rec_baseline.sh
```

Перед публикацией полноценных метрик нужно явно указать, какая конфигурация считается основной: `is_time=True` как time-aware TiM4Rec из paper или официальный KuaiRand config с `is_time=False` как отдельный контрольный запуск.
