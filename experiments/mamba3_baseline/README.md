# Frozen vanilla Mamba3 baseline

`Mamba3Rec` — зафиксированный baseline для primary-only next-item prediction на KuaiRand Protocol B. Архитектура не time-aware: порядок событий хронологический, но timestamps, delta-time и time embeddings в модель не подаются. Здесь нет механизмов TiM4Rec, auxiliary tasks, MTL/MOO/EPO, MoE или flow matching.

## Модель и воспроизводимость

Item embeddings → два Mamba-3 SISO блока → последнее валидное состояние → tied item scoring. Hidden size 64, dropout 0.2, FFN 256, d_state 128, expand 2, headdim 64, chunk_size 64, bfloat16; 610440 параметров. Конфигурация обучения не изменена после просмотра TEST.

Исходный Mamba: `state-spaces/mamba`, commit `e9594ce1c732d97440f0332fdc43170a2294dbfa`. Стек и установка описаны в [ENVIRONMENT.md](ENVIRONMENT.md); параметры — в [config_kuairand.yaml](config_kuairand.yaml).

Protocol B: 23951 пользователей, 7111 items, 1134420 взаимодействий; train 1086518, VALID 23951, TEST 23951. Chronological leave-one-out, максимальная длина последовательности 50, full-ranking. SHA256 входного `.inter`: `e275ded0b330c2827b49ccf567d6784452d6dcbf8cd719dc3009d36eadc2e2cc`.

## Выбор и заморозка

Выбор модели выполнен только по VALID NDCG@10: **0.0584**. Лог обучения фиксирует лучшую epoch **15** (нумерация RecBole с нуля); обучение остановилось после epoch 26. [VALID JSON](runs/mamba3_validation_001.json) сохранён без редактирования. [Freeze summary](runs/mamba3_frozen_summary.json) связывает исходные артефакты и Slurm jobs; это сводка существующих свидетельств, составленная после запуска, а не новый selection-run.

Frozen checkpoint, не хранящийся в Git:

`/home/daryumin/iberdov/diplom_exp_mamba3_baseline/experiments/mamba3_baseline/checkpoints/Mamba3Rec-Sep-12-2026_17-55-38.pth`

SHA256: `d0bc3bb504daf5df068b6fd5bd635d6454aa223da01c006c63c78da193bb9dbe`.

## Единственный финальный TEST

После выбора checkpoint по VALID выполнена ровно одна успешная TEST evaluation: job **4324609**, COMPLETED, exit 0:0, `FullSortEvalDataLoader`, `test_evaluation_count=1`. Предыдущий job 4323168 завершился при загрузке checkpoint до обработки TEST-батчей и не вычислил TEST-метрики.

| TEST | @5 | @10 | @20 | @50 |
|---|---:|---:|---:|---:|
| HR | 0.0660 | 0.1062 | 0.1708 | 0.3053 |
| NDCG | 0.0461 | 0.0590 | 0.0752 | 0.1017 |
| Recall | 0.0660 | 0.1062 | 0.1708 | 0.3053 |

Primary metric: **TEST NDCG@10 = 0.0590**. [Финальный JSON](runs/mamba3_final_test_001.json) сохранён без редактирования. Это не утверждение о превосходстве над TiM4Rec.

Повторная TEST evaluation запрещена без отдельного научного решения. Нельзя менять параметры baseline или выбирать другой checkpoint по увиденному TEST. Следующая архитектурная модификация Mamba3 выполняется отдельной задачей и веткой; этот baseline остаётся frozen.

## Файлы и запуск

- [model.py](model.py) — неизменённая архитектура.
- [run.py](run.py) и [training launcher](../../slurm/mamba3_baseline.sh) — исторический smoke/VALID training; они не вычисляют TEST.
- [mamba3_final_test.py](mamba3_final_test.py) и [TEST launcher](../../slurm/mamba3_final_test.sh) — код выполненной финальной оценки, сохранён для provenance, не для повторного запуска. Для собственного доверенного checkpoint использован `weights_only=False`, без глобального изменения RecBole.
- Данные доступны через `/home/daryumin/iberdov/diplom/data`, окружение — через `/home/daryumin/iberdov/diplom/envs/mamba3`. Checkpoint остаётся в отдельном Mamba-каталоге.

Большие логи, checkpoint, кэши, datasets и environment binaries в Git не включаются.
