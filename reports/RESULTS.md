# Результаты проекта

## Mamba3: VALID

KuaiRand, хронологический leave-one-out, полный каталог. [Условия оценки и входы](EVALUATION_SETUP.md).

**Подтверждение shared/separate, seeds 2026–2030:** VALID NDCG@10 составляет **0.061700 ± 0.000875** и **0.062880 ± 0.000512** соответственно (mean ± sample std, ddof=1). Separate выше в 5/5 пар, прирост по средним **+1,91%**; на четырёх новых seeds **+1,25%**, 4/4 пары. При ограничении первыми 27 эпохами прирост **+1,74%**; constant-gap уступил real-gap на seed 2026. [Парные результаты, график и ограничения](MAMBA3_TIME_MECHANISMS_RESULTS.md#confirmation).

### Первоначальное сравнение, seed 2026

| Модель | VALID NDCG@10 | Источник |
|---|---:|---|
| Vanilla Mamba3 | 0.0584 | [JSON](../experiments/mamba3_baseline/runs/mamba3_validation_001.json) |
| Shared RT-Mamba3 | 0.0605 | [JSON](../experiments/mamba3_timeaware/runs/mamba3_timeaware_validation_001.json) |
| decay_only | 0.0612 | [JSON](../experiments/mamba3_time_mechanisms/runs/mamba3_decay_only_validation_001.json) |
| scan_only | 0.0611 | [JSON](../experiments/mamba3_time_mechanisms/runs/mamba3_scan_only_validation_001.json) |
| separate | 0.0633 | [JSON](../experiments/mamba3_time_mechanisms/runs/mamba3_separate_time_validation_001.json) |

В этой таблице отдельные исходные запуски, не средние по seeds. У separate лучшая эпоха 51 (с нуля) из 63; в первых 27 эпохах максимум 0.0614. Decay_only/scan_only пока не проверены на нескольких seeds. [Полные метрики и диагностика исходного сравнения](MAMBA3_TIME_MECHANISMS_RESULTS.md#initial-study).

## Mamba3: TEST

| Модель | HR=Recall@10 | HR=Recall@20 | HR=Recall@50 | NDCG@10 | NDCG@20 | NDCG@50 |
|---|---:|---:|---:|---:|---:|---:|
| [Vanilla Mamba3](../experiments/mamba3_baseline/runs/mamba3_final_test_001.json) | 0.1062 | 0.1708 | 0.3053 | 0.0590 | 0.0752 | 0.1017 |
| [Shared RT-Mamba3](../experiments/mamba3_timeaware/runs/mamba3_timeaware_final_test_001.json) | 0.1116 | 0.1764 | 0.3136 | 0.0613 | 0.0776 | 0.1046 |

Для каждого выбранного по VALID checkpoint выполнен один финальный TEST. У decay_only, scan_only и separate TEST нет. Сравнение с опубликованным TiM4Rec вынесено в [отдельную таблицу](PAPER_RESULTS.md); наша репродукция TiM4Rec ниже является другим источником.

## Базовые модели: TEST

| Модель / run | NDCG@10 |
|---|---:|
| Random / random_002 | 0.0006 |
| MostPopular / mostpop_002 | 0.0167 |
| XGBoost / ltr_xgb_002 | 0.0150 |
| Tuned XGBoost / ltr_xgb_optuna_001 | 0.0177 |
| SSD4Rec / ssd4rec_001 | 0.0576 |
| TiM4Rec / tim4rec_001 | 0.0598 |
| Fixed-loss MTL / multitask_tim4rec_001 | 0.0581 |
| Tuned MTL / multitask_tim4rec_tuned_001 | 0.0598 |

Полные метрики и ссылки на исходные JSON: [реестр](../experiments/results.csv).

## Завершённые исследования

**Proto-Mamba3:** VALID 0.0583 против vanilla 0.0584; улучшения нет, TEST не выполнялся. Наблюдались слабая специализация и сближение прототипов. KMeans строился на TRAIN-историях encoder выбранного по VALID vanilla checkpoint, тогда как backbone нового запуска обучался с нуля: пространства инициализации не гарантированно согласованы. [Результат и диагностика](../experiments/mamba3_prototypes/README.md). Random-init control остался в отдельной ветке и не является завершённым результатом main.

<a id="stage1-convergence"></a>
<a id="stage2-tuned-moo"></a>
<a id="challenger-convergence"></a>
<a id="target-combination-screening"></a>

**MTL/MOO:** EPO дал лучший наблюдаемый результат среди представителей; screening вспомогательных задач показал небольшой прирост на одном seed. Эта линия не выбрана основой модели; ограничения бюджетов и выбора рабочей точки остаются частью исторического результата, а не текущим планом доработок.

| Исторические материалы | Содержание |
|---|---|
| [MTL/MOO study](MTL_MOO_STUDY.md) | Итоги и решение по линии |
| [Восемь семейств](MOO_FAMILIES.md), [история](MOO_EXPERIMENT_HISTORY.md), [challengers](MOO_REPRESENTATIVE_CHALLENGERS.md) | Представители, tuning и ограничения |
| [Auxiliary analysis](STAGE3_AUXILIARY_ANALYSIS.md), [target combinations](TARGET_COMBINATION_ANALYSIS.md) | Диагностика и все subsets |
| [Аудит реестра](CANONICAL_RESULTS_AUDIT.md), [evidence](evidence/README.md) | Источники и контрольные суммы |

Исторический [`slurm/epo_moe.sh`](../slurm/epo_moe.sh) сейчас неработоспособен: отсутствуют `experiments/epo_moe/{model.py,run.py,summarize.py,configs/epo_moe.yaml}`. Его удаление или архивирование требует отдельного решения.
