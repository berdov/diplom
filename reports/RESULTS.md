# Результаты проекта

## Mamba3: VALID

**Явная память внутри окна 50 событий, pilot seed2026:** no_memory / index_memory / time_memory — **0.0633 / 0.0626 / 0.0634** VALID NDCG@10. Primary time−index **+0.0008 (+1.278%)**, time−no_memory только **+0.0001**; first27 ниже исходной модели, эпох 76 против 39. Проверен ограниченный readout причинных представлений; устойчивый эффект не установлен, рабочая основа сохраняется. На VALID 89.18% временных anchor choices вне диапазона оставшихся кандидатов. 3/3 fits, TEST=0. [Результат и ограничения](MAMBA3_TIME_MECHANISMS_RESULTS.md#time-addressed-memory-pilot).

**Временные функции по слоям, paired pilot seed2026:** shared **0.0633**, layer-specific **0.0625** VALID NDCG@10; Δ **−0.0008 (−1.264%)**, first27 −0.0001. Пункт 4 проверен в pilot scope; сохраняются общие функции, confirmation не рекомендована. TEST=0. [Результат и диагностика](MAMBA3_TIME_MECHANISMS_RESULTS.md#layer-temporal-functions-pilot).

**Centered Gap-Trap, confirmation 2027–2030:** fixed **0.062925 ± 0.000793**, centered **0.062900 ± 0.000787**; paired Δ **−0.000025 ± 0.001124**, 2 выигрыша и 2 проигрыша (mean ± sample std, ddof=1). Устойчивого преимущества нет; пункт 3 завершён, рабочая основа — MIMO dual fixed-reference. TEST=0. [Таблицы, график и ограничения](MAMBA3_TIME_MECHANISMS_RESULTS.md#gap-trap-centered-confirmation).

**Centered Gap-Trap, seed2026:** VALID NDCG@10 0.0635 против 0.0633 fixed (+0.316%); first27 0.0615 против 0.0620, всего 60/39 эпох. Небольшой положительный пилот, устойчивость не установлена. Пункт 3 закрыт на текущем exploratory этапе, TEST=0. [Результат](MAMBA3_TIME_MECHANISMS_RESULTS.md#gap-trap-centered-pilot).

**One-sided Gap-Trap, seed2026:** VALID NDCG@10 0.0626 против 0.0633 fixed replay (−1.11%); first27 также ниже. Эта параметризация не улучшила пилот. TEST=0, multi-seed confirmation не запускалась. [Результат](MAMBA3_TIME_MECHANISMS_RESULTS.md#gap-trap-one-sided-pilot).

**Обучаемые масштабы MIMO dual, подтверждение2027–2030:** fixed/shared_tau/head_tau — VALID NDCG@10 **0.062925 / 0.062750 / 0.062950**. Head−shared: **+0.000200 (+0.319%)**, пары +/−/0: 3/1/0. Head−fixed: +0.040%, 2/4 wins; first27 head−shared: 0.000000. Пункт 2 завершён: head-specific reference scales не выбираются как обязательное усложнение, контроль остаётся fixed. Пять seeds с exploratory pilot показаны отдельно. TEST=0; эквивалентность и отсутствие эффекта вообще не установлены. [Полный результат, график и TeX](MAMBA3_TIME_MECHANISMS_RESULTS.md#head-timescales-confirmation).

**Обучаемые масштабы MIMO dual, пилот seed2026:** fixed/shared_tau/head_tau получили VALID NDCG@10 **0.0633 / 0.0627 / 0.0639**. Общая шкала не улучшила NDCG, индивидуальные дали небольшой выигрыш на одном seed. TEST не выполнялся; устойчивость не установлена. [Таблица, первые 27 эпох и выученные масштабы](MAMBA3_TIME_MECHANISMS_RESULTS.md#head-timescales-pilot).

KuaiRand, хронологический leave-one-out, полный каталог. [Условия оценки и входы](EVALUATION_SETUP.md).

**MIMO base/dual/triple:** на четырёх новых seeds 2027–2030 средний VALID NDCG@10 **0.059125 / 0.062925 / 0.063325**. Dual и triple выше base в 4/4 тройках (+6.427% / +7.104% по средним). Triple−dual: **+0.000400 (+0.636%)**, 3 улучшения и 1 снижение. С exploratory pilot 2026 средние dual/triple **0.063000 / 0.063020**, 3 положительные пары и 2 отрицательные. TEST для этой серии не выполнялся; значимость и эквивалентность не установлены. [Завершённое подтверждение, таблицы и график](MAMBA3_TIME_MECHANISMS_RESULTS.md#mimo-time-confirmation); [исторический пилот](MAMBA3_TIME_MECHANISMS_RESULTS.md#mimo-time-pilot).

**SISO dual/triple:** на четырёх подтверждающих seeds 2027–2030 средний VALID NDCG@10 **0.062775 / 0.063150**, парный прирост **+0.000375 (+0.597%)**, 3 улучшения и 1 снижение. С exploratory pilot 2026 отдельно: **0.062520 / 0.062980**, 4/5 положительных пар. Добавлено 66 параметров; TEST не выполнялся. [Таблицы, график, диагностика и ограничения](MAMBA3_TIME_MECHANISMS_RESULTS.md#siso-dual-triple-confirmation).

**Подтверждение shared/separate, seeds 2026–2030:** VALID NDCG@10 составляет **0.061700 ± 0.000875** и **0.062880 ± 0.000512** соответственно (mean ± sample std, ddof=1). Separate выше в 5/5 пар, прирост по средним **+1,91%**; на четырёх новых seeds **+1,25%**, 4/4 пары. При ограничении первыми 27 эпохами прирост **+1,74%**; constant-gap уступил real-gap на seed 2026. [Парные результаты, график и ограничения](MAMBA3_TIME_MECHANISMS_RESULTS.md#confirmation).

**Контекстная калибровка и временные эксперты, seed2026:** завершены пять TRAIN→VALID запусков. Routed/uniform получили NDCG@10 **0.0628**, separate replay **0.0633**, dense12 **0.0635**. Преимущество обучаемой маршрутизации не показано; separate остаётся рабочей основой, routed не выбран. Replay не добавляется в пятисидовую статистику. [Таблица, график и диагностика пилота](MAMBA3_TIME_MECHANISMS_RESULTS.md#context-time-pilot).

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
