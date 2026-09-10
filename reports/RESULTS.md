# Результаты проекта: краткая сводка и индекс

Главный canonical narrative: **[MTL_MOO_STUDY.md](MTL_MOO_STUDY.md)**. MTL/MOO study завершён как диагностическое исследование и не выбран основой proposed method. Следующий этап — design of the new end-to-end architecture/pipeline.

## Historical TEST: baseline и MTL

| Модель / run | Зафиксированный TEST NDCG@10 |
| --- | ---: |
| Random / random_002 | 0.0006 |
| MostPopular / mostpop_002 | 0.0167 |
| XGBoost / ltr_xgb_002 | 0.0150 |
| Tuned XGBoost / ltr_xgb_optuna_001 | 0.0177 |
| SSD4Rec / ssd4rec_001 | 0.0576 |
| TiM4Rec / tim4rec_001 | 0.0598 |
| Fixed-loss MTL / multitask_tim4rec_001 | 0.0581 |
| Tuned MTL / multitask_tim4rec_tuned_001 | 0.0598 |

Это существующие historical TEST-результаты; нового TEST не было. Полные метрики — в неизменённом [experiments/results.csv](../experiments/results.csv). Опубликованные внешние результаты — отдельно в [PAPER_RESULTS.md](PAPER_RESULTS.md).

## VALID: завершённый MTL/MOO study

EPO — лучший observed MOO representative: Stage 1 **0.0584**, Stage 2 **0.0588** VALID NDCG@10. Screening **16/16** auxiliary subsets дал максимум **0.0595** против собственного primary-only **0.0588**, то есть сохранённый прирост **+0.0007** на одном seed. Эти VALID-оценки не объединяются в рейтинг с historical TEST из таблицы выше.

<a id="stage1-convergence"></a>
<a id="stage2-tuned-moo"></a>
<a id="challenger-convergence"></a>
<a id="target-combination-screening"></a>

Screening остаётся one-seed экспериментом; подтверждённого multi-seed результата нет. Различия operating-point rules MosT/GradHV — historical limitation; дальнейшие эксперименты по линии не планируются. Эти ограничения не являются текущими TODO.

## Индекс appendix/evidence

| Материал | Содержание |
| --- | --- |
| [MTL_MOO_STUDY.md](MTL_MOO_STUDY.md) | Research question, этапы исследования, final decision, provenance |
| [MOO_FAMILIES.md](MOO_FAMILIES.md) | Восемь семейств и исходные представители |
| [MOO_EXPERIMENT_HISTORY.md](MOO_EXPERIMENT_HISTORY.md) | Подробная история Stage 1/2 и ограничения бюджетов |
| [MOO_REPRESENTATIVE_CHALLENGERS.md](MOO_REPRESENTATIVE_CHALLENGERS.md) | Завершённые FERERO, MosT, PHN-HVI и caveats |
| [STAGE3_AUXILIARY_ANALYSIS.md](STAGE3_AUXILIARY_ANALYSIS.md) | Вспомогательные задачи и градиентная диагностика |
| [TARGET_COMBINATION_ANALYSIS.md](TARGET_COMBINATION_ANALYSIS.md) | Все 16 subsets и существующие эффекты |
| [CANONICAL_RESULTS_AUDIT.md](CANONICAL_RESULTS_AUDIT.md) | Исторический аудит реестра и источников |
| [evidence/README.md](evidence/README.md) | Raw результаты, summary и контрольные суммы |

Scientific results, raw JSON/evidence, canonical CSV и PAPER_RESULTS сохранены. Эта редакция консолидирует документацию без новых экспериментов, анализа или расчётов.
