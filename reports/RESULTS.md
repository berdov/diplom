# Результаты проекта: краткая сводка и индекс

Canonical narrative завершённого MTL/MOO study: **[MTL_MOO_STUDY.md](MTL_MOO_STUDY.md)**. Этот диагностический этап не выбран основой proposed method. Завершённый Mamba3 architecture stage приведён отдельно ниже.

## Зафиксированные TEST: baseline и MTL

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
| Frozen vanilla Mamba3Rec / mamba3_final_test_001 | 0.0590 |

Исторические TEST-результаты дополнены 13.09.2026 frozen vanilla [Mamba3Rec](../experiments/mamba3_baseline/README.md): primary-only, non-time-aware baseline на Protocol B, full-ranking. Checkpoint выбран по VALID; финальный TEST выполнен ровно один раз (`test_evaluation_count=1`). Полные метрики — в [experiments/results.csv](../experiments/results.csv). Опубликованные внешние результаты — отдельно в [PAPER_RESULTS.md](PAPER_RESULTS.md).

## Mamba3 architecture stage

KuaiRand Protocol B, full-ranking по 7111 items; VALID и TEST не смешиваются.

| Model | VALID NDCG@10 | TEST NDCG@10 | Status |
|---|---:|---:|---|
| Vanilla Mamba3 | 0.0584 | 0.0590 | frozen baseline |
| RT-Mamba3 | 0.0605 | 0.0613 | completed |
| Proto-Mamba3 KMeans | 0.0583 | — | VALID only |

Для контекста: TiM4Rec TEST NDCG@10 = **0.0598** при том же full-ranking Protocol B.

### RT-Mamba3

Реальные inter-event gaps модифицируют native Mamba3 internal DT.
[VALID](../experiments/mamba3_timeaware/runs/mamba3_timeaware_validation_001.json)
даёт **+3.60%** относительно vanilla; [final TEST](../experiments/mamba3_timeaware/runs/mamba3_timeaware_final_test_001.json)
даёт **+3.90%** к vanilla и **+2.51%** к TiM4Rec. Это один наблюдаемый run,
статистическая значимость не заявляется. Checkpoint выбран только по VALID;
TEST count=1, обучение и VALID reruns во время final TEST=0, повторного TEST
и post-test tuning не было. [Описание RT-Mamba3](../experiments/mamba3_timeaware/README.md).

### Proto-Mamba3

K=8 soft learnable prototypes: MiniBatchKMeans по TRAIN histories frozen vanilla
encoder, затем soft cosine assignment и gated residual, end-to-end обучение.
[VALID JSON](../experiments/mamba3_prototypes/runs/mamba3_prototypes_validation_001.json):
**0.0583** против vanilla **0.0584**, delta **-0.0001**; улучшения в этом run нет.
На best checkpoint mean off-diagonal cosine **0.996003**, assignments почти
равномерны (средние probabilities 0.124060–0.126336): prototype collapse / weak
specialization. Это ограничение наблюдаемой formulation, не доказательство
бесполезности прототипов вообще. **TEST NOT RUN**, count=0.
[Описание Proto-Mamba3](../experiments/mamba3_prototypes/README.md).

KMeans examples are TRAIN-only, but they are encoded using a validation-selected
frozen vanilla checkpoint, while the scientific Proto-Mamba3 backbone is trained
from scratch. Это coordinate-space caveat, не TEST leakage.

Random-init control подготовлен только в `exp/mamba3-prototypes-controls`:
не запускался, не является завершённым результатом и не включён в main.

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
