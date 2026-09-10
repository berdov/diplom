# Дипломный проект

Последовательные рекомендательные системы на KuaiRand: основная задача — предсказание следующего объекта (next-item recommendation), сильный baseline — TiM4Rec.

## Protocol B

KuaiRand-Pure / KuaiRand-27K после 5-core фильтрации: **23 951 пользователей, 7 111 объектов, 1 134 420 взаимодействий**. Хронологическое разбиение: TRAIN 1 086 518, VALID 23 951, TEST 23 951; максимальная длина истории — 50.

Последнее взаимодействие пользователя относится к TEST, предпоследнее — к VALID, предшествующая история — к TRAIN. Целевой объект исключён из контекста. Основные метрики получены ранжированием по полному каталогу 7 111 объектов; sampled-кандидаты не смешиваются с full-ranking оценкой.

[Manifest протокола](outputs/data/protocol_b_manifest.json) · [Отчёт по данным](reports/kuairand_protocol_b_data_report.md).

## Основные baseline

| Run | Модель | Historical TEST NDCG@10 |
| --- | --- | ---: |
| random_002 | Random | 0.0006 |
| mostpop_002 | MostPopular | 0.0167 |
| ltr_xgb_002 | XGBoost LambdaMART | 0.0150 |
| ltr_xgb_optuna_001 | Tuned XGBoost LambdaMART | 0.0177 |
| ssd4rec_001 | SSD4Rec | 0.0576 |
| tim4rec_001 | TiM4Rec | 0.0598 |

Это зафиксированные исторические результаты, а не новые оценки TEST. Полные метрики — в [experiments/results.csv](experiments/results.csv); опубликованные внешние результаты сохранены отдельно в [PAPER_RESULTS.md](reports/PAPER_RESULTS.md).

## Completed MTL/MOO study

Исследование вспомогательных поведенческих задач и восьми MOO-семейств завершено как диагностический этап и **не выбрано основой proposed method**. EPO был лучшим observed MOO representative. Screening 16 auxiliary subsets дал максимум **+0.0007 VALID NDCG@10 на одном seed**; tuned MTL и TiM4Rec имеют одинаковый зафиксированный **historical TEST NDCG@10 = 0.0598**. VALID EPO и TEST TiM4Rec не сравниваются как одна метрика.

В зафиксированном завершённом study multi-seed confirmation не проводился, поскольку MTL/MOO линия не была выбрана для proposed method. Разные operating-point rules MosT/GradHV сохранены как historical limitation; дальнейшие эксперименты по этой линии не планируются. Это ограничения завершённого исследования, а не текущие TODO.

Главный отчёт — **[reports/MTL_MOO_STUDY.md](reports/MTL_MOO_STUDY.md)**. [RESULTS.md](reports/RESULTS.md) содержит компактный индекс; подробные MOO/MTL appendix reports и raw evidence сохранены.

## Current stage

**Design of the new end-to-end architecture/pipeline.** Следующий этап — проектирование новой сквозной архитектуры и пайплайна для primary next-item recommendation. Новая proposed architecture ещё не зафиксирована.
