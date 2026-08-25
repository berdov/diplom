# Аудит результатов

Сгенерировано: `2026-08-25T14:29:14.108592+00:00`.

## Счётчики

| record_type | rows |
| --- | --- |
| experiment | 11 |
| experiment_validation_only | 1 |
| search | 2 |
| sanity | 12 |
| paper_reference | 2 |

## Замечания

- multitask_tim4rec_tuned_001 открыт на test после diagnostic tolerance: NDCG@10 diff=0.0010000000000000009, HR@10 diff=0.0023999999999999994; checkpoint не переобучался после первичного validation gate.
- Исторические `random_001`, `mostpop_001`, `ltr_xgb_001` сохранены как `sampled_100` и исключены из основной full-ranking таблицы.
- SSD4Rec paper v2 не сообщает @50 metrics; эти поля оставлены пустыми в `paper_reference`.
- Validation/search rows и locked test rows представлены отдельными записями.

## Проблемы валидации

- Проблем валидации нет.
