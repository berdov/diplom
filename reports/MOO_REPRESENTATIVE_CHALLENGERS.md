# Завершённое сравнение представителей MOO

10 сентября 2026. Это отдельное challenger comparison после Stage 1; исторические строки Stage 1 сохранены без замены. Только VALID KuaiRand-Pure / Protocol B, seed 2026, full-sort 7111 items. TEST не загружался и не оценивался; нового tuning не было.

Три завершённых convergence-запуска внесены в [canonical CSV](../experiments/results.csv) с `record_type=challenger_convergence`; [сводная таблица](RESULTS.md#challenger-convergence). Числа smoke/sanity в canonical результаты не включены. [Обоснование восьми семейств и текущий выбор представителей](MOO_FAMILIES.md) · [Аудит переноса](CANONICAL_RESULTS_AUDIT.md).

| family | old representative | old NDCG@10 | challenger | challenger NDCG@10 | decision | caveat |
| --- | --- | ---: | --- | ---: | --- | --- |
| Конечный набор с preferences | EPO (Stage 1) | 0.0584 | FERERO-adapter | 0.0579 | Оставить EPO | Δ −0.0005; один seed, адаптация FERERO; отдельное ограничение auxiliary BCE 2.75–3.95 |
| Конечный набор без preferences в оптимизаторе | GradHV-style (Stage 1) | 0.0486 | MosT-style | 0.0522 | MosT — предварительный кандидат; окончательную замену GradHV отложить | Δ +0.0036; operating-point selection различается, fairness не закрыта |
| Гиперсетевое представление | PHN-adapter (Stage 1) | 0.0423 | PHN-HVI-adapter | 0.0443 | Выбрать PHN-HVI-adapter для следующего этапа | Δ +0.0020; один seed, адаптеры, отличия objective/sampling; не full-weight hypernetwork |

Выбор относится к конкретным реализациям в данном validation-only протоколе. Он не доказывает превосходство семейства или статистическую значимость различий.

## Завершённые запуски и freeze

| Challenger run | Job ID | Статус / gates | HR@10 | Лучшая эпоха | Остановка |
| --- | --- | --- | ---: | ---: | ---: |
| [ferero_convergence_001](../experiments/moo_representative_challengers/runs/ferero_convergence_001.json) | 4315313 | completed / passed | 0.1083 | 15 | 30 |
| [most_convergence_001](../experiments/moo_representative_challengers/runs/most_convergence_001.json) | 4315314 | completed / passed | 0.0947 | 10 | 25 |
| [phn_hvi_convergence_001](../experiments/moo_representative_challengers/runs/phn_hvi_convergence_001.json) | 4315315 | completed / passed | 0.0789 | 20 | 35 |

Exact code SHA всех трёх: `1a98ef966a2923a8f234b71602291b1c71ab82c3`, тот же, что у sanity. Максимум 100 эпох, validation каждые 5, minimum training epochs 20, patience 3 validation checks, min_delta 0. Все завершились по early stopping. Настройки методов, seed, data split и stopping после sanity не менялись; это проверено [при отправке](https://github.com/berdov/diplom/blob/47953484490964febf47c05733545bdc29308702/experiments/moo_representative_challengers/deployment/convergence_submission_snapshot.json). Результаты скопированы с кластера без изменений; [SHA-256](evidence/completed_run_hashes.json).

Сравнение использует старые [EPO](../experiments/moo_8families/runs/epo_convergence_001.json), [GradHV](../experiments/moo_8families/runs/gradhv_convergence_001.json), [PHN](../experiments/moo_8families/runs/phn_convergence_001.json) из Stage 1, а не tuned результаты Stage 2. Общие протокол, fixed parameters и расписание совпадают; вычислительная стоимость и механизмы оптимизации различаются. Fidelity и отличия: [DESIGN на зафиксированном code SHA](https://github.com/berdov/diplom/blob/1a98ef966a2923a8f234b71602291b1c71ab82c3/experiments/moo_representative_challengers/DESIGN.md).

## Fairness: MosT vs historical GradHV

Проблема **не закрыта**. Проверены код и сохранённые результаты:

- Historical GradHV: `validation.ranking_operating_point_selection = best_validation_NDCG@10_among_preference_free_finite_solutions`, `selection_is_validation_oracle = true`. Из трёх решений выбирается максимальный VALID NDCG@10; по этой величине выбирается epoch и работает early stopping. Реализация: [select_validation_points](../experiments/moo_8families/evaluation/pareto.py). Лучшая эпоха 50, остановка 65.
- MosT: из трёх решений выбирается `argmin_i Σ_j r_j z_ij`, где `r=(0.6,0.1,0.1,0.1,0.1)`, `z=(1−NDCG@10, click BCE, long_view BCE, like BCE, profile BCE)/(1,2,2,2,2)`; ties — по solution index. По NDCG@10 выбранного таким способом решения выбирается epoch и работает early stopping. Правило зафиксировано до sanity, `selection_is_validation_oracle = false`; [select_operating_point на code SHA запуска](https://github.com/berdov/diplom/blob/1a98ef966a2923a8f234b71602291b1c71ab82c3/experiments/moo_representative_challengers/common.py). Лучшая эпоха 10, остановка 25.

Поэтому 0.0522 > 0.0486 поддерживает предварительный выбор MosT для дальнейшей работы, но не закрывает сравнение алгоритмов при одинаковом правиле выбора. Historical GradHV использует более благоприятный для ranking oracle внутри набора; тем не менее одинаковыми процедуры отбора и остановки не становятся. Пересчёт только финальных трёх решений не выравнивает выбор epoch/early stopping. Нужен отдельно согласованный контроль с общим заранее заданным правилом; в этой задаче такой запуск не проводился, сохранённые результаты и selection не переписывались. До контроля GradHV остаётся historical reference, MosT — предварительным кандидатом на замену.

## Отдельное наблюдение FERERO auxiliary

У заранее выбранной точки `rank_heavy` BCE: click **2.7504**, long_view **3.1372**, like **3.9520**, profile_enter **3.3508**. Это существенное ограничение качества вспомогательных предсказаний выбранной модели. Оно не меняет primary ranking результат **NDCG@10 = 0.0579** и не служит основанием для переопределения operating point задним числом. Решение оставить EPO по primary ranking основано на 0.0584 против 0.0579.
