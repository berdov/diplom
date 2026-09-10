# MTL/MOO study: завершённое диагностическое исследование

Canonical narrative завершённого MTL/MOO этапа. Документ механически объединяет уже зафиксированные результаты; новые эксперименты, расчёты, tuning и оценки TEST при его подготовке не проводились. Подробные отчёты остаются appendix/evidence.

## Research question

Проверялось, улучшают ли вспомогательные поведенческие задачи и методы многокритериальной оптимизации primary next-item recommendation сильной последовательной модели TiM4Rec. Качество вспомогательных предсказаний само по себе не заменяет улучшение основной задачи.

Постановка: KuaiRand-Pure / Protocol B, 23 951 пользователей, 7 111 объектов, 1 134 420 взаимодействий. Хронологическое разбиение: TRAIN 1 086 518, VALID 23 951, TEST 23 951; максимальная длина истории 50. Оценки VALID и historical TEST далее явно разделены.

## TiM4Rec baseline

Сильный baseline — воспроизведение `tim4rec_001`. Зафиксированные historical TEST-метрики:

| Run | Split | HR@10 | HR@20 | HR@50 | NDCG@10 | NDCG@20 | NDCG@50 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tim4rec_001 | historical TEST | 0.1053 | 0.1696 | 0.3031 | 0.0598 | 0.0759 | 0.1022 |

Это наше воспроизведение, а не результат исходной статьи. Опубликованные внешние значения сохранены отдельно в [PAPER_RESULTS.md](PAPER_RESULTS.md). Новая оценка TEST не проводилась.

## MTL setup

Общее представление TiM4Rec используется основной next-item задачей и четырьмя auxiliary heads: `is_click`, `long_view`, `is_like`, `is_profile_enter`. Исследованы fixed-loss MTL, настроенные фиксированные веса и MOO-механизмы согласования задач.

| Run | Split | NDCG@10 | Роль |
| --- | --- | ---: | --- |
| multitask_tim4rec_001 | historical TEST | 0.0581 | Fixed-loss MTL |
| multitask_tim4rec_tuned_001 | historical TEST | 0.0598 | Tuned fixed-weight MTL |
| multitask_optuna_search_001 | VALID | 0.0599 | Исторический поиск конфигурации |

TiM4Rec и tuned MTL имеют одинаковый зафиксированный **historical TEST NDCG@10 = 0.0598**. Это сравнение одного split; VALID-результаты MOO не подставляются в эту таблицу как TEST-оценки.

Исторический Stage 3 исследовал отдельные auxiliary targets и градиентные взаимодействия. Его primary-only (VALID 0.0586) и all-four diagnostic с `tuned_task_weights` (VALID 0.0597) сохранены отдельно: они не заменяют контроль и all-four нового screening с `uniform_normalized_aux`. Полная диагностика — [appendix Stage 3](STAGE3_AUXILIARY_ANALYSIS.md).

## 8 MOO families

Исходный охват восьми семейств и ограничения адаптаций сохранены в [MOO_FAMILIES.md](MOO_FAMILIES.md).

| Семейство | Исходный представитель Stage 1 |
| --- | --- |
| Балансировка потерь | STCH |
| Взвешивание градиентов | FAMO |
| Коррекция конфликтующих градиентов | PCGrad |
| Конечный набор с preferences | EPO |
| Конечный набор без preferences в оптимизаторе | GradHV-style |
| Гиперсетевое представление | PHN-adapter |
| Сеть с preference conditioning | COSMOS-style |
| Комбинация моделей | PaLoRA |

Сравнение характеризует конкретные реализации и адаптации в данной постановке, а не предел качества каждого семейства.

## Stage 1 convergence

Восемь завершённых convergence-запусков, только VALID, seed 2026. Smoke/sanity не являются строками научного сравнения. Максимум 100 эпох, validation каждые 5 эпох, minimum training epochs 20, patience 3 validation checks.

| Representative | Run | VALID NDCG@10 |
| --- | --- | ---: |
| EPO | epo_convergence_001 | 0.0584 |
| GradHV-style | gradhv_convergence_001 | 0.0486 |
| COSMOS-style | cosmos_convergence_001 | 0.0453 |
| PCGrad | pcgrad_convergence_001 | 0.0444 |
| STCH | stch_convergence_001 | 0.0424 |
| PHN-adapter | phn_convergence_001 | 0.0423 |
| PaLoRA | palora_convergence_001 | 0.0422 |
| FAMO | famo_convergence_001 | 0.0412 |

EPO — лучший observed MOO representative Stage 1. Полные метрики и история — [MOO_EXPERIMENT_HISTORY.md](MOO_EXPERIMENT_HISTORY.md).

## Stage 2 top-4 tuning

Историческая настройка четырёх лучших представителей Stage 1; только VALID. Фактические бюджеты различались.

| Метод | Stage 1 VALID NDCG@10 | Stage 2 VALID NDCG@10 | Завершённые / запланированные trials | Ограничение |
| --- | ---: | ---: | ---: | --- |
| EPO | 0.0584 | 0.0588 | 5/10 | Исчерпан 36-часовой лимит |
| GradHV-style | 0.0486 | 0.0488 | 12/12 | Бюджет завершён |
| COSMOS-style | 0.0453 | 0.0455 | 9/12 | Остановка по preference_sensitivity |
| PCGrad | 0.0444 | 0.0464 | 12/12 | Бюджет завершён |

EPO сохранил лучший observed VALID-результат этого среза. Неодинаковый бюджет не позволяет трактовать таблицу как равнобюджетное доказательство превосходства. Устаревшие и неуспешные trials не используются как финальные результаты. Stage 2 не подменяет Stage 1 в challenger comparison; [история и provenance](MOO_EXPERIMENT_HISTORY.md).

## Challenger convergence

Три завершённых запуска после smoke/sanity; VALID, seed 2026, без нового tuning и TEST. Они сопоставляются с исходным Stage 1.

| Historical representative | Stage 1 VALID NDCG@10 | Challenger | Challenger VALID NDCG@10 |
| --- | ---: | --- | ---: |
| EPO | 0.0584 | FERERO-adapter | 0.0579 |
| GradHV-style | 0.0486 | MosT-style | 0.0522 |
| PHN-adapter | 0.0423 | PHN-HVI-adapter | 0.0443 |

EPO сохранён; PHN-HVI-adapter улучшил наблюдаемый результат PHN-adapter. Исторические правила выбора operating point GradHV и MosT различались; соответствующая оговорка сохранена в [appendix challengers](MOO_REPRESENTATIVE_CHALLENGERS.md). Здесь новый пересчёт и переопределение исторических результатов не выполнялись.

Auxiliary BCE FERERO **2.75–3.95** — отдельное ограничение вспомогательных предсказаний выбранной точки; primary VALID NDCG@10 **0.0579** не переинтерпретируется. Эти результаты не доказывают превосходство семейства.

## 16 target combinations

Полный screening завершён: **16/16** subsets четырёх auxiliary, primary next-item присутствует всегда. Только VALID, один seed 2026; fixed hyperparameters, максимум 80 эпох, validation каждую эпоху, patience 5. Loss: `L_rank + 0.13182740780834337 * mean(active auxiliary BCE)`; для пустого subset — `L_rank`.

| Вариант | Auxiliary subset | VALID NDCG@10 | Δ к primary-only этого screening |
| --- | --- | ---: | ---: |
| Primary-only | — | 0.0588 | 0.0000 |
| Best single | click | 0.0592 | +0.0004 |
| Best pair | like + profile_enter | 0.0595 | +0.0007 |
| Best triple | click + like + profile_enter | 0.0595 | +0.0007 |
| All-four | click + long_view + like + profile_enter | 0.0589 | +0.0001 |

Пара и тройка делят максимум при сохранённой точности метрик. **Максимальный прирост +0.0007 относится к VALID и одному seed**, не является multi-seed или статистическим подтверждением. Контроль — собственный primary-only 0.0588, а не исторический Stage 3 или EPO. Полная таблица 16 строк и уже рассчитанные эффекты сохранены в [TARGET_COMBINATION_ANALYSIS.md](TARGET_COMBINATION_ANALYSIS.md).

## Final decision

В проведённой постановке MTL с auxiliary behavioral targets и исследованные MOO-подходы не продемонстрировали убедительного улучшения primary next-item recommendation относительно сильного TiM4Rec baseline. EPO был лучшим observed MOO representative. Полный screening 16 auxiliary subsets дал максимум +0.0007 VALID NDCG@10 на одном seed; tuned MTL и TiM4Rec имеют одинаковый зафиксированный TEST NDCG@10=0.0598. Поэтому MTL/MOO сохраняется как завершённое диагностическое исследование, но не выбирается основой proposed method.

Этот вывод не получен прямым сравнением VALID EPO и TEST TiM4Rec как одной метрики: MOO/screening выводы относятся к VALID, равенство TiM4Rec и tuned MTL — к уже зафиксированному historical TEST. Завершение диагностического исследования не означает доказательства бесполезности всех MTL/MOO-методов.

EPO+MoE M0/M2/M4/M8 — **historical technical experiment; jobs failed; no scientific result; abandoned as current direction**. Отсутствующие научные результаты не заменяются нулями и не трактуются как отрицательное качество архитектуры. [Историческая постановка](EPO_MOE_BENCHMARK.md) сохранена.

Текущий следующий этап — **design of the new end-to-end architecture/pipeline**. Новая proposed architecture этим документом не реализуется и не оценивается.

## Evidence/provenance

- [experiments/results.csv](../experiments/results.csv) — неизменённый canonical реестр с разделением этапов и split.
- [CANONICAL_RESULTS_AUDIT.md](CANONICAL_RESULTS_AUDIT.md) — существующий аудит переноса результатов.
- [MOO_FAMILIES.md](MOO_FAMILIES.md), [MOO_EXPERIMENT_HISTORY.md](MOO_EXPERIMENT_HISTORY.md) — классификация, Stage 1/2 и история.
- [MOO_REPRESENTATIVE_CHALLENGERS.md](MOO_REPRESENTATIVE_CHALLENGERS.md) — три convergence run IDs, fidelity и caveats; code SHA `1a98ef966a2923a8f234b71602291b1c71ab82c3`.
- [STAGE3_AUXILIARY_ANALYSIS.md](STAGE3_AUXILIARY_ANALYSIS.md), [TARGET_COMBINATION_ANALYSIS.md](TARGET_COMBINATION_ANALYSIS.md) — отдельные диагностические постановки; screening code SHA `599bcdb6e50dceea73233169b8834eaceef083a8`.
- [evidence/README.md](evidence/README.md), [screening summary](evidence/target_combinations/summary.json), [контрольные суммы](evidence/completed_run_hashes.json) — сохранённые raw результаты и происхождение.
- [PAPER_RESULTS.md](PAPER_RESULTS.md) — неизменённые опубликованные внешние результаты.

Консолидация выполнена в `docs/canonical-moo-mtl-results` из существующего snapshot `93053d86eb3b7516ecf748fe060baa179d4e678b`. Raw JSON/evidence, canonical CSV и PAPER_RESULTS не переписывались и не удалялись; новые научные результаты не добавлялись.
