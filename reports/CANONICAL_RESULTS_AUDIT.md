# Аудит canonical результатов MOO/MTL

10 сентября 2026. Обновление подготовлено в отдельной ветке `docs/canonical-moo-mtl-results` от актуального после `git fetch` main: `16ba99b69929a711137f616df28e18e7f92b76c1`. Источник завершённых отчётов и evidence — экспериментальная ветка на commit `47953484490964febf47c05733545bdc29308702`. Merge в main не выполнялся. Scientific settings, код обучения, исходные результаты и кластерные checkouts не изменялись.

[Canonical summary](RESULTS.md) · [Canonical CSV](../experiments/results.csv) · [README](../README.md) · [Происхождение и SHA-256 evidence](evidence/README.md).

## Аудит запрошенных файлов

| Файл | Состояние до обновления | Изменение |
| --- | --- | --- |
| [RESULTS.md](RESULTS.md) | Stage 1/2 уже отражены; нет challengers и factorial screening; вопрос обоснования представителей помечен как открытый | Добавлены отдельные этапы и источники; найденное обоснование связано со сводкой; historical all-four diagnostic 0.0597 показан отдельно |
| [experiments/results.csv](../experiments/results.csv) | 28 строк: baselines/MTL, Stage 1, Stage 2, Stage 3; нет 19 новых результатов | Добавлены 3 challenger convergence и все 16 scientific combinations; прежние строки и заголовок сохранены байт-в-байт |
| [MOO_FAMILIES.md](MOO_FAMILIES.md) | В main есть литературное обоснование; в экспериментальной ветке оно дополнено завершёнными challengers | Сохранены критерии и исходная таблица Stage 1; добавлены завершённые сравнения и текущий статус, выбор GradHV/MosT явно открыт |
| [MOO_REPRESENTATIVE_CHALLENGERS.md](MOO_REPRESENTATIVE_CHALLENGERS.md) | Завершённый отчёт есть только в экспериментальной ветке | Перенесён и связан с canonical; ссылки на код/freeze закреплены на commit, локальные run JSON доступны |
| [TARGET_COMBINATION_ANALYSIS.md](TARGET_COMBINATION_ANALYSIS.md) | Завершённый отчёт 16/16 есть только в экспериментальной ветке | Перенесён и связан с canonical; one-seed descriptive, ничья pair/triple, отдельный baseline; пояснено отличие historical all-four |
| [PAPER_RESULTS.md](PAPER_RESULTS.md) | Фактический путь — `reports/PAPER_RESULTS.md`; содержит только опубликованные внешние результаты | Добавлена навигация к нашим этапам; опубликованные строки и числа не изменены |
| [README](../README.md) | Нет challengers/screening, EPO + MoE назван текущим этапом при отсутствии сохранённых метрик | Обновлены завершённые этапы, таблицы, ссылки и открытые вопросы; missing MoE не выдан за результат |

В [MOO_EXPERIMENT_HISTORY.md](MOO_EXPERIMENT_HISTORY.md) и [STAGE3_AUXILIARY_ANALYSIS.md](STAGE3_AUXILIARY_ANALYSIS.md) добавлены ссылки на актуальную сводку; исторические результаты и прежние выводы сохранены.

## Где находится обоснование восьми семейств

Основной исследовательский документ — [MOO_FAMILIES.md](MOO_FAMILIES.md): классификация из обзора, охват восьми конечных ветвей, литературные кандидаты, критерии включения, совместимость с TiM4Rec, выполнимость и fidelity адаптаций. Первоначальный выбор не обосновывается полученными NDCG. Эмпирические решения после завершённых сравнений вынесены отдельно.

История документа: commit [145cc812b1fa607a90e8617876b5f731c337a103](https://github.com/berdov/diplom/commit/145cc812b1fa607a90e8617876b5f731c337a103) — исходное обоснование; [b87cb06f56dd7b0d469aa2681a480044268d703b](https://github.com/berdov/diplom/commit/b87cb06f56dd7b0d469aa2681a480044268d703b) — академическая редакция; включение через [PR #3](https://github.com/berdov/diplom/pull/3), редакция main `16ba99b69929a711137f616df28e18e7f92b76c1`. Этот аудит фиксирует место исследования в репозитории; новых литературных или экспериментальных выводов не добавляет.

Рабочий состав: STCH, FAMO, PCGrad, EPO, **GradHV-style / MosT-style (выбор открыт)**, PHN-HVI-adapter, COSMOS-style, PaLoRA. FERERO не заменяет EPO. PHN-HVI выбран для следующего этапа с ограничениями адаптации и одного seed. MosT остаётся предварительным кандидатом до общего operating-point rule с учётом checkpoint и early stopping.

## Разделение записей и перенос метрик

Схема CSV не расширялась. Этап определяется через `record_type`; научный split указан в `split`, происхождение — в `source_json`, исходном `git_commit` и `notes_path`.

| `record_type` | Строк после обновления | Смысл |
| --- | ---: | --- |
| `experiment` | 8 | Historical TEST baselines/MTL; без изменений |
| `experiment_validation_only` | 1 | Historical PCGrad, не текущая реализация Stage 1 |
| `search` | 1 | Historical MTL Optuna search на VALID |
| `convergence_screening` | 8 | Только historical Stage 1 convergence |
| `tuning_budgeted_validation_only` | 4 | Итоги Stage 2, неодинаковый фактический бюджет сохранён |
| `stage3_auxiliary_analysis` | 6 | Historical диагностика: primary-only, 4 одиночные auxiliary, all-four diagnostic |
| `challenger_convergence` | 3 | Новые FERERO-adapter, MosT-style, PHN-HVI-adapter convergence |
| `target_combination_validation_screening` | 16 | Все subsets scientific screening попытки 002, без smoke |
| **Всего** | **47** | **28 прежних + 19 новых; ни одного удаления** |

У challenger источников собственное поле `stage=convergence_screening`; в общем реестре намеренно используется отдельный `record_type=challenger_convergence`, чтобы не смешивать их с исходным Stage 1. Метрики переносятся из сохранённого `validation.ranking_operating_point.metrics`, best epoch — из `training.best_epoch`, actual epochs — из `training.stop_epoch`. Правило отбора и результаты других точек не пересчитывались.

У target-combination метрики берутся из `ranking_metrics`, эпохи — из `best_epoch` и `actual_epochs`. Все 12 HR/Recall/NDCG полей K=5/10/20/50 сверены с JSON. Схема subset и ограничения зафиксированы в `notes_path` и исходном JSON; `parent_run` у новых строк пуст, чтобы не подразумевать наследование checkpoint. Имена target subsets однозначны в `run_id`/`model_variant`.

## Подтверждённые итоги и ограничения

- Challengers: FERERO **0.0579** (Stage 1 EPO **0.0584**), MosT **0.0522** (Stage 1 GradHV **0.0486**, fairness открыта), PHN-HVI **0.0443** (Stage 1 PHN-adapter **0.0423**). Это отдельный convergence-этап, без значений sanity и без подстановки tuned Stage 2.
- Target combinations **16/16**: primary-only **0.0588**, best single click **0.0592**, best pair like + profile_enter **0.0595**, best triple click + like + profile_enter **0.0595**, all-four **0.0589**. Best pair/triple делят первое место при четырёх знаках метрик. Выводы описательные по одному seed; multi-seed проверки ещё не выполнены.
- Historical Stage 3 primary-only **0.0586** и all-four diagnostic **0.0597** сохранены. Последний использует `tuned_task_weights`, новый all-four — `uniform_normalized_aux`; это не одна и та же конфигурация.
- Auxiliary BCE FERERO **2.75–3.95** остаётся отдельным наблюдением и не переопределяет его primary ranking.
- Сохранённый EPO + MoE summary не содержит подтверждённых метрик; текущие состояния jobs на кластере не проверялись. Новые архитектурные результаты не добавлены.

## Проверки

Выполнены без запуска обучения, tuning или оценки модели:

- SHA-256 всех **25** файлов из исходного `completed_run_hashes.json` совпадают; источник snapshot сохранён без изменения байтов.
- Все **19** новых запусков completed, gates passed, seed 2026, VALID, `test_evaluation_count=0`; записи доступа к данным содержат только train/valid loaders.
- Все **16** уникальных subsets сверены с индивидуальными JSON и summary; отдельно сохранённый smoke исключён. Проверены все HR/Recall/NDCG, best epoch и delta к текущему primary-only.
- **4** marginal effects (8 matched backgrounds на эффект) и **6** pairwise interactions (4 backgrounds на взаимодействие) независимо пересчитаны и совпали с summary.
- Исторические Stage 1, Stage 2 и Stage 3 сверены со своими сохранёнными источниками. CSV не содержит дублей `run_id`, все `source_json` доступны локально; 28 исходных строк и прежняя схема сохранены.
- Таблицы canonical summary, README и специализированных отчётов проверены на согласованность с источниками; локальные ссылки и формат diff проверены. Научный код и конфигурации не менялись, поэтому обучение и GPU-тесты для этого обновления не запускались.

Итог численных проверок: **388** полей CSV и **346** числовых ячеек Markdown; проверены **117** локальных ссылок и **4** ссылки на зафиксированные версии файлов в Git. Два исходных CSV в evidence имеют CRLF; этот формат сохранён ради исходных SHA-256. Поэтому проверка whitespace выполнена с `core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol` только для команды проверки, без изменения настроек репозитория.

Historical TEST сохранён как история. Он не использован для новых сравнений, выбора representatives или target subsets. Показанные значения VALID не объединяются с ним или с опубликованными paper-строками в единый рейтинг.
