# Experimental setup

**KuaiRand — хронологический leave-one-out, оценка по полному каталогу.**
Это уточнение публичного названия, а не изменение данных или оценки.

## Данные и разбиение

- Источник: KuaiRand-Pure, `log_standard_4_08_to_4_21_pure.csv`,
  `is_rand=0`. Использован standard log, не randomized log.
- Итеративная фильтрация users/items с минимумом 5 взаимодействий:
  23 951 пользователей, 7 111 items, 1 134 420 взаимодействий.
- Дубликаты не удалены: 16 508 лишних exact-duplicate строк после фильтрации.
  Сортировка: `user_id, timestamp, source_row_id`; последний ключ означает
  исходную нулевую позицию строки CSV и разрешает равные timestamps.
- Для каждого пользователя последнее событие — TEST, предпоследнее — VALID,
  предыдущие — TRAIN. Это user-wise chronological split, не global-time holdout.
  TRAIN/VALID/TEST: 1 086 518 / 23 951 / 23 951 событий.
- Последовательный TRAIN содержит 1 062 567 примеров: первое событие каждого
  пользователя не имеет предшествующего контекста. Максимальная история — 50.
  У каждой VALID/TEST query ровно один relevant target. Повторный target допустим.

## Оценка Mamba3

Full-catalog scores включают все 7 111 реальных items. Padding ID 0 получает
`-inf`. В установленном RecBole 1.2.0 `FullSortEvalDataLoader.collate_fn` для
sequential model возвращает `history_index=None`: **seen items не исключаются**.
`Trainer._full_sort_batch_eval` маскирует history только при ненулевом индексе.
Это проверено чтением установленного кода; новой оценки для проверки не было.
HR и Recall совпадают здесь из-за одного relevant target, но во внешней таблице
сохранено обозначение Recall из источника. Sampled результаты с ними не смешиваются.

Истории right-padded. Целевые item/timestamp исключены из входного контекста;
временные модели используют только adjacent gaps внутри наблюдаемой истории,
float64 timestamps. Первое событие и padding нейтральны. Настоящий нулевой gap
между двумя valid событиями остаётся активным наблюдением.
TRAIN-only reference — 838 393 ms; bounds scale — [0.5, 2].

Checkpoint выбирается по VALID NDCG@10. Adam 0.001, batch 2048, eval batch 4096,
CE, максимум 300 эпох, `stopping_step=10`, VALID каждую эпоху.
Наблюдаемые ранние результаты имеют один seed 2026, метрики округлены RecBole
до четырёх знаков. Это не multi-seed mean и не оценка статистической значимости.

## Воспроизводимость

- [Неизменённый manifest](../outputs/data/protocol_b_manifest.json), SHA256
  `9f39aa12ec16f697a8bedb91eeb21c46dce514e6f414b7e47fc4064c1fcffd2c`.
- `.inter` SHA256:
  `e275ded0b330c2827b49ccf567d6784452d6dcbf8cd719dc3009d36eadc2e2cc`.
- [Frozen TRAIN statistics](../experiments/mamba3_timeaware/runs/train_time_stats_001.json),
  SHA256 `fa5df0e5ec97d84e5dffd157373318ebfa2cb94fe2853c2aec4898ecd5f89943`.
- [Подготовка данных](../src/prepare_kuairand_protocol_b.py) и
  [точные исторические timestamps](../experiments/mamba3_timeaware/dataset.py).

Старые `B` / `protocol_b` сохранены только как legacy identifiers для
воспроизводимости: в путях, immutable JSON/CSV, frozen code/config и архивных
отчётах. Они не обозначают отдельный публичный метод. Данные, маски, SHA,
исторические JSON и строки CSV не переписаны. [Инвентарь](evidence/legacy_identifiers.json).

Внешняя сопоставимость с TiM4Rec не подтверждена по всем этим деталям:
см. [PAPER_RESULTS.md](PAPER_RESULTS.md). Совпадение counts недостаточно.
