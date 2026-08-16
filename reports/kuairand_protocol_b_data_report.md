# Отчёт по данным KuaiRand Protocol B

Отчёт построен по фактическому полному препроцессингу на cHARISMa. Большие обработанные датасеты сохранены во внешнем каталоге, в Git попадают только код, конфигурация, manifest, checksums и компактные stats.

## 1. Протокол из источников

- Статья SSD4Rec: https://arxiv.org/html/2409.01192v1.
- Статья TiM4Rec: https://arxiv.org/html/2409.16182v1.
- Официальный репозиторий TiM4Rec: https://github.com/AlwaysFHao/TiM4Rec.
- SSD4Rec: бенчмарк KuaiRand имеет контрольный fingerprint `23,951 users / 7,111 items / 1,134,420 interactions`, использует leave-one-out разбиение по SASRec и `MAX_ITEM_LIST_LENGTH=50` для KuaiRand.
- TiM4Rec: сортировка по timestamp, минимум 5 interactions для users/items, тот же fingerprint `23,951 / 7,111 / 1,134,420`.
- Официальный config TiM4Rec `config/config4kuai_64d.yaml` задаёт `MAX_ITEM_LIST_LENGTH=50`, `load_col=[user_id,item_id,timestamp]`, `user_inter_num_interval=[5,inf)`, `item_inter_num_interval=[5,inf)`, `train_neg_sample_args=~`, но не задаёт явный `eval_args`.
- Поэтому для TiM4Rec применяется стандартная sequential-настройка RecBole 1.2.0: `{'split': {'LS': 'valid_and_test'}, 'order': 'TO', 'group_by': 'user', 'mode': {'valid': 'full', 'test': 'full'}}`.
- Канонический вариант B в этом репозитории: совместимый с SSD4Rec/SASRec/TiM4Rec хронологический leave-one-out по раннему standard log из KuaiRand-Pure.

## 2. Исходные данные

- Исходный лог: `/home/daryumin/iberdov/Corpora/KuaiRand-Pure/KuaiRand-Pure/data/log_standard_4_08_to_4_21_pure.csv`.
- Прочитано строк: 1,141,112.
- Исходные users/items: 26,210 / 7,538.
- Диапазон дат: `20220409`-`20220421`.
- Значения `is_rand`: `['0']`. Random logs в Protocol B не используются.

## 3. Фильтрация

- Правило фильтрации: итеративный RecBole-style 5-core по users и items.
- Дубликаты interactions не удаляются, что соответствует default `rm_dup_inter: ~` в RecBole.
- Итераций k-core: 2.
- Итоговые users/items/interactions: 23,951 / 7,111 / 1,134,420.
- Минимум interactions на user/item после фильтрации: 5 / 5.
- Совпадение с ожидаемым fingerprint: `True`.

## 4. Разбиение

- Правило разрешения равных timestamp: `сортировка по user_id, timestamp, source_row_id; source_row_id - нулевая позиция строки данных в исходном CSV`.
- Interactions в train: 1,086,518.
- Interactions в validation: 23,951.
- Interactions в test: 23,951.
- Users в train/validation/test: 23,951 / 23,951 / 23,951.
- Длина последовательности median/p95/max: 34.0 / 131.0 / 806.

## 5. Валидация

- Лишние строки exact duplicate после фильтрации: 16,508; протокол их не удаляет.
- Лишние строки user+timestamp duplicate после фильтрации: 87,198.
- Нарушения временного порядка: 0.
- Validation users без истории в train: 0.
- Test users без истории в train: 0.
- Split полностью покрывает filtered rows: `True`.

## 6. Воспроизводимость

- Commit кода препроцессинга: `9da072b3124b267d4a06bb3e23cd3d7d129153c6`.
- Скрипт: `src/prepare_kuairand_protocol_b.py`.
- Slurm-скрипт: `slurm/prepare_protocol_b.sh`.
- Путь хранения на кластере: `/home/daryumin/iberdov/diplom/data/processed/protocol_b`.
- Путь manifest: `/home/daryumin/iberdov/diplom/outputs/data/protocol_b_manifest.json`.

Команда sanity-прогона:

```bash
python src/prepare_kuairand_protocol_b.py --sanity-limit 10000 --output-dir data/processed/protocol_b_sanity --repo-output-dir outputs/data_sanity --report-path reports/kuairand_protocol_b_data_report_sanity.md
```

Команда полного Slurm-прогона:

```bash
sbatch slurm/prepare_protocol_b.sh
```

## 7. Файлы

| relative_path | размер | sha256 |
| --- | --- | --- |
| `full_filtered.parquet` | 7.8 MiB | `3c0abfbcca8810a57980a03ad28a12ac7a26f10629161fbb73ace4273468837b` |
| `item_id_mapping.parquet` | 38.4 KiB | `d8010cd0539c3d403d85832918f5214f9b26fc919cb8b32d2544b8f8d3d09e04` |
| `recbole/kuairand/kuairand.inter` | 26.5 MiB | `e275ded0b330c2827b49ccf567d6784452d6dcbf8cd719dc3009d36eadc2e2cc` |
| `recbole/kuairand_protocol_b.yaml` | 520 B | `63bc56ce2cd442fb5db3e7dbf99c559e71bf4151d447a2b5e1186fd2001e1df9` |
| `sequences.parquet` | 12.0 MiB | `e175095e9d5c379d529daa32c41285af7b6971e40719211402b716e1fe33f857` |
| `test.parquet` | 335.7 KiB | `b25e7032fd071cba235db04bcd8623df95f8eacde9c4545ce763e53dc05b3ab2` |
| `train.parquet` | 7.5 MiB | `ab619ee06f5cc2f033c43cfd0b8a0e2489cd4e8267e4cf74139b3deb8d0f67d8` |
| `user_id_mapping.parquet` | 134.9 KiB | `ad1e9cae32b6e815b5db74a3a8867216d830f821ac308f57b9a487f449918134` |
| `validation.parquet` | 337.7 KiB | `92733da83c5b3b75dc451b11aeb2d167e7cadfb77461b070f47d0dc6bfebc40e` |
