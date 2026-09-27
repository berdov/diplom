# Аудит публикации SISO dual/triple

Публикация 27.09.2026 использует только существующие результаты. Новых jobs, model forward/backward, optimizer steps и TRAIN/VALID/TEST evaluation нет. Кластер оставлен на execution commit `995c5cde6449ea429c1d80ca6ab276b9791041c0`, без изменений файлов, Git или environment.

## Сохранение и происхождение

- [Source index](../../../experiments/mamba3_three_time/confirmation/resume_plan_003.json) задаёт ровно десять источников: 2 pilot, 1 parent dual2027, 7 continuation. Выбор через glob не применялся. Унаследованный dual2027 относится к job4355052 / `3d3b305`, не к последнему job4355314 / `995c5cd`.
- [Manifest копирования](../../../experiments/mamba3_three_time/confirmation/evidence/job4355314/preservation_manifest.json): 59 недостающих файлов сохранены в исходных repository-relative путях, ещё 2 raw pilot JSON уже совпадали. Всего 61 файл / 3255086 байт проверены remote/local SHA256. Manifest перечисляет источник, destination, размер, SHA и действие.
- Десять бинарных checkpoint существуют; их SHA256 проверены потоковым чтением на кластере и сверены с raw JSON и metadata. В Git сохранены только metadata, модели не копировались и не десериализовались. Проверка существования относится к моменту публикации, не гарантирует будущую доступность кластерного диска.
- Raw JSON/Markdown, logs, reservation, pipeline и locks не редактировались. У attempt003 нет raw SVG: `resume_report.py` создаёт JSON и Markdown. [График](paired_delta.svg) является отдельным производным файлом, не «восстановленным оригиналом».
- Сохранённые inherited gates проверены по прежним путям и SHA256. Они не запускались повторно. Старые FAIL/NOT_RUN и exact-zero failures остаются историей; policy v1 и допуски не менялись.

## Научные источники и документы

До публикационных правок проверены все 371 файла текущего execution manifest локально и на кластере. Исторические snapshots дополнительно проверены по Git-объектам: pilot `6b5618a` (277 файлов), parent `3d3b305` (313), continuation `995c5cd` (371). Механизм: SHA256 байтов `git show <execution_commit>:<path>`, без checkout исторических версий. [Helper](report.py) повторяет эту проверку с `--audit`.

После публикации **не заявляется совпадение всего текущего дерева со старым manifest**. В continuation manifest входят `experiments/results.csv` и `reports/RESULTS.md`, они закономерно обновлены. Их исторические байты проверяются по execution Git-объектам. Математические Python/YAML, model/config, numeric policy, frozen plans/README, tests/guards и старые raw evidence остались неизменны. Исходники запусков воспроизводятся с указанными execution commits; merge/publication commits не заменяют provenance. Между parent и continuation проверены ровно пять заранее описанных изменений и их before/after SHA из lineage, а также привязки reservation к плану, manifest и login verification.

## Независимая сверка

[Helper](report.py) использует только Python standard library. Он читает ровно десять разрешённых JSON и metadata: проверяет run/mode/seed/job/commit/source/core/pin, backend upstream SISO, policy v1, полный каталог, сохранённые данные и TRAIN reference, непрерывность histories, последнее равное лучшее значение, best metrics/diagnostics, все epoch metrics в stderr, checkpoint evidence и TEST=NOT_RUN/count=0.

Внутри пар сверены initial backbone, общие calibrator hashes, RNG перед fit и первый фактический batch. Для четырёх новых пар также сверены компоненты RNG, optimizer settings и precision; в пилотных JSON отдельных полей `rng_components`/`optimizer_settings`/`precision` нет. Config/effective config совпадают с соответствующим pilot за исключением seed и checkpoint directory. Никаких новых forward для подтверждения этих записей не выполнялось.

[Агрегация](summary.json) независимо пересчитана из histories и сверена с неизменённой исходной summary: четыре и пять пар отдельно, full horizon и первые 27 эпох, sample std с ddof=1, знаки и относительные приросты. Средние/std парных разностей не подменяются разностью model std. Диагностика относится к best epoch; доли возле bounds явно агрегированы как максимум по heads.

CSV: **68 + 10 = 78**. Header и все прежние строки сохранены побайтно; SHA256 старого префикса `f33df502ac3c5f68561322a08670dd09c65a62372f91aa676d626e4a442534d8`. Нет дублей dual2027, gates, FAIL/NOT_RUN и summary-строк. Все новые строки являются validation/completed, с фактическим commit/path и TEST count=0. TEST-таблицы и прежние разделы основного отчёта сохранены.

## Проверки и ограничения

Все 14 publication-only unit tests прошли: арифметика, ddof=1, отрицательные пары, first27, точный source index, отказ при неполных/дублированных источниках и FAIL/TEST, округление, SVG/TeX, реестр, прежние таблицы и относительные ссылки/anchor. Старые model/GPU tests не запускались. Проверка `git diff --check` относится к новым документам/helper; завершающие пробелы в неизменённых raw stderr сохранены намеренно для SHA256.

SVG просмотрен целиком в локальном Chrome headless, viewport 960×560: пять отдельных точек, явный ноль, отрицательная пара 2027 и отделённый pilot2026, без CI и соединения seeds; подписи не обрезаны. Quick Look thumbnail обрезал правую часть, поэтому он не использовался как финальная визуальная проверка. TeX проверен текстово и численно; локальный TeX-компилятор отсутствует, PDF preview не собран. Пакеты и шрифты не устанавливались.

Для Overleaf поместить [TeX-файл](siso_dual_triple_table.tex) в `tables/`, подключить `booktabs` в преамбуле и включить в раздел VALID-абляций:

```tex
\input{tables/siso_dual_triple_table}
```

Для узкой двухколоночной рукописи может потребоваться `table*` вместо `table`; это решается в актуальном Overleaf-проекте, который здесь не редактировался. Таблица не заменяет TEST benchmark и не содержит утверждений о значимости.

```bash
python3 -B reports/assets/three_time_confirmation/report.py --audit
python3 -B -m unittest discover -s reports/assets/three_time_confirmation -p 'test_*.py' -v
```
