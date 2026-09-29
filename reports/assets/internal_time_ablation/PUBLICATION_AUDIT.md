# Проверки публикации SISO/MIMO VALID

Публикация 29 сентября 2026 года объединяет **25 уже полученных runs**:
10 SISO и 15 MIMO. Это индекс и пересчёт существующих результатов, не новый
эксперимент. Основные научные выводы находятся в
[едином отчёте](../../MAMBA3_TIME_MECHANISMS_RESULTS.md).

## Источники и сохранение

SISO выбран по `resume_plan_003.json` и существующему source index:

| Runs | n | Job | Execution commit |
|---|---:|---|---|
| SISO pilot2026 dual/triple | 2 | 4353980 | `6b5618a769d8e3424df0b7232605426fac8c573a` |
| SISO dual2027, унаследованный | 1 | 4355052 | `3d3b305c4c74e208cd226d3ed4ccec37b4fd8311` |
| SISO остальные confirmation | 7 | 4355314 | `995c5cde6449ea429c1d80ca6ab276b9791041c0` |
| MIMO pilot2026 base/dual/triple | 3 | 4358147 | `c1dd31eec7907c67348769b6a1811b89aa4011c0` |
| MIMO confirmation2027–2030 | 12 | 4358583 | `5670e898ed04924a929756e52f39d1d00eb79c5a` |

Новые 12 results выбраны строго по MIMO `study_plan.json` и путям из config.
Их evidence сохранён отдельным commit `7479abf` до изменений публикации.
[Manifest](../../../experiments/mamba3_mimo_time/confirmation/evidence/job4358583/preservation_manifest.json)
содержит **79 файлов / 3 518 393 байта**, source/destination/size/SHA256.
Все копии совпали побайтно; raw JSON и исходные summaries не форматировались.

SHA256 manifest сохранения:
`a9f6c4c7ba55d5e567b866bfeced3168121cdbe85aa62c9a91eb42879c8e401b`.
Source hash execution:
`19fbe3d2e7134b785f35dd34f3776ee2bccaf956a4d8f7a638dddce78de99ed6`.
Все **180 frozen source files** сверены по Git blobs execution commit,
source hash пересчитан независимо. Из существующих файлов меняются только
четыре публикационных Markdown и реестр; модель, kernels, trainer, config,
policy, experiment tests и guards остаются прежними.

Все **25 checkpoint** при этой публикации проверены потоковым SHA256 на
кластере: 12 новых и 13 ранее опубликованных. Hash каждого совпал с result
и checkpoint metadata. Веса не копировались и не десериализовались.
[Единый индекс](sources.json) содержит raw/metadata пути и SHA, удалённый
checkpoint путь/SHA, run identity, split, роль пилота/подтверждения, job и commit.
Старые preservation manifests не переписывались ради новой даты проверки.

Сохранены исходные policy и admission/smoke от pilot job4358147. Их нельзя
приписывать job4358583. Исторические exact-zero FAIL не отменены; ограниченный
численный допуск не означает глобальную математическую эквивалентность.

## Проверки данных и пересчёта

- Identity, status, mode/seed, TEST=NOT_RUN/count0; все 25 научных строк.
- Config/effective-config каждого MIMO режима совпадают с его пилотом после
  исключения только seed и checkpoint_dir. Dataset/policy/core не менялись.
- Начальный backbone, RNG, first batch, optimizer и precision совпадают
  внутри MIMO троек. Для SISO богатые отдельные RNG/optimizer/precision записи
  есть на четырёх новых seeds; pilot содержит только более узкие hashes.
- Новые MIMO histories содержат 563 последовательные эпохи; все 12 метрик
  каждой эпохи совпали с process logs. Best metrics/diagnostics совпали с
  history, epoch/metrics — с metadata. Dual2028 выбран по last-tie правилу:
  максимумы 36/38/46, checkpoint46.
- Все средние, sample std (`ddof=1`), парные std, знаки и relative differences
  пересчитаны из raw JSON. Отрицательные пары сохранены. Новых p-values нет.
- First27 всех трёх MIMO режимов: seeds2027,2028,2030, n=3. Отдельный
  dual/triple: seeds2027–2030, n=4. Короткие успешные fits не объявлены
  незавершёнными; отсутствующие эпохи не заполнены. Срез не является новой
  репликацией или равным GPU-бюджетом.
- CSV: **81 + 12 = 93**, 25 уникальных научных runs текущей линии.
  Первые **36 614 байт** CSV (заголовок и старые строки) неизменны:
  SHA256 `9c837008a31c643fa6a1e9e633e41c766a60c48d0db1fb815f28bc6d09aee3fa`.
  Повторный append не добавляет строк. TEST-таблицы и старый SISO SVG/TeX
  сохранены; нет средних, technical jobs или повторных pilot в качестве fits.

Воспроизводимая проверка без записей:

```bash
python3 -B reports/assets/internal_time_ablation/report.py --audit
python3 -B -m unittest discover -s reports/assets/internal_time_ablation -p 'test_*.py'
python3 -B reports/assets/three_time_confirmation/report.py --audit
```

Новые тесты относятся только к источникам, арифметике, first27, registry,
производным файлам, ссылкам и сохранности старых результатов. Старый SISO
`test_report.py` содержит историческое ожидание 78 строк (до MIMO pilot);
его целиком не запускаем и не переписываем под 93. Experiment tests не
запускались: они могут импортировать модель/создавать optimizer.

Итог локальных проверок: **7/7 reporting tests PASS**, оба read-only helper
с `--audit` завершились успешно, `git diff --check` чистый. Проверены 145
локальных ссылок в обновлённой навигации и отчётах. Исторический раздел
основного отчёта после SISO сохранён побайтно.

## SVG и TeX

[MIMO SVG](mimo_paired_delta.svg) и [общая TeX-таблица](siso_mimo_valid_table.tex)
сгенерированы тем же stdlib helper из [одной агрегации](summary.json).
SVG осмотрен в Chrome: пять отдельных точек, нулевая линия, отдельный символ
пилота, отрицательные2026/2029, все подписи видны. Линий между seeds и CI нет.
Существующий SISO график не менялся.

TeX: English headers, четыре блока, booktabs, n_seeds, mean ± sample std,
явный VALID. Сопоставимой серии SISO base нет, поэтому её среднего в таблице
нет. Проверены числа, структура столбцов и баланс скобок. **Компиляция LaTeX
не выполнялась:** pdflatex и tectonic недоступны. Новые библиотеки и шрифты
не устанавливались. В документе требуется `\usepackage{booktabs}`:

```tex
\input{reports/assets/internal_time_ablation/siso_mimo_valid_table}
```

Это отдельный файл; облачные main.tex, Introduction, Related Work и
results_overview.tex не заменялись.

## Границы публикации

Кластер использован только для чтения. Execution checkout остаётся на
`exp/mamba3-mimo-confirmation`, commit5670e898; он не обновляется под main.
Новых jobs, model computations, TRAIN/VALID/TEST evaluation, optimizer steps,
gates/smoke при публикации: **0**.

За рамками merge оставлены `codex/close-experimental-stage` (частичная MTL
серия), `exp/mamba3-input-time` (технический отказ, scientific modes NOT_RUN),
`exp/mamba3-prototypes-controls` (подготовка random-init) и `exp/ple-tim4rec`
(подготовка baseline). Ветки challengers/target-screening, не входящие в историю main, содержат
исторические результаты, уже выборочно опубликованные; это не повод мержить
их заново. Untracked cluster каталоги input_time, stage_confirmation,
target_combination_analysis, старые SISO partial summaries и MIMO attempt002
остались как были. Ничего из них не отбиралось без соответствующего плана.

Следующая гипотеза head-timescales не разрабатывалась; новой ветки и
STUDY_DESIGN для неё нет. Сообщение Дмитрию готовится отдельно и автоматически
не отправляется.
