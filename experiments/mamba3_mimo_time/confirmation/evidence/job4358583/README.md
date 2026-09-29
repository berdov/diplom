# Сохранённые результаты job4358583

Завершены 12 fits по [study plan](../../study_plan.json): base/dual/triple,
seeds 2027–2030. Execution commit:
`5670e898ed04924a929756e52f39d1d00eb79c5a`, source hash:
`19fbe3d2e7134b785f35dd34f3776ee2bccaf956a4d8f7a638dddce78de99ed6`.

[Manifest](preservation_manifest.json) содержит исходный и локальный путь,
размер и SHA256 каждого из 79 файлов (3 518 393 байта). Results, summary,
metadata и process logs скопированы побайтно из execution checkout.
Сохранённые preflight/CPU logs — исторические записи до запуска, а не
повторные проверки при публикации. Reservation token — идентификатор
однократного запуска, не учётные данные доступа.

SHA256 всех 12 checkpoint повторно прочитан потоково и совпадает с results
и metadata. Веса оставлены на кластере; в Git их нет, десериализации не было.
Admission и smoke унаследованы от job4358147, execution
`c1dd31eec7907c67348769b6a1811b89aa4011c0`: они не выполнялись в job4358583.
Исходные locks и старые failed attempts не менялись.

Статусы исходных summary/pipeline: PASS; scientific fits 12/12.
Эта серия использует только VALID; TEST=NOT_RUN, test_evaluation_count=0.
Сохранение выполнялось через SSH только для чтения. Новых jobs, evaluation,
model forward/backward, optimizer steps и повторных gates не было.

Итоги публикации: [основной отчёт](../../../../../reports/MAMBA3_TIME_MECHANISMS_RESULTS.md#mimo-time-confirmation).
