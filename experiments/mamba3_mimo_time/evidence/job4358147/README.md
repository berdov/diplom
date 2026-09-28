# Завершённый MIMO pilot, job 4358147

KuaiRand: хронологический leave-one-out, полный каталог. Только TRAIN → VALID;
TEST=NOT_RUN, test_evaluation_count=0. Три scientific fits на seed2026 завершены.

Исходные JSON и компактные логи сохранены по первоначальным путям без изменений.
[Manifest](preservation_manifest.json) содержит SHA256 всех 52 копий, потоковые
SHA256 трёх checkpoint и проверку 158 исходных файлов против execution commit
`c1dd31eec7907c67348769b6a1811b89aa4011c0`. Веса не копировались и не загружались.

[Admission](../../runs/attempt_003/admission_001.json): 45/45 cases и 2342/2342
обязательных checks, по `mimo_numeric_acceptance_v1` с документированными
численными ограничениями. Исторические exact-zero failures не переименованы в PASS.
[Smoke](../../runs/attempt_003/smoke_001.json): PASS; технические шаги не являются
дополнительными scientific fits.

[Исходная сводка](../../runs/attempt_003/pilot_summary.md) проверена по raw JSON:
непрерывные histories, последняя эпоха при равном максимуме, best metrics и
diagnostics, metadata checkpoint и число эпох в логах согласованы.
Первые 27 эпох у base/triple представлены только 23 наблюдениями; окно неполное.
