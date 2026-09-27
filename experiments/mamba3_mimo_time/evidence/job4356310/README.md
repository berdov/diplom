# Неудачная инфраструктурная попытка 4356310

Slurm: FAILED, exit 1:0, 00:05:12, cn-043, 27.09.2026.
Execution: f4938792ab3415e431320d5ca7c2d6ce042ab35a.
Source: 52c903ac4d2212ae8d81645079e424265b30c0f0733a3b48ec70571de8f9fc12.

Остановка на preflight: `ValueError('CPU preflight initialized CUDA')`.
Первый вызов, инициализировавший CUDA, в traceback не установлен.
Запись CUDA_VISIBLE_DEVICES внутри RecBole сама по себе не доказывает его причину.

Admission и smoke не начались, scientific fits 0/3, TRAIN/VALID/TEST=0.
В сводке все три модели NOT_RUN. Проверено отсутствие admission/smoke JSON,
трёх scientific JSON, locks, checkpoints и metadata (14 путей).

20 исходных небольших файлов скопированы без изменения bytes; исходники
reservation/logs/summary на кластере не удалялись и не перезаписывались.
SHA256 и пути приведены в `preservation_manifest.json`.
`scientific_settings.json` отдельно получен из `config.settings(mode)` исходного
checkout без model construction/forward. Это снимок конфигураций, не результат.
Фиктивные admission/scientific JSON не создавались.

Повторная попытка требует отдельного namespace, reservation и явной проверки
именно этой причины отказа и отсутствия начавшихся scientific fits.
