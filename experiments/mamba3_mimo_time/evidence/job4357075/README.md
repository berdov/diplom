# Сохранение неудачной попытки 002

Job 4357075: FAILED, exit 1:0, 00:18:19, cn-043.
Execution: `5e8335f3a56af68bac74aac975b749ca6214b371`.
Source: `b9375435bac4bec10ec0cc061eb58c5f55524a969282217d8af4911a36dd3868`.

CPU-preflight PASS; 28 admission cases PASS. Case
`prefix_base_L17_P7_x1` остановился с
`RuntimeError: can't retain_grad on Tensor that has requires_grad=False`:
embedding hook оставался активным во время intervention под `torch.no_grad()`.
Полный admission не пройден. Частичные измерения этого case не были сохранены
и не восстанавливаются задним числом. Smoke, scientific fits и TEST не начались.

`preservation_manifest.json` содержит исходные пути, размеры и SHA256 35 файлов.
Оригиналы и копии сверены побайтно по SHA256; старые reservation, locks, логи,
JSON и summary не изменены. Scientific results/checkpoints/metadata/locks и
runtime-каталоги отсутствовали (16 путей в manifest). Admission JSON включает
исходный traceback. Кеши, datasets, окружение и модели сюда не копировались.

`scientific_settings.json` получен из чистого execution checkout через
`config.settings(mode)`, без модели, forward, evaluation или fit.
Это evidence ошибки обвязки, не разрешение на обучение и не численный PASS.
