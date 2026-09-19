# Сохранённая attempt002

Execution: `d0c8a2356b235981436715e2d340803e1a0951a0`, job `4338153`.
Исходные файлы скопированы без изменений; SHA256 сверены с кластером.

SISO: FAIL structural regression combined stable_scan. ADT-only устранил
tiny future-gradient на prefix fixtures, но полная structural suite для него
в этой попытке не выполнялась. Причина substantial drift пока не установлена.
MIMO: native L17 forward прошёл; backward остановился на upstream assertion
`Sequence length 17 must be divisible by chunk_size 8`.

Известные дефекты старого evidence оставлены как есть: шесть isolated-angle
case IDs ошибочно равны `stable_scan`, их исходные progress records остались
RUNNING; ideal PyTorch elementary-function comparisons не входили в parent
status. Длины, prefix, loss multiplier и реальные ошибки сохранены в rows.
Это не исправление старых FAIL и не подтверждение полной correctness.
