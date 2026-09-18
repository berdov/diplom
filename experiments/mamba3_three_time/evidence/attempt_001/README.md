# Первая техническая попытка: FAIL

Job `4337949`, execution commit `1bff0863e13c90161b6828c1905b2f5a0ab62a14`.
SISO: 34 PASS / 1 FAIL; MIMO: native backward FAIL; общий FAIL.

Оригинальные файлы скопированы без редактирования; SHA256 всех 15 копий
проверены с кластером в `preservation_manifest.json`. План и source manifest
относятся только к первому immutable execution commit, не к будущим ревизиям.

SISO `E_causality_padding.future_gradient_zero`: max_abs
`8.477159252340272e-11` при atol=rtol=0. Причина пока не локализована.
MIMO rank4/chunk16: official backward запросил 223904 bytes dynamic shared
memory; исключение и исходные статусы сохранены. Chunk8 будет новой попыткой,
а не переинтерпретацией этого результата. Научные fits/TRAIN/VALID/TEST = 0.
