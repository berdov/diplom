# SISO dual/triple: результаты пилота и аудит сопоставимости

Дата: 26.09.2026. Только сохранение существующих результатов и чтение evidence/исходников; новых jobs, forward/backward, TRAIN/VALID/TEST не было. Кластер использован только для чтения.

## Сохранение и происхождение

Raw results сохранены коммитом `660c7c0567ffeec45f1772f91255a4da213a99d8`: 12 файлов, 831 783 байта; SHA256 кластерных оригиналов и локальных копий совпали 12/12. [Manifest сохранения](../experiments/mamba3_three_time/validation_pilot/evidence/pilot_001/preservation_manifest.json) содержит исходные/локальные пути, размеры и SHA256. Сохранены admission, smoke, два scientific JSON, summary JSON/MD, submission/pipeline records, metadata checkpoint и два небольших stderr. Бинарные checkpoint не копировались и не десериализовались. Метрики и байты оригиналов не редактировались, включая исходные trailing spaces в логах.

Execution commit пилота: `6b5618a769d8e3424df0b7232605426fac8c573a`, job `4353980`; [submission](../experiments/mamba3_three_time/validation_pilot/evidence/pilot_001/slurm_logs/submission_001.json), [pipeline](../experiments/mamba3_three_time/validation_pilot/evidence/pilot_001/slurm_logs/pipeline_status.json). Pilot source hash: `105e09559cde506d0a6de46f8731213a5a6a8af0c56b20924d5c702c9ec87b58`. Все 277 файлов [frozen manifest](../experiments/mamba3_three_time/validation_pilot/source_manifest.json) сохранены без изменений; этот отчёт не входит в frozen sources. Numerical policy v1 и backend upstream не менялись; stable_adt/stable_scan не использовались.

## Результаты пилота

KuaiRand Protocol B, full-ranking VALID, seed 2026. Эпохи нумеруются с 0; оба run имеют `PASS`, `TEST=NOT_RUN`, `test_evaluation_count=0`.

| Режим | VALID NDCG@10 | HR@10 | Максимум NDCG@10 за 0-26 | Best epoch | Actual epochs | Параметры | TRAIN / VALID, с |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [Dual](../experiments/mamba3_three_time/validation_pilot/runs/mamba3_three_time_siso_dual_seed2026_001.json) | .0615 | .1142 | .0615 | 15 | 27 | 610572 | 593.23 / 11.18 |
| [Triple](../experiments/mamba3_three_time/validation_pilot/runs/mamba3_three_time_siso_triple_seed2026_001.json) | .0623 | .1148 | .0623 | 15 | 27 | 610638 | 600.22 / 10.50 |

Triple минус dual: **+.0008, +1.30%** NDCG@10. Первые 27 эпох совпадают с полным горизонтом обоих новых runs и не являются независимым подтверждением. Время включает JIT; точный steady-state overhead из него не следует.

## A/B/C: процедура и доступное evidence

Использованы JSON и исходники именно execution commits, а не текущий код вместо исторического:

- **A:** [историческая separate](../experiments/mamba3_time_mechanisms/runs/mamba3_separate_time_validation_001.json), job `4332920`, commit `9334bd93c8fa30fbacabaeb220cef17e336c261b`.
- **B:** [context separate_replay](../experiments/mamba3_context_time/runs/mamba3_context_separate_replay_seed2026_001.json), job `4337124`, commit `6feb8329334fe94d894c9d79f5a412a1241d5b3f`.
- **C:** новый dual из таблицы выше, job `4353980`, commit `6b5618a769d8e3424df0b7232605426fac8c573a`.

`NOT_RECORDED` означает отсутствие записи, а не несовпадение и не восстановление состояния из seed. «По коду» ниже не равно измеренному runtime flag.

| Признак | A: separate | B: separate_replay | C: dual |
| --- | --- | --- | --- |
| VALID NDCG@10; best / actual | .0633; 51 / 63 | .0633; 51 / 63 | .0615; 15 / 27 |
| Полный effective config | NOT_RECORDED; есть исходный config | Сохранён | Сохранён; B/C одинаковы, кроме класса модели, режимов и output path |
| Основные настройки | Общие настройки ниже, исходный config | Те же, effective config | Те же, effective config |
| Protocol B inter SHA / TRAIN stats SHA | Совпадают с B/C | Совпадают с A/C | Совпадают с A/B |
| Manifest SHA как поле run | NOT_RECORDED; связь через TRAIN stats | Сохранён, совпадает с C | Сохранён, совпадает с B |
| Исходные TRAIN / VALID / TEST размеры | 1086518 / 23951 / 23951 | Те же | Те же |
| Последовательный TRAIN / VALID | 1062567 / 23951, по assertions кода | Те же, по assertions кода | Те же, по assertions кода |
| Полный порядок batches/эпох | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED |
| Initial backbone SHA | NOT_RECORDED | `971a713918ae...` | `971a713918ae...`, совпадает с B |
| Initial calibrator SHA | NOT_RECORDED | NOT_RECORDED для separate_replay | decay = scan = `452efbe042f1...` |
| RNG перед fit, aggregate SHA | NOT_RECORDED | `f0ef83cd7cbd...` | `6ff5fb2fb57e...`, отличается от B |
| Первый потреблённый TRAIN batch SHA | NOT_RECORDED | `96290ff5095b...` | `96290ff5095b...`, совпадает с B |
| DataLoader: worker / shuffle / transform | Effective flags NOT_RECORDED | 0 / true / null; отдельный generator | То же |
| Precision | bf16 mixer по config, fp32 outer по коду | То же | То же |
| AMP / scaler / clipping | Effective flags NOT_RECORDED | false / false / null | То же |
| TF32 matmul / cuDNN | NOT_RECORDED | NOT_RECORDED | false / true |
| Determinism | reproducibility=true; фактические flags NOT_RECORDED | То же | То же |
| Optimizer / stopping / ties | Adam; checkpoint при >=, diagnostics при > | Adam; checkpoint и best diagnostics при >= | Как B, fit унаследован без изменения |
| Checkpoint | Полный RecBole snapshot | Pure state_dict + metadata/SHA | Как B; без warm start |
| Diagnostics | До embedding; reservoir 8192 | После embedding, до dropout; reservoir 2048 | Hook до embedding; два collector по 2048 |
| Backend | Official combined SISO wrapper | Тот же | Split-DT autograd wrapper на upstream primitives |
| Runtime / GPU | Общие версии ниже; A100-SXM4-80GB | Те же; cn-045 | Те же; cn-043 |
| TEST count | 0 | 0 | 0 |

Общие настройки: Adam lr=.001, weight_decay=0, CE; batch TRAIN=2048, eval_batch_size=4096; history=50; epochs=300, eval_step=1, stopping_step=10; full-ranking VALID NDCG@10, metric_decimal_place=4. hidden=64, layers=2, dropout=.2, d_state=128, expand=2, headdim=64, ngroups=1, rope_fraction=.5, chunk=64, SISO; temporal heads=2, scale bounds [.5, 2], TRAIN reference=838393. У B/C также одинаковы single_spec=true, repeatable=true, valid_metric_bigger=true, отсутствие negative sampling; все остальные effective поля совпали. У A raw config совпадает после исключения названий режимов и checkpoint path, но это не заменяет отсутствующий effective config.

Общие версии: torch 2.9.1+cu128, CUDA 12.8, RecBole 1.2.0, mamba-ssm 2.3.2.post1, Triton 3.5.1, NumPy 1.26.4, TileLang 0.1.8, apache-tvm-ffi 0.1.9. Pinned Mamba: `e9594ce1c732d97440f0332fdc43170a2294dbfa`; core: `460bd3e687d26a458cafc21960391f83905da962ca5c992d6062c7b048f0a73f`. Совпадение версий не доказывает идентичность всех исторических installed files: их SHA в A/B не записаны. init_seed при reproducibility=true запрашивает cudnn.benchmark=false/deterministic=true; отдельного runtime snapshot этих flags нет.

Данные: inter SHA `e275ded0b330c2827b49ccf567d6784452d6dcbf8cd719dc3009d36eadc2e2cc`; manifest SHA B/C `9f39aa12ec16f697a8bedb91eeb21c46dce514e6f414b7e47fc4064c1fcffd2c`; TRAIN stats SHA `fa5df0e5ec97d84e5dffd157373318ebfa2cb94fe2853c2aec4898ecd5f89943`. TRAIN history stats совпадают. PreciseHistoryDataset одинаков в execution sources A/B/C: исходная RecBole сортировка по timestamp, сохранение float64 history timestamps. Равный hash входа/первого batch не является доказательством полного порядка всех эпох.

## Что именно означает сравнение hashes

Функции `tensor_hash`/`rng_hash` в [provenance B](../experiments/mamba3_context_time/provenance.py) не менялись между execution commits B/C. Tensor hash включает отсортированные keys, dtype, shape и contiguous CPU bytes. Из backbone исключены `mechanisms.*` у B и `times.*` у C; оставшиеся backbone keys одинаковы. Общий SHA: `971a713918ae4f6309805e5a78ded9800c4137d171222dd87791388d2f919555`.

Calibrator hashes B отсутствуют: нельзя объявить совпадение/различие весов по C или по разным префиксам. Если сравнивать их в будущем, требуется явно нормализовать `mechanisms.calibrators.<name>` и `times.calibrators.<name>`; в этом аудите отсутствующие веса не реконструировались.

RNG hash объединяет Python, NumPy, CPU torch и инициализированные CUDA RNG states. B: `f0ef83cd7cbd93e427a84d85cd300aa92cd5dec15c0c693ef8410f99eaf1ae8b`; C: `6ff5fb2fb57e3a4f80802afe70cf426918971cc6464d8f05d781a9cbb4f7320f`. Aggregate digest не локализует различие по компонентам. First-batch hash считается одной схемой по всему уже потреблённому Interaction, не по предварительному пробному iterator; B/C совпали: `96290ff5095b675be399fef25e4db30b014f4464488d69e085fd42a48e1edcfc`.

## Подтверждённые отличия и границы объяснения

1. **Конструктор и CPU RNG.** [TimeMechanisms](../experiments/mamba3_time_mechanisms/time_mechanisms.py) создаёт первый TimeCalibrator без изоляции RNG, второй через deepcopy. Linear initialization потребляет CPU RNG даже у слоя, затем обнулённого. [ContextMamba3Rec](../experiments/mamba3_context_time/model.py) для separate_replay сохраняет этот путь. [ThreeTimes](../experiments/mamba3_three_time/calibrators.py) создаёт calibrators внутри `fork_rng(devices=[])`: CPU state восстанавливается, CUDA states не входят в этот fork. Это реальное различие потребления CPU RNG; aggregate pre-fit digest B/C также отличается.
2. **Цепочка до shuffle/dropout не доказана.** DataLoaders создаются до модели. В просмотренном RecBole AbstractDataLoader есть собственный `torch.Generator().manual_seed(config['seed'])`; Torch RandomSampler и iterator base_seed используют переданный generator. B/C имеют worker=0 и transform=null. Поэтому нельзя автоматически приписать иной shuffle глобальному CPU RNG. CUDA dropout после `.to(cuda)` использует CUDA RNG, а не восстановленный CPU RNG. После конструктора просмотрены trainer/Adam setup и fit; подтверждённой цепочки CPU RNG -> порядок batches либо CUDA dropout не найдено. Совпавший первый batch не доказывает всю эпоху; компонентные RNG states не сохранены.
3. **Вычислительный путь действительно другой.** A/B вызывают official `mamba3_siso_combined`; C использует local split-DT autograd wrapper и upstream forward/backward primitives. У dual один scan tensor передаётся для write/phase, gradients суммируются через новый wrapper. Admission B подтвердил нулевые ошибки output/parameter gradients на сохранённых nonzero-calibrator synthetic fixtures L50/64, eval/train, batch=2. Это локальное evidence, не доказательство идентичной траектории всего TRAIN при batch=2048. stable_adt/stable_scan здесь не участвовали.
4. **Учёт и checkpoint.** Все diagnostics работают без gradients, reservoir имеет независимый CPU generator; место hook, размер выборки и схема diagnostics отличаются. Их quantiles нельзя механически считать одинаковыми измерениями. C наследует B fit/checkpoint semantics, добавляя diagnostics/timing/CUDA synchronization. A сохраняет полный RecBole snapshot, B/C сохраняют безопасный state_dict; загрузка checkpoint не предшествует fit. При ties fit использует >=, а A diagnostics callback строго >; это различие учёта, не объяснение расхождения уже на эпохе 0.

В записанных B/C training flags и данных **конкретный configuration/data drift не найден**, кроме явного выбора нового режима/класса модели. Нельзя заключить «изменился только RNG». Влияние CPU-state, нового compute path, runtime flags или численной недетерминированности на метрику остаётся гипотезами; причина расхождения **UNKNOWN**. Различия TF32 A/B против C не установлены: у A/B нет записи.

## Истории и early stopping

История A сверена с уже опубликованным [extracted.json](assets/time_mechanisms/extracted.json) (SHA `fa02a085e687a274904b2d2958ab6fec8cdd84ede50a3442f7320f9dbcb3f8da`) и существующим кластерным `experiments/mamba3_time_mechanisms/slurm_logs/m3-separate-val-4332920.err` (SHA `53cb0e1a73da809f1e60a8f61f092903b6af08696c7b5fa025198e7589e69101`). A/B имеют одинаковые 63 значения NDCG@10; train loss совпадает до 4 знаков, доступных в историческом логе. Не заявляется совпадение несохранённых unrounded losses A.

| Признак | A | B | C |
| --- | ---: | ---: | ---: |
| Epoch 0 TRAIN loss | 3886.6419 (лог) | 3886.6418676376343 | 3889.1561794281006 |
| Epoch 0 VALID NDCG@10 | .0442 | .0442 | .0437 |
| Лучший NDCG@10 за 0-26 | .0614 (epoch 23) | .0614 (epoch 23) | .0615 (epoch 15) |
| Полный лучший NDCG@10 | .0633 (epoch 51) | .0633 (epoch 51) | .0615 (epoch 15) |

Первое B/C расхождение TRAIN loss и VALID уже на **epoch 0**. За 0-26 C выше B на .0001 (+.16%), но по полным завершённым runs ниже на .0018 (-2.84%). B после этого окна обновлял максимум: epoch 28=.0617, 29=.0620, 35=.0621, 45=.0622, 48=.0627, 51=.0633. У C после best=15 идут 11 худших эпох 16-26: штатное `cur_step > stopping_step` остановило fit. Лимит 27 эпох не задавался; B/C имели одинаковые epochs=300/stopping_step=10. Новый run не продлевался ради достижения epoch 51.

## Вывод и минимальный следующий шаг

**Triple лучше dual в новом парном запуске seed2026. Превосходство над исторической separate и устойчивость на дополнительных seeds пока не установлены.** Внутреннее сравнение пилота не аннулируется более высокой исторической метрикой, но новый лучший результат проекта не заявляется.

Новую парную серию можно продолжать как отдельную серию с собственным dual comparator, одинаковой frozen процедурой и парными seeds. Старые separate/separate_replay нельзя включать в неё как взаимозаменяемые повторения нового dual; MIMO этим пилотом не разрешается. Новых запусков сейчас нет.

Если требуется объяснить историческое расхождение, минимальный следующий шаг под отдельное разрешение: один контролируемый парный TRAIN batch/optimizer step B/C при явно одинаковых начальных параметрах и RNG, с фиксацией Python/NumPy/CPU/CUDA/DataLoader states до/после конструктора и сравнением loss/gradients/update. Без полного fit, VALID/TEST, подбора параметров или искусственного потребления RNG в scientific коде. В данном запросе эта проверка **не выполнялась**.
