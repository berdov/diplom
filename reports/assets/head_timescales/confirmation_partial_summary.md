# Подтверждающая серия: ожидание остановлено по лимиту

**INCOMPLETE.** На 2026-09-30T15:26:54.305222+00:00 job4365206 оставался PENDING. За отведённые4 часа очереди не начался ни один из12 fits. Запрошена одна A100 на8 часов; фактически использовано0 GPU-часов на момент проверки. Второй job не отправлялся, TEST=0.

Это сводка состояния сессии, а не terminal report. NOT_RUN в таблице означает «не начат на момент сохранённого наблюдения». Job не отменён и может начаться позднее; после этой проверки мониторинг остановлен.

Новые2027–2030: n=0/4, метрики и sample std отсутствуют. В группе all5 доступен только exploratory pilot2026 (n=1/5), поэтому её нельзя считать подтверждением на пяти seeds. Первичный контраст пилота head−shared=+0.0012; head−fixed=+0.0006; shared−fixed=−0.0006. В first27 они равны+0.0004,+0.0001,−0.0003. Все значения ниже получены из сохранённых pilot JSON.

[Опубликованный пилот](../../MAMBA3_TIME_MECHANISMS_RESULTS.md#head-timescales-pilot) · [JSON этой сводки](confirmation_partial_summary.json) · [Зафиксированный source index](../../../experiments/mamba3_head_timescales/confirmation/source_index.json) · [Submission evidence и SHA](../../../experiments/mamba3_head_timescales/confirmation/evidence/submission_4365206/preservation_manifest.json) · [Последнее наблюдение и бюджет](../../../experiments/mamba3_head_timescales/confirmation/evidence/submission_4365206/queue_limit.json).

Execution `0de6c380bb531b93a08e4915508b578af77fff3e`, source `9288aa3386a652f15231ccdbc0116c963a0cb9ccb8d1d361b868999da51922d2`. Кластерный checkout остаётся на execution commit и tracked-clean. Main содержит опубликованный пилот `a45e181471c511cc600ed829f6f6a70957a330d8`; реестр93→96 после пилота, подтверждающих строк добавлено0. Неполная серия остаётся в ветке `exp/mamba3-head-timescales-confirmation`.

Отдельный аудит завершённых fits, paired-delta график и TeX подтверждения пока не выполнялись: новых результатов нет. Скрипты публикации подготовлены, но завершённый раздел отчёта не создавался. Следующее безопасное действие в новой разрешённой сессии — проверить существующий job4365206 и его reservation; не повторять submit и не менять используемый checkout.

# Head-timescales confirmation

Status: INCOMPLETE; new fits 0/12; TEST=0.

| Seed | Variant | Status | NDCG@10 | HR@10 | Best epoch | Epochs | First27 complete | First27 |
|---|---|---|---:|---:|---:|---:|---|---:|
| 2026 | fixed | PASS | 0.0633 | 0.1162 | 27 | 39 | True | 0.062 |
| 2026 | shared_tau | PASS | 0.0627 | 0.1165 | 27 | 39 | True | 0.0617 |
| 2026 | head_tau | PASS | 0.0639 | 0.1191 | 48 | 60 | True | 0.0621 |
| 2027 | fixed | NOT_RUN | — | — | — | 0 | False | — |
| 2027 | shared_tau | NOT_RUN | — | — | — | 0 | False | — |
| 2027 | head_tau | NOT_RUN | — | — | — | 0 | False | — |
| 2028 | fixed | NOT_RUN | — | — | — | 0 | False | — |
| 2028 | shared_tau | NOT_RUN | — | — | — | 0 | False | — |
| 2028 | head_tau | NOT_RUN | — | — | — | 0 | False | — |
| 2029 | fixed | NOT_RUN | — | — | — | 0 | False | — |
| 2029 | shared_tau | NOT_RUN | — | — | — | 0 | False | — |
| 2029 | head_tau | NOT_RUN | — | — | — | 0 | False | — |
| 2030 | fixed | NOT_RUN | — | — | — | 0 | False | — |
| 2030 | shared_tau | NOT_RUN | — | — | — | 0 | False | — |
| 2030 | head_tau | NOT_RUN | — | — | — | 0 | False | — |

## new4_full

| Variant | Available / expected | Seeds | Mean ± sample std |
|---|---:|---|---:|
| fixed | 0/4 | [] | — |
| shared_tau | 0/4 | [] | — |
| head_tau | 0/4 | [] | — |

| Paired contrast | n / expected | Seeds | Δ mean ± std | + / 0 / − | Relative % |
|---|---:|---|---:|---:|---:|
| head_tau-shared_tau | 0/4 | [] | — | 0/0/0 | — |
| head_tau-fixed | 0/4 | [] | — | 0/0/0 | — |
| shared_tau-fixed | 0/4 | [] | — | 0/0/0 | — |

## new4_first27

| Variant | Available / expected | Seeds | Mean ± sample std |
|---|---:|---|---:|
| fixed | 0/4 | [] | — |
| shared_tau | 0/4 | [] | — |
| head_tau | 0/4 | [] | — |

| Paired contrast | n / expected | Seeds | Δ mean ± std | + / 0 / − | Relative % |
|---|---:|---|---:|---:|---:|
| head_tau-shared_tau | 0/4 | [] | — | 0/0/0 | — |
| head_tau-fixed | 0/4 | [] | — | 0/0/0 | — |
| shared_tau-fixed | 0/4 | [] | — | 0/0/0 | — |

## all5_full

| Variant | Available / expected | Seeds | Mean ± sample std |
|---|---:|---|---:|
| fixed | 1/5 | [2026] | 0.063300 (n=1) |
| shared_tau | 1/5 | [2026] | 0.062700 (n=1) |
| head_tau | 1/5 | [2026] | 0.063900 (n=1) |

| Paired contrast | n / expected | Seeds | Δ mean ± std | + / 0 / − | Relative % |
|---|---:|---|---:|---:|---:|
| head_tau-shared_tau | 1/5 | [2026] | 0.001200 (n=1) | 1/0/0 | 1.9138755980861122 |
| head_tau-fixed | 1/5 | [2026] | 0.000600 (n=1) | 1/0/0 | 0.9478672985782088 |
| shared_tau-fixed | 1/5 | [2026] | -0.000600 (n=1) | 0/0/1 | -0.9478672985781866 |

## all5_first27

| Variant | Available / expected | Seeds | Mean ± sample std |
|---|---:|---|---:|
| fixed | 1/5 | [2026] | 0.062000 (n=1) |
| shared_tau | 1/5 | [2026] | 0.061700 (n=1) |
| head_tau | 1/5 | [2026] | 0.062100 (n=1) |

| Paired contrast | n / expected | Seeds | Δ mean ± std | + / 0 / − | Relative % |
|---|---:|---|---:|---:|---:|
| head_tau-shared_tau | 1/5 | [2026] | 0.000400 (n=1) | 1/0/0 | 0.6482982171799101 |
| head_tau-fixed | 1/5 | [2026] | 0.000100 (n=1) | 1/0/0 | 0.1612903225806539 |
| shared_tau-fixed | 1/5 | [2026] | -0.000300 (n=1) | 0/0/1 | -0.4838709677419395 |

First27 uses only complete observed windows0–26, with independent pair subsets. Pilot2026 is exploratory. No significance, personalization or head-specialization claim. R denotes global normalization scales. VALID is not compared to published TEST.

Blocking reason: Queue wait reached4h with job4365206 still PENDING. Monitoring stopped; no fit was started at this observation.
