# RT-Mamba3

Временной вариант primary-only Mamba3: интервалы между историческими взаимодействиями масштабируют native DT. [Входы и временные ограничения](../../reports/EVALUATION_SETUP.md).

## Результаты

| Split | HR=Recall@5 | @10 | @20 | @50 | NDCG@5 | @10 | @20 | @50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| [VALID](runs/mamba3_timeaware_validation_001.json) | 0.0682 | 0.1111 | 0.1800 | 0.3204 | 0.0468 | 0.0605 | 0.0778 | 0.1055 |
| [TEST](runs/mamba3_timeaware_final_test_001.json) | 0.0683 | 0.1116 | 0.1764 | 0.3136 | 0.0475 | 0.0613 | 0.0776 | 0.1046 |

Лучшая эпоха 15 (с нуля), выполнено 27 эпох. Один seed, один TEST выбранного по VALID checkpoint; повторный TEST и post-test tuning не предусмотрены. [Сравнения с baselines](../../reports/RESULTS.md) и [опубликованными числами](../../reports/PAPER_RESULTS.md) разделены.

## Модель

[Модель](model.py) наследует vanilla embeddings, два mixer, residual/FFN и tied scorer. Один calibrator используется обоими слоями; всего 610506 параметров. В [адаптации forward](time_mamba3.py):

```text
DT_base = softplus(dd_dt + dt_bias)
DT_real = DT_base * time_scale(history_gap)
ADT_real = A * DT_real
```

DT влияет на scan/input weights и rotary accumulation, ADT на decay. A, B/C RMSNorm, Trap, Angles, gating, D и output projection сохранены. Поддержан только SISO full-sequence, без cache/step/packed sequences. Основа: [pinned upstream](https://github.com/state-spaces/mamba/blob/e9594ce1c732d97440f0332fdc43170a2294dbfa/mamba_ssm/modules/mamba3.py), [лицензия](LICENSE.upstream); установленный пакет не редактируется.

[Calibrator](time_inputs.py): log1p(gap/reference), Linear(1,16), SiLU, Linear(16,2), exp(log(2)*tanh). Последний Linear нулевой: начальный scale=1, первый event/padding всегда нейтральны. При первом backward градиент первого Linear нулевой; после обновления последнего слоя он проходит дальше. Reference 838393 ms взят из [TRAIN statistics](runs/train_time_stats_001.json); bounds [0.5,2] заданы до эксперимента. [Dataset adapter](dataset.py) восстанавливает float64 history timestamps, не меняя float32 sort; [precision probe](precision_probe.py) является отдельной синтетической проверкой RecBole.

## Конфигурация и кластерные entrypoints

[config.py](config.py) объединяет baseline YAML и [overlay](config_kuairand.yaml); overlay нельзя передавать как самостоятельный полный конфиг. [TRAIN/VALID runner](run.py) выбирает checkpoint по VALID NDCG@10, не оценивая TEST. [HSE launcher](../../slurm/mamba3_timeaware_validation.sh) использует существующий envs/mamba3 и режимы smoke/train; smoke не является результатом качества.

Отдельные [TEST runner](mamba3_timeaware_final_test.py) и [launcher](../../slurm/mamba3_timeaware_final_test.sh) обслуживали единственный финальный TEST. Lock и существующий JSON блокируют повтор. Пути checkpoint, hashes, jobs и предупреждения записаны в исходных JSON; checkpoint и raw logs не входят в Git.

## Проверки

В локальном CPU-окружении с PyTorch, pytest, PyYAML, einops и NumPy:
`python -m pytest experiments/mamba3_timeaware/tests -q`.
Проверяются bounds, gaps, padding, causality, формы, градиенты, config/parser; это не GPU evidence.

Исторические [GPU equivalence](runs/gpu_equivalence_001.json) и [smoke](runs/mamba3_timeaware_smoke_001.json) завершены. Equivalence сравнивала official mixer и адаптацию при scale=1, bf16, eval/train, L=50/64: output/input-gradient errors=0, atol=1e-6, rtol=1e-5. Новые проверки не требуются для чтения результатов.
