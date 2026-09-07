# MOO representative challengers

Новый validation-only этап в ветке `exp/moo-representative-challengers`.
Дизайн, ограничения fidelity и оценка PHN overhead: [DESIGN.md](DESIGN.md).
Источники и лицензии: [provenance.yaml](provenance.yaml).
Параметры и selection rule: [config.yaml](config.yaml).

## Локальная проверка

Создать отдельный Python 3.11 environment, установить `requirements.unit.lock.txt`.
Официальный код хранится вне Git, по умолчанию `~/.cache/diplom-moo-references`.

```bash
python -m experiments.moo_representative_challengers.references --fetch
python -m experiments.moo_representative_challengers.verify
```

Dataset/GPU для этих проверок не нужны. `verification.json` содержит отдельные
проверки и hash исходников; после изменения кода/config его нужно пересоздать.
Перед Slurm реализация и сертификат должны быть committed и pushed.

## Кластер

Checkout: `/home/daryumin/iberdov/diplom_exp_moo_challengers`.
Существующий environment разрешено использовать только read-only и после проверки
импортов. Если требуется новая установка — отдельный runtime здесь; версии фиксируются
до установки. Ни shared TiM4Rec env, ни processed data не изменять.

Из checkout отправлять по одному методу, указав проверенный Python:

```bash
python -m experiments.moo_representative_challengers.submit \
  --method ferero --stage smoke --python /absolute/path/to/runtime/bin/python
```

Повторить для `most`, `phn_hvi`. После successful smoke каждого метода можно
отправить его `sanity`; после successful sanity — `convergence_screening`.
Gate проверяется до sbatch и повторно внутри job. Pending не является основанием
для повторного submit. Неполные runs не перезаписываются и сами не возобновляются.

Logs: `slurm_logs/`; job IDs: `submissions/`; small scientific JSON: `runs/`;
checkpoints/progress: `artifacts/<run_id>/` (не коммитить binaries).

```bash
python -m experiments.moo_representative_challengers.summarize
```

`summary.json` не заменяет старый Stage 1 или canonical `experiments/results.csv`.
После completed convergence собрать small JSON/notes отдельным results commit.
Никаких TEST, Optuna, PR, merge и выводов о замене представителей без review.
