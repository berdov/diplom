# SSD4Rec baseline для KuaiRand Protocol B

Цель ветки `exp/ssd4rec-baseline` - подготовить воспроизводимый запуск официального SSD4Rec на уже зафиксированном KuaiRand Protocol B. На этой ветке выполнены smoke test и 5-epoch sanity на полном train split; полный 300-epoch запуск пока не стартовал.

## Состав

- `AUDIT.md` - русскоязычный аудит статьи, кода, протокола и рисков воспроизведения.
- `config_kuairand.yaml` - рабочий RecBole-конфиг для KuaiRand Protocol B.
- `environment.txt` - требования и отдельный путь окружения на кластере.
- `smoke_test.py` - короткая GPU-проверка механики official SSD4Rec без полного обучения.
- `sanity_train.py` - 5-epoch sanity run на полном Protocol B, только с validation.
- `UPSTREAM_PATCHES.md` - описание upstream snapshot и runtime shims.
- `upstream/` - минимальный снимок official SSD4Rec code без датасетов.
- `runs/` - компактные JSON/MD результаты smoke и sanity.

## Smoke

Запуск на кластере:

```bash
sbatch slurm/ssd4rec_baseline.sh
```

Проверка должна пройти на E-node GPU и подтвердить:

- import `torch`, `recbole`, `mamba_ssm`;
- чтение Protocol B `kuairand.inter`;
- создание custom SSD4Rec variable-length dataloaders;
- один train loss, backward и optimizer step;
- один full-ranking validation batch.

Последний успешный smoke:

- [runs/smoke_20260819T110252Z.json](runs/smoke_20260819T110252Z.json)
- [runs/smoke_20260819T110252Z.md](runs/smoke_20260819T110252Z.md)

## Sanity

Пятиэпоховый sanity run на полном KuaiRand Protocol B:

```bash
SSD4REC_STAGE=sanity sbatch --job-name=ssd4rec-sanity --partition=test slurm/ssd4rec_baseline.sh
```

Последний успешный sanity:

- run id: `ssd4rec_sanity_001`;
- Slurm job: `4264406`;
- partition: `test`;
- constraint: `type_e`;
- node: `cn-046`;
- GPU: `NVIDIA A100-SXM4-80GB`;
- status: `COMPLETED`, exit code `0:0`;
- JSON: [runs/ssd4rec_sanity_001.json](runs/ssd4rec_sanity_001.json);
- notes: [runs/ssd4rec_sanity_001_notes.md](runs/ssd4rec_sanity_001_notes.md).

## Что не сделано этим коммитом

Полное 300-epoch обучение SSD4Rec не запускалось. Следующий отдельный run должен использовать этот каталог, фиксировать `run_id`, сохранить compact JSON/notes в `runs/`, а тяжелые checkpoints/logs держать на cHARISMa вне Git. Sanity run не добавлен в `experiments/results.csv`, потому что это проверка pipeline, а не итоговый эксперимент.
