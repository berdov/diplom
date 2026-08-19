# SSD4Rec baseline для KuaiRand Protocol B

Цель ветки `exp/ssd4rec-baseline` - подготовить воспроизводимый запуск официального SSD4Rec на уже зафиксированном KuaiRand Protocol B. Полное обучение в этом коммите не запускается.

## Состав

- `AUDIT.md` - русскоязычный аудит статьи, кода, протокола и рисков воспроизведения.
- `config_kuairand.yaml` - рабочий RecBole-конфиг для KuaiRand Protocol B.
- `environment.txt` - требования и отдельный путь окружения на кластере.
- `smoke_test.py` - короткая GPU-проверка official SSD4Rec mechanics без полного обучения.
- `UPSTREAM_PATCHES.md` - описание upstream snapshot и runtime shims.
- `upstream/` - минимальный снимок official SSD4Rec code без датасетов.
- `runs/` - compact JSON/MD результаты smoke.

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

## Что не сделано этим коммитом

Полное обучение SSD4Rec не запускалось. Следующий отдельный run должен использовать этот каталог, фиксировать `run_id`, сохранить compact JSON/notes в `runs/`, а тяжелые checkpoints/logs держать на cHARISMa вне Git.
