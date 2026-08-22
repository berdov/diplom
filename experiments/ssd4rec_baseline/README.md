# SSD4Rec baseline для KuaiRand Protocol B

Цель ветки `exp/ssd4rec-baseline` - подготовить и выполнить воспроизводимый запуск официального SSD4Rec на уже зафиксированном KuaiRand Protocol B. На этой ветке выполнены smoke test, 5-epoch sanity и один full reproduction run `ssd4rec_001` с early stopping по validation `NDCG@10`.

## Состав

- `AUDIT.md` - русскоязычный аудит статьи, кода, протокола и рисков воспроизведения.
- `config_kuairand.yaml` - рабочий RecBole-конфиг для KuaiRand Protocol B.
- `environment.txt` - требования и отдельный путь окружения на кластере.
- `smoke_test.py` - короткая GPU-проверка механики official SSD4Rec без полного обучения.
- `sanity_train.py` - 5-epoch sanity run на полном Protocol B, только с validation.
- `full_train.py` - full run `ssd4rec_001`: максимум 300 эпох, patience 10, final test ровно один раз после загрузки best validation checkpoint.
- `UPSTREAM_PATCHES.md` - описание upstream snapshot и runtime shims.
- `upstream/` - минимальный снимок official SSD4Rec code без датасетов.
- `runs/` - компактные JSON/MD результаты smoke, sanity и full run.

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

## Full Run

Итоговый reproduction run на полном KuaiRand Protocol B:

```bash
SSD4REC_STAGE=full SSD4REC_RUN_ID=ssd4rec_001 SSD4REC_CONSTRAINT=type_h sbatch --job-name=ssd4rec-001 --partition=rocky --constraint=type_h --time=14:00:00 slurm/ssd4rec_baseline.sh
```

Фактический запуск:

- run id: `ssd4rec_001`;
- Slurm job: `4270754`;
- partition: `rocky`;
- constraint: `type_h`;
- node: `cn-050`;
- GPU: `NVIDIA H200 NVL`;
- status: `COMPLETED`, exit code `0:0`, elapsed `00:37:55`;
- JSON: [runs/ssd4rec_001.json](runs/ssd4rec_001.json);
- notes: [runs/ssd4rec_001_notes.md](runs/ssd4rec_001_notes.md).

Запрошено максимум `300` эпох, фактически выполнено `28` эпох из-за early stopping
с patience `10`; лучший validation checkpoint выбран на эпохе `17`. Final test
посчитан ровно один раз после загрузки этого checkpoint: `HR@10=0.1032`,
`HR@20=0.1683`, `NDCG@10=0.0576`, `NDCG@20=0.0739`.

Sanity запускался на `A100/type_e`; full run выполнен на `H200/type_h`, потому что
был найден более ранний Slurm slot. Модель, config, data split, seed, loss и
evaluation protocol не менялись. Тяжелые checkpoints/logs остаются на кластере
в `/home/daryumin/iberdov/diplom/experiments/ssd4rec_baseline/ssd4rec_001` и не
коммитятся в Git.
