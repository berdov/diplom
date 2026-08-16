Репозиторий дипломной работы ученика 4 курса НИУ ВШЭ СПБ Бердова Игоря Вячеславовича.

Проект исследует рекомендательные системы на датасете KuaiRand: структуру interaction logs, standard/random exposure, feedback signals, sequential histories, leakage risks и возможные benchmark protocols.

## Структура

- `notebooks/01_kuairand_eda.ipynb` - основной исследовательский EDA notebook.
- `src/eda_utils.py` - общие функции для inventory, path validation, lazy scans, summaries.
- `src/eda_27k.py` - memory-efficient aggregation script для полного KuaiRand-27K.
- `slurm/eda_27k.sh` - шаблон Slurm job для запуска 27K EDA на cHARISMa.
- `outputs/eda/` - компактные 27K summary-файлы; raw данные и logs не коммитятся.
- `reports/kuairand_27k_eda_report.md` - итоговый отчёт по полному KuaiRand-27K EDA.

## Данные и окружение

Код пишется локально, но реальные данные KuaiRand ожидаются на cHARISMa:

```text
/home/daryumin/iberdov/Corpora/KuaiRand-Pure/KuaiRand-Pure/
/home/daryumin/iberdov/Corpora/KuaiRand-1K/KuaiRand-1K/
/home/daryumin/iberdov/Corpora/KuaiRand-27K/KuaiRand-27K/
```

Серверное окружение Python:

```text
/home/daryumin/iberdov/diplom/.conda
```

Jupyter kernel: `Python 3.11 — KuaiRand`.

Notebook использует `pathlib.Path` и по умолчанию ищет данные в `/home/daryumin/iberdov/Corpora`. Если каталог не найден, он сообщает `KuaiRand data directory not found` и не скачивает данные.

## Как выполнить EDA

На cHARISMa:

```bash
cd /home/daryumin/iberdov/diplom
git pull
jupyter lab notebooks/01_kuairand_eda.ipynb
```

Основной notebook рассчитан на Pure и 1K для интерактивного EDA. Полный 27K не загружается целиком в notebook; для него подготовлен отдельный lazy aggregation script.

Полный KuaiRand-27K EDA был выполнен на cHARISMa через Slurm. Использован E-node (`constraint=type_e`) через partition `rocky`; Slurm policy запрещает запуск type_e без GPU GRES, поэтому job запрашивает минимальный `gpu:a100:1`, но сам EDA-код CPU-only и не использует GPU.

Для повторного Slurm-запуска 27K EDA:

```bash
sbatch slurm/eda_27k.sh
```

Скрипт пишет `outputs/eda/27k_summary.json` и небольшие CSV summary tables. Sanity/log files остаются локальными generated artifacts.

## Git workflow

Работа ведется в ветке `eda`, которая отслеживает `origin/eda`.

Не коммитить:

- локальные Python environments (`.conda/`, `.venv/`);
- KuaiRand datasets;
- произвольные CSV/parquet exports, кроме компактных `outputs/eda/27k_*` summaries;
- generated outputs/logs/sanity files;
- notebook checkpoints.
