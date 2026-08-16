Репозиторий дипломной работы ученика 4 курса НИУ ВШЭ СПБ Бердова Игоря Вячеславовича.

Проект исследует рекомендательные системы на датасете KuaiRand: структуру interaction logs, standard/random exposure, feedback signals, sequential histories, leakage risks и возможные benchmark protocols.

## Структура

- `notebooks/01_kuairand_eda.ipynb` - основной исследовательский EDA notebook.
- `src/eda_utils.py` - общие функции для inventory, path validation, lazy scans, summaries.
- `src/eda_27k.py` - memory-efficient aggregation script для полного KuaiRand-27K.
- `slurm/eda_27k.sh` - шаблон Slurm job для запуска 27K EDA на cHARISMa.
- `outputs/eda/` - локальная папка для генерируемых summary-файлов; содержимое игнорируется Git.

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

Для будущего Slurm-запуска 27K EDA сначала заполните cluster-specific TODO в `slurm/eda_27k.sh`, затем:

```bash
sbatch slurm/eda_27k.sh
```

Скрипт пишет `outputs/eda/27k_summary.json` и небольшие CSV summary tables.

## Git workflow

Работа ведется в ветке `eda`, которая отслеживает `origin/eda`.

Не коммитить:

- локальные Python environments (`.conda/`, `.venv/`);
- KuaiRand datasets;
- CSV/parquet exports;
- generated outputs;
- notebook checkpoints.
