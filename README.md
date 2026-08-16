Репозиторий дипломной работы ученика 4 курса НИУ ВШЭ СПБ Бердова Игоря Вячеславовича.

Проект исследует рекомендательные системы на датасете KuaiRand: структуру логов взаимодействий, режимы сбора `standard`/`random`, feedback-сигналы, последовательные пользовательские истории, риски leakage и возможные benchmark-протоколы.

## Структура

- `notebooks/01_kuairand_eda.ipynb` - основной исследовательский EDA-ноутбук.
- `src/eda_utils.py` - общие функции для инвентаризации файлов, проверки путей, lazy scans и сводных таблиц.
- `src/eda_27k.py` - скрипт агрегации для полного KuaiRand-27K, рассчитанный на экономное использование памяти.
- `slurm/eda_27k.sh` - шаблон Slurm-задания для запуска 27K EDA на cHARISMa.
- `outputs/eda/` - компактные 27K summary-файлы; raw-данные и logs не коммитятся.
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

Ядро Jupyter: `Python 3.11 — KuaiRand`.

Ноутбук использует `pathlib.Path` и по умолчанию ищет данные в `/home/daryumin/iberdov/Corpora`. Если каталог не найден, он сообщает `KuaiRand data directory not found` и не скачивает данные.

## Как выполнить EDA

На cHARISMa:

```bash
cd /home/daryumin/iberdov/diplom
git pull
jupyter lab notebooks/01_kuairand_eda.ipynb
```

Основной ноутбук рассчитан на Pure и 1K для интерактивного EDA. Полный 27K не загружается целиком в ноутбук; для него подготовлен отдельный lazy-скрипт агрегации.

Полный KuaiRand-27K EDA был выполнен на cHARISMa через Slurm на V100. Финальный успешный запуск: job `4253874`, partition `test`, constraint `type_a`, GRES `gpu:v100:1`, node `cn-012`, 10 CPU, `mem=0`, runtime `00:09:45`, MaxRSS `46537000K`. EDA-код CPU-only, но GPU GRES нужен как часть Slurm-ограничений для выбранных узлов.

Практическая причина выбора: `normal/type_c/gpu:v100:1` стартовал сразу, но упал из-за несовместимости GLIBC старой ОС с текущим `.conda`; очередь `rocky` на V100 прогнозировала слишком поздний старт. Поэтому финальный полный запуск выполнен на Rocky-compatible `test/type_a/gpu:v100:1`.

Для повторного Slurm-запуска 27K EDA:

```bash
sbatch slurm/eda_27k.sh
```

Скрипт пишет `outputs/eda/27k_summary.json` и небольшие сводные CSV-таблицы. Sanity/log-файлы остаются локальными сгенерированными артефактами.

## Git-процесс

Работа ведется в ветке `eda`, которая отслеживает `origin/eda`.

Не коммитить:

- локальные Python-окружения (`.conda/`, `.venv/`);
- датасеты KuaiRand;
- произвольные CSV/parquet-экспорты, кроме компактных сводок `outputs/eda/27k_*`;
- сгенерированные outputs/logs/sanity files;
- checkpoints ноутбука.
