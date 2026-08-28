# Normalization Diagnostics

Этот файл фиксирует normalization policy до sanity-запусков.

## Что измеряется

На fixed train-only diagnostic sample измеряются:

- mean/std loss для `rank`, `is_click`, `long_view`, `is_like`, `is_profile_enter`;
- backbone gradient norm по каждому task;
- pairwise cosine между task gradients;
- positive rate по auxiliary targets;
- effective `pos_weight` из tuned Multitask TiM4Rec trial 110.

Diagnostic sample не использует validation и test.

## Как используется

- STCH получает normalized objectives: `loss / train_mean_loss`, затем использует log-normalized Smooth Tchebycheff.
- FAMO получает normalized objectives, чтобы adaptive weights не сводились к масштабу BCE/CE.
- EPO получает normalized objectives для LP и gradient diagnostics.
- GradHV считает dominated hypervolume в normalized loss space с deterministic train-only reference point.
- PHN, COSMOS и PaLoRA обучаются weighted normalized objective на той же preference grid.

## Текущий статус

Код измерения находится в `experiments/moo_8families/train.py` и пишет JSON в `runs/<run_id>.json`.
Локально на этой машине нет установленного `torch`/`recbole` и нет `/home/daryumin/...` data mount, поэтому реальные diagnostics должны выполняться на cHARISMa `type_e`.
