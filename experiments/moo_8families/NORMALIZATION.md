# Normalization Diagnostics

Этот файл фиксирует normalization policy до sanity-запусков.

## Policy

All families are evaluated under a common train-only objective normalization protocol unless the original method mathematically requires otherwise. Это нужно потому, что ranking CE, auxiliary BCE и scaled auxiliary contributions имеют разные естественные масштабы; без общей шкалы scale-sensitive методы могут сравнивать не trade-off, а единицы измерения loss.

## Что измеряется

На fixed train-only diagnostic sample измеряются:

- mean/std loss для `rank`, `is_click`, `long_view`, `is_like`, `is_profile_enter`;
- backbone gradient norm по каждому task;
- pairwise cosine между task gradients;
- positive rate по auxiliary targets;
- effective `pos_weight` из tuned Multitask TiM4Rec trial 110.

Diagnostic sample не использует validation и test.

Raw mean losses и выбранные loss scales сохраняются в каждом `runs/<run_id>.json`:

- `normalization.mean_loss` - raw train-only mean loss per objective;
- `normalization.std_loss` - raw train-only std per objective;
- `normalization.loss_scales` - делители для normalized objective vector;
- `training.epochs[*].losses.raw_<task>` - epoch-mean raw objective;
- `training.epochs[*].losses.normalized_<task>` - epoch-mean normalized objective.

## Как используется

- STCH получает normalized objectives: `loss / train_mean_loss`, затем использует log-normalized Smooth Tchebycheff.
- FAMO получает normalized objectives, чтобы adaptive weights не сводились к масштабу BCE/CE.
- EPO получает normalized objectives для LP и gradient diagnostics.
- HV-Gradient / GradHV-style считает dominated hypervolume в normalized loss space с deterministic train-only reference point.
- PHN-adapter, COSMOS-style и PaLoRA обучаются weighted normalized objective на continuous Dirichlet samples; fixed `preferences.yaml` grid используется только для validation operating points.

Scale-sensitive методы в этом benchmark: STCH, FAMO, EPO, HV-Gradient / GradHV-style, PHN-adapter, COSMOS-style и PaLoRA. PCGrad берется как historical validation-only result и дополнительно проверяется только на закрытый test protocol.

## Текущий статус

Код измерения находится в `experiments/moo_8families/train.py` и пишет JSON в `runs/<run_id>.json`.
Локально на этой машине нет установленного `torch`/`recbole` и нет `/home/daryumin/...` data mount, поэтому реальные diagnostics должны выполняться на cHARISMa `type_e`.

Validation и test не используются для вычисления `mean_loss`, `loss_scales` или hypervolume reference point. Benchmark оценивает methods under normalized-objective protocol; raw validation metrics и raw auxiliary BCE остаются отдельными reported outcomes.
