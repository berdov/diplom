# Эксперименты

Этот каталог фиксирует воспроизводимые эксперименты дипломного проекта.

Правила:

- одна существенная модель или идея живёт в отдельной Git branch;
- один конкретный training run получает стабильный `run_id`;
- каждый завершённый run добавляет строку в `experiments/results.csv`;
- конфиги, метрики, notes и компактные JSON/YAML сохраняются в Git;
- большие artifacts, модели, candidates, features, predictions и logs сохраняются на cHARISMa;
- test candidates фиксируются и не пересэмплируются между runs внутри одного candidate protocol;
- evaluation protocol нельзя менять между runs без нового protocol identifier.

Текущий candidate/evaluation protocol для первого baseline: `protocol_b_1pos_100neg_seed42`.

Подготовленные каталоги:

- `ltr_xgb_baseline/` - full-ranking XGBoost baseline и простые ранжирующие baselines.
- `tim4rec_baseline/` - аудит и smoke-подготовка TiM4Rec для KuaiRand Protocol B без полного обучения.
