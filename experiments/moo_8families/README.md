# MOO 8 Families Benchmark

Этот эксперимент - контролируемый validation-only benchmark представителей 8 terminal families из Figure 2 обзора
`Gradient-Based Multi-Objective Deep Learning: Algorithms, Theories, Applications, and Beyond` (`arXiv:2501.10945`).

Рабочая ветка: `exp/moo-8families-benchmark`.

## Инварианты

- Стартовая точка: `main` на commit `8baede3cc578103053f1d108a54c877fc8c4f69e`.
- Protocol B не меняется.
- Test split закрыт: `test_evaluated=false`, test dataloader не создается.
- Backbone для методов 1-5: тот же `MultitaskTiM4Rec`; меняется только multi-objective training rule.
- PLE (`exp/ple-tim4rec`) не используется и не merge-ится.
- Подбор гиперпараметров не проводится. Используются зафиксированные параметры tuned Multitask TiM4Rec trial 110.

## Представители семейств

| Семейство | Представитель | Тип решения |
|---|---|---|
| Loss Balancing | STCH | single solution |
| Gradient Weighting | FAMO | single solution |
| Gradient Manipulation | PCGrad | historical validation-only result |
| Finite set with preference vectors | EPO | finite preference set |
| Finite set without preference vectors | HV-Gradient / GradHV-style | finite Pareto set |
| Infinite set hypernetwork | PHN-adapter | continuous preference-conditioned adapter |
| Infinite set preference-conditioned net | COSMOS-style direct conditioned model | continuous preference-conditioned model |
| Infinite set model combination | PaLoRA | continuous preference-conditioned low-rank adapters |

PHN помечен как `PHN-adapter`, потому что полный PHN с генерацией всех параметров TiM4Rec, включая item embedding и SSD state-space блоки, не является честным drop-in для текущего RecBole full-sort пути. Реализация генерирует параметры компактного FiLM-adapter поверх общего TiM4Rec representation и явно логирует `representative_fidelity = family-level adaptation`.

`gradhv` помечен как `HV-Gradient / GradHV-style`: код оптимизирует exact dominated hypervolume через autograd по inclusion-exclusion objective, но не является точной репродукцией HIGA-MO/GradHV algorithm.

## Preference Sampling

Для PHN-adapter, COSMOS-style и PaLoRA обучение использует continuous Dirichlet sampling на simplex, а не циклический перебор fixed grid:

- PHN-adapter: `Dirichlet(alpha=0.2)`, как default alpha в official PHN trainer.
- COSMOS-style: `Dirichlet(alpha=1.2)`, как default в official COSMOS examples.
- PaLoRA: `Dirichlet(alpha=1.0)`, как official PaLoRA Dirichlet ray sampler default.

`preferences.yaml` остается frozen grid только для validation, Pareto plots и reproducible operating-point evaluation. Validation results не используются для изменения training distribution.

## Evaluation Protocol

Validation Pareto hypervolume для Families 4-8 использует один frozen reference point из `config.yaml`:

```text
[1 - NDCG@10, click_BCE, long_view_BCE, like_BCE, profile_BCE] <= [1.0, 2.0, 2.0, 2.0, 2.0]
```

Reference зафиксирован до MOO sanity results: ranking coordinate остается analytic upper bound `1.0` для `1-NDCG@10`, а BCE coordinates получают conservative fixed bound `2.0`, потому что BCE mathematically unbounded above. Проверка существующих historical/control validation records перед запуском MOO sanity нашла maximum BCE `0.6414133221375151` и не нашла BCE выше `2.0`, поэтому `2.0` выбран как достаточно плохой frozen bound. Это evaluation metric-space reference; он не смешивается с train-only normalized-loss reference для HV-Gradient training, не строится по новым MOO outputs и одинаков для Families 4-8. Если validation point хуже reference по любой координате, run падает вместо silent clipping.

Primary ranking result берется не как oracle over all preferences:

- EPO, PHN-adapter, COSMOS-style, PaLoRA: predefined `rank_heavy`.
- STCH, FAMO, PCGrad: единственная solution.
- HV-Gradient / GradHV-style: best validation NDCG@10 among preference-free finite solutions, явно `selection_is_validation_oracle = true`.

`oracle_best_validation_point` сохраняется отдельно от `ranking_operating_point`.

Smoke для PHN-adapter, COSMOS-style и PaLoRA дополнительно проверяет real-model preference sensitivity на train batch: `rank_heavy` vs `like_heavy`. Если trained pathway не меняет ranking score выше tolerance, smoke завершается ошибкой.

## Запуски

Локально без RecBole/Torch можно проверить только синтаксис. На кластере E:

```bash
sbatch --export=ALL,MOO_METHOD=stch,MOO_STAGE=smoke,MOO_RUN_ID=stch_smoke_001 slurm/moo_8families.sh
sbatch --export=ALL,MOO_METHOD=famo,MOO_STAGE=smoke,MOO_RUN_ID=famo_smoke_001 slurm/moo_8families.sh
sbatch --export=ALL,MOO_METHOD=epo,MOO_STAGE=smoke,MOO_RUN_ID=epo_smoke_001 slurm/moo_8families.sh
sbatch --export=ALL,MOO_METHOD=gradhv,MOO_STAGE=smoke,MOO_RUN_ID=gradhv_smoke_001 slurm/moo_8families.sh
sbatch --export=ALL,MOO_METHOD=phn,MOO_STAGE=smoke,MOO_RUN_ID=phn_smoke_001 slurm/moo_8families.sh
sbatch --export=ALL,MOO_METHOD=cosmos,MOO_STAGE=smoke,MOO_RUN_ID=cosmos_smoke_001 slurm/moo_8families.sh
sbatch --export=ALL,MOO_METHOD=palora,MOO_STAGE=smoke,MOO_RUN_ID=palora_smoke_001 slurm/moo_8families.sh
```

После успешных 7 smoke-запусков:

```bash
python -m experiments.moo_8families.run_benchmark --stage sanity --submit
```

PCGrad не перезапускается автоматически: используется существующий `pcgrad_001`, если его validation-only guards проходят.

## Артефакты

- `config.yaml` - frozen protocol, tuned params, method defaults.
- `preferences.yaml` - зафиксированная preference grid для validation/Pareto plots/reproducible operating points.
- `NORMALIZATION.md` - план и результаты train-only normalization diagnostics.
- `runs/*.json` - machine-readable run artifacts.
- `runs/*_notes.md` - краткие notes по каждому run.
- `BENCHMARK_REPORT.md` - итоговая таблица после `build_results.py`.
