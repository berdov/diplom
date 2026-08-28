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
| Finite set without preference vectors | GradHV | finite Pareto set |
| Infinite set hypernetwork | PHN adapter | continuous preference-conditioned adapter |
| Infinite set preference-conditioned net | COSMOS-style direct conditioned model | continuous preference-conditioned model |
| Infinite set model combination | PaLoRA | continuous preference-conditioned low-rank adapters |

PHN помечен как `PHN adapter`, потому что полный PHN с генерацией всех параметров TiM4Rec, включая item embedding и SSD state-space блоки, не является честным drop-in для текущего RecBole full-sort пути. Реализация генерирует параметры компактного FiLM-adapter поверх общего TiM4Rec representation и явно логирует этот статус.

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
- `preferences.yaml` - зафиксированная preference grid до просмотра новых validation results.
- `NORMALIZATION.md` - план и результаты train-only normalization diagnostics.
- `runs/*.json` - machine-readable run artifacts.
- `runs/*_notes.md` - краткие notes по каждому run.
- `BENCHMARK_REPORT.md` - итоговая таблица после `build_results.py`.
