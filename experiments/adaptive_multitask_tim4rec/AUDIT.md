# Аудит адаптивной multitask-оптимизации

## Научная постановка

База проекта зафиксирована: `tim4rec_001`, `multitask_tim4rec_001`, `multitask_tim4rec_tuned_001`. Главный baseline для adaptive methods - tuned fixed-weight MultitaskTiM4Rec.

Наблюдение по завершённым экспериментам:

- Fixed multitask дал negative transfer относительно TiM4Rec: `NDCG@10 0.0581` против `0.0598`, `HR@10 0.1041` против `0.1053`.
- Optuna для статических weights сняла большую часть negative transfer: tuned fixed `NDCG@10 0.0598`, `HR@10 0.1071`.
- По `NDCG@10` tuned fixed фактически упёрся в TiM4Rec, но улучшил hit-rate.

Гипотеза нового этапа: статические weights недостаточны, потому что баланс между behavior tasks меняется во время обучения. Adaptive methods могут динамически управлять gradient magnitudes/directions, ослаблять negative transfer и сохранять benefit редких high-value behaviors, особенно `is_like`.

## Зафиксированная постановка

- Датасет: KuaiRand Protocol B multitask.
- Главная задача: next-item ranking, `L_rank`.
- Вспомогательные задачи: `is_click`, `long_view`, `is_like`, `is_profile_enter`.
- Backbone: тот же TiM4Rec, что в `tim4rec_001`.
- Behavior heads: четыре существующие `Linear(hidden_size, 1)`.
- Split, target definitions, recommendation objective и full-ranking evaluation не меняются.
- Test safety: smoke использует только Protocol B train batches, `test_evaluation_count=0`.

## Shared parameters

Task-specific heads исключаются из сравнения gradient conflicts и adaptive shared-gradient replacement:

- исключены: `click_head`, `long_view_head`, `like_head`, `profile_enter_head`;
- включены в full diagnostic: `item_embedding`, `in_layer_norm`, `layer_norm_time`, `ssd_layers`;
- включён в GradNorm norm loss: последний shared block `ssd_layers.1.*`, потому что original GradNorm рекомендует выбирать последний shared layer для снижения overhead.

Это различие важно: diagnostic cosine matrix отвечает на вопрос о конфликтах во всём shared TiM4Rec backbone, а GradNorm update weights вычисляются на компактном последнем shared block.

## Выводы по tuned fixed model

Optuna trial `110` выбрал:

| target | normalized task weight | effective pos_weight |
|---|---:|---:|
| `is_click` | 0.2556 | 1.0573 |
| `long_view` | 0.2364 | 1.2867 |
| `is_like` | 3.2470 | 9.0922 |
| `is_profile_enter` | 0.2610 | 7.5986 |

`is_like` был резко усилен относительно остальных задач. Возможные причины:

- target редкий: train positive rate около `1.86%`;
- effective `pos_weight` остаётся высоким даже после Optuna exponent;
- raw loss и gradient magnitude могут быть недостаточными без усиления;
- `is_like` может быть более ценным semantic behavior, чем click/long-view, но это нельзя объявлять новизной без отдельной проверки.

Smoke artifact измеряет raw loss, weighted loss, shared backbone gradient norm и cosine similarities, чтобы отделить rarity/pos_weight effect от gradient conflict/scale effect.

## Методы

| Метод | Статья | Идея | Главные гиперпараметры | Дополнительная цена backward | Подходит для ranking+aux? | Реализовано |
|---|---|---|---|---|---|---|
| GradNorm | Chen et al., 2018, "GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks" | Учить task weights так, чтобы weighted gradient norms соответствовали target norms из relative inverse training rates. | `alpha`, task-weight LR, выбранный shared layer `W` | 4 auxiliary `autograd.grad` + backward по весам + model backward на шаг | Да, если `L_rank` зафиксирован, а GradNorm балансирует только auxiliary weights | Да, auxiliary-only |
| PCGrad | Yu et al., 2020, "Gradient Surgery for Multi-Task Learning" | Если два task gradients конфликтуют, убрать компоненту одного градиента вдоль другого. | projection mode, task order/random seed | 5 task gradients + model backward на шаг | Сильный вариант для ranking-primary setup через ranking-anchored projection | Да, `ranking_anchored` и `all_tasks` |
| MetaBalance | He et al., WWW 2022, "MetaBalance: Improving Multi-Task Recommendations via Adapting Gradient Magnitudes of Auxiliary Tasks" | Масштабировать auxiliary gradients так, чтобы их moving-average magnitudes приближались к target-task gradient magnitude. | `relax_factor`, `beta`, optional decay schedule | 5 task gradients + model backward на шаг | Лучший recommender-specific fit: явно моделирует target plus auxiliary tasks | Да, MetaBalance-Fix |
| Uncertainty weighting | Kendall et al., 2018 | Учить параметры uncertainty, которые уменьшают вклад задач с высокой uncertainty. | learned `sigma_j` | примерно один model backward | Слабее как основной метод: задачи симметричны, gradient-conflict signal нет | Нет |
| DWA | Liu et al., 2019 | Увеличивать веса задач, loss которых снижается медленнее. | temperature `T`, averaging window | примерно один model backward | Простой baseline, но нужны несколько итераций/эпох для оценки rates | Нет |
| MGDA | Sener and Koltun, 2018 | Найти общее descent direction для multi-objective optimization. | solver/normalization choices | task gradients плюс малая optimization problem | Методологически чисто, но не ranking-primary | Нет |
| CAGrad | Liu et al., 2021 | Оптимизировать average loss, ограничивая конфликт с worst-case local objective. | conflict-aversion coefficient `c` | task gradients плюс optimization | Релевантно, но сложнее, чем нужно для первого sanity | Нет |
| Nash-MTL | Navon et al., 2022 | Рассматривать MTL как bargaining game и решать Nash bargaining update. | solver iterations, update frequency | высокая | Интересно, но слишком тяжело для первого KuaiRand sanity | Нет |
| IMTL | Liu et al., 2021 | Считать impartial task weights через баланс projections между task gradients. | normalization/linear solve choices | task gradients плюс linear solve | Хороший общий MTL baseline, но не target-aware | Нет |
| GradVac | Wang et al., 2020 | Использовать moving target cosine similarities, чтобы снижать вредную gradient interference. | beta, target cosine update | task gradients | Релевантно, но менее прямой первый шаг, чем PCGrad | Нет |
| GradDrop | Chen et al., 2020 | Отбрасывать gradient signs по sign agreement между задачами. | sign/drop rule details | task gradients/sign statistics | Полезный ориентир, но неудобен для parameter-level smoke в TiM4Rec | Нет |

## Решение по GradNorm

Original GradNorm задаёт `L(t)=sum_i w_i(t)L_i(t)`, gradient norm `G_W^(i)=||grad_W w_i(t)L_i(t)||_2`, loss ratio `L_i(t)/L_i(0)`, relative inverse training rate `r_i(t)`, target norm `G_bar(t) * r_i(t)^alpha` и `L_grad=sum_i |G_i-target_i|`.

В этом проекте GradNorm применяется только к четырём auxiliary losses:

```text
L_total = L_rank + lambda_aux * sum_i w_aux_i(t) * L_aux_i
```

Обоснование:

- ranking является научной целью и не должен отключаться adaptive balancing;
- tuned fixed weights уже показали, что auxiliary tasks могут помогать при контролируемом весе;
- GradNorm по всем 5 задачам мог бы снизить вес `L_rank`, если ranking учится быстрее/медленнее по критерию GradNorm; тогда recommendation objective стал бы вторичным.

Weights инициализируются из tuned fixed normalized weights и нормируются к `sum(w_aux)=4`.

## Решение по PCGrad

Рассмотрены два варианта:

- A: classic PCGrad по всем пяти task gradients;
- B: ranking-anchored PCGrad, где `g_rank` не меняется, а каждый auxiliary gradient проецируется только при конфликте с `g_rank`.

Smoke по умолчанию использует вариант B:

```text
if dot(g_aux, g_rank) < 0:
    g_aux <- g_aux - dot(g_aux, g_rank) / ||g_rank||^2 * g_rank
```

Это более защитимый вариант для ranking-primary recommender: auxiliary tasks могут регуляризовать representation, но не должны напрямую толкать shared backbone против ranking gradient.

## Решение по MetaBalance

MetaBalance - самый близкий recommender-specific метод. Официальный код: `facebookresearch/MetaBalance`, commit `1c342a99e09ec3b465d95f46e0e1b5c4d86deb94`, license `Attribution-NonCommercial 4.0 International`.

Реализованный smoke-вариант - MetaBalance-Fix:

- `L_rank` является target task;
- четыре behavior losses являются auxiliary tasks;
- для каждого shared parameter tensor обновляются moving averages target и auxiliary gradient magnitudes с `beta`;
- auxiliary gradients сдвигаются к target gradient magnitude через `relax_factor`;
- gradient direction не проецируется, что соответствует отличию статьи от PCGrad/GradSurgery.

Внешний код не добавлен в репозиторий.

## Результат cluster smoke

Run `adaptive_smoke_001` выполнен на cluster E как Slurm job `4275695`: partition `gpu-ef-quick`, node `cn-045`, GPU A100, `3` real train batches, runtime около `4m35s` по Slurm и `2026-08-25T08:21:39Z` -> `2026-08-25T08:25:04Z` внутри JSON. Python env: `torch 2.3.0+cu118`, `recbole 1.2.0`, `optuna 4.9.0`, `mamba-ssm 2.2.2`.

Test safety зафиксирован в артефакте:

- `test_dataset_loaded=false`;
- `test_dataloader_created=false`;
- `test_evaluated=false`;
- `test_evaluation_count=0`;
- validation-only benchmark rows: train `1086518`, validation `23951`, test `0`;
- forbidden test paths loaded: `[]`.

Диагностика tuned fixed на первом batch:

| task | raw loss | weighted loss | shared grad norm | cosine with rank |
|---|---:|---:|---:|---:|
| `rank` | 5.959172 | 5.959172 | 0.411454 | 1.000000 |
| `is_click` | 0.653172 | 0.022006 | 0.002114 | 0.001247 |
| `long_view` | 0.679393 | 0.021175 | 0.001902 | 0.001515 |
| `is_like` | 0.317194 | 0.135772 | 0.278751 | 0.085518 |
| `is_profile_enter` | 0.510580 | 0.017568 | 0.004159 | 0.005433 |

Pairwise cosine audit нашёл `3` negative pairs из `10`: `is_click` vs `is_like` (`-0.1023`), `is_click` vs `is_profile_enter` (`-0.0503`), `long_view` vs `is_like` (`-0.0433`). На первом smoke batch не было negative `rank` vs auxiliary pairs.

Итог smoke по adaptive methods:

| method | status | mean step sec | max allocated VRAM | autograd calls per step | finite gradients | key observation |
|---|---|---:|---:|---|---|---|
| fixed tuned | completed | 0.1030 | 1.82 GB | `backward=1`, `autograd.grad=0` | true | Static tuned baseline для сравнения overhead. |
| GradNorm auxiliary-only | completed | 2.4663 | 5.31 GB | `backward=2`, `autograd.grad=4` | true | Aux weights изменились с `{0.2556, 0.2364, 3.2470, 0.2610}` до `{0.2878, 0.2661, 3.1521, 0.2940}`. |
| PCGrad ranking-anchored | completed | 0.1972 | 2.24 GB | `backward=1`, `autograd.grad=5` | true | Rank-vs-aux conflict отсутствовал, поэтому projection event count был `0`; aux-aux conflicts остались `3`. |
| MetaBalance-Fix | completed | 0.2493 | 3.57 GB | `backward=1`, `autograd.grad=5` | true | Aux magnitudes были сдвинуты ближе к ranking magnitude; directions не проецировались, поэтому negative pairs стали `4` на первом batch. |

Интерпретация:

- `is_like` имеет самый большой tuned fixed weight (`3.2470`) и высокий effective `pos_weight` (`9.0922`). Его shared gradient norm (`0.2788`) намного ближе к ranking (`0.4115`), чем у других auxiliary tasks, а cosine with ranking положительный (`0.0855`), поэтому большой вес в этом batch не выглядит явно вредным.
- Real gradient conflicts есть, но в этом smoke они auxiliary-auxiliary, а не ranking-auxiliary. Поэтому ranking-anchored PCGrad - самый консервативный следующий метод: он вмешивается только когда auxiliary gradient направлен против `g_rank`.
- GradNorm научно полезен, но в этой реализации намного дороже, потому что требует second-order weight-gradient plumbing поверх auxiliary gradient norms.

## Результат 5-epoch validation sanity

Выполнены только два разрешённых sanity runs: `pcgrad_sanity_001` и `metabalance_sanity_001`. Оба использовали exact tuned fixed trial `110`, полный train split Protocol B и full-ranking validation; test split не загружался и не оценивался (`test_evaluation_count=0`).

Финальные jobs были выполнены на non-preemptive Slurm partition `test`, node `cn-050`, GPU `NVIDIA H200 NVL`: PCGrad job `4275801` (`8m39s`) и MetaBalance job `4275802` (`9m03s`). Первые попытки на `gpu-ef-quick/type_e` были вытеснены Slurm preemption на `cn-044`; `test/type_e` имел поздний старт, поэтому для завершённых sanity runs был использован явный submit override `--constraint=type_h`.

Reference rows ниже приведены только как validation ориентиры и не равны по бюджету 5-epoch sanity:

| method | run | budget | best epoch | HR@10 | HR@20 | HR@50 | NDCG@10 | NDCG@20 | NDCG@50 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| TiM4Rec reference | `tim4rec_001` | full reference | 12 | 0.1086 | 0.1740 | 0.3126 | 0.0593 | 0.0757 | 0.1030 |
| Tuned fixed multitask | `multitask_tim4rec_tuned_001` | full tuned reference | 16 | 0.1069 | 0.1769 | 0.3195 | 0.0589 | 0.0765 | 0.1046 |
| PCGrad ranking-anchored | `pcgrad_sanity_001` | 5 epochs | 5 | 0.1036 | 0.1656 | 0.3007 | 0.0568 | 0.0723 | 0.0990 |
| MetaBalance-Fix | `metabalance_sanity_001` | 5 epochs | 5 | 0.0951 | 0.1511 | 0.2665 | 0.0518 | 0.0659 | 0.0886 |

Gradient conflict summary:

| method | rank-aux fraction before | rank-aux fraction after | aux-aux fraction before | aux-aux fraction after | peak VRAM |
|---|---:|---:|---:|---:|---:|
| PCGrad ranking-anchored | 0.2778 | 0.1303 | 0.2441 | 0.2405 | 2.47 GB |
| MetaBalance-Fix | 0.2381 | 0.2189 | 0.2336 | 0.1712 | 5.27 GB |

Итог sanity:

- PCGrad за 5 эпох приблизился к tuned fixed validation reference, но ещё ниже full-budget baseline: `NDCG@10 0.0568` против `0.0589`.
- MetaBalance-Fix на этом коротком бюджете хуже PCGrad по ranking metrics: `NDCG@10 0.0518`.
- PCGrad ожидаемо сильнее снижает rank-aux conflicts, потому что явно проецирует auxiliary gradients против `g_rank`.
- MetaBalance сильнее снижает auxiliary-auxiliary conflict fraction после перескалирования magnitudes, но это не дало ranking выигрыша в 5-epoch sanity.

## Результат полного validation-only запуска PCGrad

Запуск `pcgrad_001` выполнен как полный validation-only эксперимент: максимум `300` эпох, early stopping по full-ranking validation `NDCG@10`, patience `10`. Использовался тот же tuned fixed trial `110` и тот же ranking-anchored PCGrad: `g_rank` является anchor, auxiliary gradients проецируются только против `g_rank`, auxiliary-auxiliary conflicts не обрабатываются.

Slurm job `4276024` завершился на partition `test`, constraint `type_h`, node `cn-049`, GPU `NVIDIA H200 NVL`. Время выполнения `23m31s`, фактически выполнено эпох `19`, лучшая эпоха `9`, причина остановки `early_stopping_no_validation_ndcg10_improvement_10`. Test split не загружался и не оценивался (`test_evaluation_count=0`).

| метод | запуск | бюджет | лучшая эпоха | HR@10 | HR@20 | HR@50 | NDCG@10 | NDCG@20 | NDCG@50 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| TiM4Rec reference | `tim4rec_001` | full reference | 12 | 0.1086 | 0.1740 | 0.3126 | 0.0593 | 0.0757 | 0.1030 |
| Fixed Multitask | `multitask_tim4rec_001` | full reference | 14 | 0.1061 | 0.1733 | 0.3115 | 0.0580 | 0.0749 | 0.1022 |
| Tuned fixed Multitask | `multitask_tim4rec_tuned_001` | full tuned reference | 16 | 0.1069 | 0.1769 | 0.3195 | 0.0589 | 0.0765 | 0.1046 |
| PCGrad ranking-anchored | `pcgrad_001` | validation-only full | 9 | 0.1082 | 0.1744 | 0.3089 | 0.0586 | 0.0752 | 0.1018 |

Решение относительно tuned fixed validation reference:

- absolute delta `NDCG@10`: `-0.0003`;
- relative delta `NDCG@10`: `-0.51%`;
- разница marginal/practically tied, но PCGrad не превзошёл tuned fixed;
- locked PCGrad test на этом основании открывать не следует.

Конфликты градиентов в diagnostic sample (`10` train batches на эпоху):

| эпоха | rank-aux до | rank-aux после | любой конфликт до | любой конфликт после | доля batch с проекцией |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.0750 | 0.0250 | 0.3000 | 0.1000 | 0.3000 |
| 3 | 0.3000 | 0.1250 | 0.7000 | 0.4000 | 0.7000 |
| 5 | 0.2250 | 0.0500 | 0.6000 | 0.2000 | 0.6000 |
| 9 | 0.2500 | 0.1250 | 0.8000 | 0.5000 | 0.8000 |
| 10 | 0.2250 | 0.1250 | 0.5000 | 0.4000 | 0.5000 |
| 19 | 0.3750 | 0.2000 | 0.9000 | 0.6000 | 0.9000 |

PCGrad действительно проецирует gradients в поздних эпохах: всего `10972` projection events по всем train batches, среднее `1.1127` events на train batch. При этом снижение conflict rate не дало validation improvement относительно tuned fixed baseline.

## Возможные собственные adaptive weighting идеи

1. Behavior-value-aware gradient weighting: стартовать из Optuna tuned weights, затем ограничивать или усиливать auxiliary gradients по rarity, behavior semantics и cosine with `g_rank`.
2. Ranking-anchored adaptive routing: оставлять `g_rank` неизменным и пропускать auxiliary updates через per-block gates по magnitude ratio и conflict rate.

Это только будущие гипотезы. Они не заявляются как новизна без дополнительной проверки литературы.

## Источники

- GradNorm: https://arxiv.org/abs/1711.02257
- PCGrad: https://arxiv.org/abs/2001.06782
- MetaBalance: https://arxiv.org/abs/2203.06801
- MetaBalance code: https://github.com/facebookresearch/MetaBalance
- MGDA: https://papers.nips.cc/paper_files/paper/2018/hash/432aca3a1e345e339f35a30c8f65edce-Abstract.html
- CAGrad: https://arxiv.org/abs/2110.14048
- GradDrop: https://arxiv.org/abs/2010.06808
