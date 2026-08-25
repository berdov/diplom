# Behavior-MoE TiM4Rec audit

## Motivation

Цель этапа - проверить, может ли compact behavior-specialized MoE быть технически корректной надстройкой над tuned MultitaskTiM4Rec без изменения Protocol B, TiM4Rec backbone и ranking objective.

Рабочая гипотеза: разные типы поведения KuaiRand требуют частично специализированных latent transformations. Behavior-MoE может уменьшать interference не через gradient projection, а через representational specialization после shared sequence representation.

## Relation to previous experiments

TiM4Rec reproduction зафиксировал основной full-ranking baseline на Protocol B.

Fixed MultitaskTiM4Rec добавил четыре behavior heads (`is_click`, `long_view`, `is_like`, `is_profile_enter`) поверх того же shared representation, но дал небольшой negative transfer относительно TiM4Rec.

Tuned fixed multitask через `multitask_tim4rec_optuna_v1` почти устранил этот negative transfer: tuned fixed validation `NDCG@10=0.0589`, locked test `NDCG@10=0.0598`.

PCGrad validation-only run `pcgrad_001` уменьшил rank-aux gradient conflicts, но не превзошёл tuned fixed validation reference: `NDCG@10=0.0586` против `0.0589`, delta `-0.0003` (`-0.51%`). Поэтому locked PCGrad test не открывался.

## Related work

Близкие работы, которые нужно учитывать при формулировке novelty:

- HM2Rec: heterogeneous behavior / MoE-style recommendation для multi-behavior setting.
- HM4SR: hierarchical mixture-of-experts для sequential recommendation.
- TriSSR: sequential recommendation с multi-interest / multi-expert specialization.
- Generic MoE sequential recommenders, например HyMoERec и FAME.

Мы не заявляем новизну как "Mamba + MoE". Такая формулировка слишком широкая: MoE уже используется в sequential recommendation, а Mamba-like backbones уже комбинируются с разными routing/mixture mechanisms.

Рабочая идея уже: behavior-specialized experts, где интерпретация experts привязана к типам пользовательского поведения KuaiRand:

- `interest`: click/exposure response;
- `consumption`: long-view/watch behavior;
- `positive`: explicit positive engagement;
- `shared`: residual/general representation.

Отличия от generic MoE: routing анализируется по behavior tasks, а не только как универсальное увеличение capacity.

Отличия от modality experts: experts не соответствуют отдельным modalities; вход остаётся тем же item+time sequence representation.

Отличия от temporal experts: experts не делят события по временным режимам; time-aware representation приходит из TiM4Rec backbone.

Отличия от generic auxiliary multitask: auxiliary heads получают разные task-conditioned MoE representations, а не один общий `h`.

## Proposed minimal architecture

Схема:

```text
history + time
  -> TiM4Rec backbone
  -> shared representation h
  -> Behavior MoE
  -> task-specific representations
  -> ranking + behavior heads
```

Backbone не заменяется на MoE. MoE - компактная residual-надстройка после shared representation.

## Experts

Первая версия использует 4 одинаковых experts:

| expert | initial interpretation |
|---|---|
| `interest` | `is_click` |
| `consumption` | `long_view` |
| `positive` | `is_like + is_profile_enter` |
| `shared` | residual/general |

Каждый expert: `Linear(hidden, hidden) -> GELU -> Dropout -> Linear(hidden, hidden)`. Hidden size остаётся `64`. Experts не мощнее backbone.

## Router

Router learnable и получает только shared representation `h`; current behavior labels не используются как router input.

Для каждого task свой router head `Linear(hidden, 4)`. Soft routing:

```text
p(task, expert | h) = softmax(router_task(h) / temperature)
```

Initial logits почти равномерны. Для auxiliary tasks добавлен слабый semantic bias `0.05` к соответствующему expert, чтобы старт был интерпретируемым, но не жёстким.

## Task-Conditioned Routing

Отдельные routing distributions считаются для:

- `ranking`;
- `click`;
- `long_view`;
- `like`;
- `profile`.

Ranking route имеет доступ ко всем experts и не фиксируется на одном expert.

## Loss

Smoke использует locked tuned fixed loss configuration:

- source: `multitask_tim4rec_optuna_v1`, trial `110`;
- `lambda_aux=0.13182740780834337`;
- normalized task weights из `best_params.yaml`;
- effective pos weights из `best_params.yaml`;
- `learning_rate=0.0019075370668084298`;
- `weight_decay=1.925502656735127e-06`;
- `head_lr_multiplier=0.45851058258621097`.

Loss и MoE одновременно не тюнятся.

## Collapse Prevention

Load-balancing loss подготовлен, но в первом smoke выключен:

```text
mean((mean_task_batch_p_expert - 1 / num_experts) ** 2)
```

Причина: сначала нужно измерить routing distribution без дополнительной регуляризации. Если smoke покажет collapse, следующий sanity должен включить минимальный load balancing.

## Smoke Diagnostics

Smoke должен проверить:

- forward для ranking и 4 behavior logits;
- finite total/rank/aux losses;
- backward;
- optimizer step;
- finite gradients and parameters;
- gradients для всех experts, router и behavior heads;
- mean expert probabilities по task;
- entropy и normalized entropy;
- max/min expert share;
- pairwise distances между task routings;
- parameter/cost overhead против tuned fixed MultitaskTiM4Rec;
- `test_evaluation_count=0`.

Финальный smoke `behavior_moe_smoke_001` выполнен на Slurm job `4276396`: partition `test`, constraint `type_h`, node `cn-050`, GPU `NVIDIA H200 NVL`. Batch step `4276396.batch` завершился `COMPLETED`, exit `0:0`, elapsed `00:05:41`, MaxRSS `2885488K`. Выполнено `5` optimization steps на real train batches, batch size `2048`. Full epochs не запускались (`epochs_run=0`), full validation не запускался, test dataset не загружался и не оценивался (`test_evaluation_count=0`).

Forward/backward checks:

- shared representation, MoE representations, ranking scores и 4 auxiliary logits finite;
- total/rank/aux losses finite;
- gradients finite;
- optimizer step выполнен;
- experts, router и all behavior heads получили gradients и обновились.

Routing после smoke:

| task | interest | consumption | positive | shared |
|---|---:|---:|---:|---:|
| ranking | 0.2270 | 0.2568 | 0.2783 | 0.2379 |
| click | 0.2682 | 0.2394 | 0.2454 | 0.2470 |
| long_view | 0.2702 | 0.2593 | 0.2337 | 0.2369 |
| like | 0.2272 | 0.2678 | 0.2521 | 0.2530 |
| profile | 0.2296 | 0.2505 | 0.2582 | 0.2617 |

Expert utilization across tasks:

| expert | mean utilization |
|---|---:|
| interest | 0.2445 |
| consumption | 0.2547 |
| positive | 0.2535 |
| shared | 0.2473 |

Router entropy remains high, as expected for smoke:

| task | normalized entropy | max expert share | min expert share |
|---|---:|---:|---:|
| ranking | 0.9964 | 0.2783 | 0.2270 |
| click | 0.9976 | 0.2682 | 0.2394 |
| long_view | 0.9975 | 0.2702 | 0.2337 |
| like | 0.9974 | 0.2678 | 0.2272 |
| profile | 0.9971 | 0.2617 | 0.2296 |

Collapse checks:

- expert collapse: `false`;
- shared expert domination: `false`;
- all-task same routing: `false`;
- minimum experts used per task: `4`.

Strongest task-routing difference: `ranking` vs `click`, L1 distance `0.1007`, cosine distance `0.0063`. Это слабый initial behavior-specialization signal, а не quality evidence: router ещё почти равномерный, но task-conditioned heads уже дают разные routing distributions.

## Parameter/Cost Overhead

Ожидаемый overhead умеренный: 4 маленьких MLP experts и 5 router heads поверх `hidden_size=64`. Checkpoint сохранять не нужно; в Git попадают только compact smoke JSON/notes.

Фактический parameter overhead:

| model | params |
|---|---:|
| TiM4Rec | 593498 |
| Tuned MultitaskTiM4Rec | 593758 |
| BehaviorMoETiM4Rec | 628338 |

Delta против tuned fixed multitask: `34580` parameters, relative increase `5.82%`.

Cost smoke на H200:

| measurement | tuned fixed | Behavior-MoE | ratio |
|---|---:|---:|---:|
| raw mean step, sec | 0.1244 | 0.0387 | 0.3108 |
| mean step excluding first measured fixed step, sec | 0.0259 | 0.0357 | 1.3783 |

Raw timing из JSON нельзя читать как ускорение MoE: первый measured tuned fixed step содержит остаточный CUDA/cache overhead (`0.5185s`), тогда как следующие fixed steps около `0.026s`. Для оценки архитектурного overhead на smoke разумнее смотреть trimmed ratio `1.38x`. Peak allocated VRAM: tuned fixed `1877095936` bytes, Behavior-MoE `1898208256` bytes.

## Risks

- router collapse на один expert;
- одинаковый routing для всех tasks;
- доминирование `shared` expert;
- слишком большой model overhead;
- отсутствие gradients у части experts/router/heads;
- нестабильные routing logits;
- MoE может затопить ranking representation, если residual слишком большой.

## Next Sanity Experiment

Если smoke подтверждает корректный pipeline, отсутствие collapse и хотя бы слабый task-specific routing signal, следующий запуск должен быть 5-epoch validation sanity для plain Behavior-MoE. Если collapse проявится уже на smoke, следующий запуск должен быть Behavior-MoE + minimal load balancing.

Итог `behavior_moe_smoke_001`: pipeline корректен, router не collapsed, task-specific routing signal есть, overhead умеренный. Следующий sanity run лучше запускать как plain Behavior-MoE без load balancing; load balancing оставить как fallback, если collapse появится в 5-epoch sanity.

## 5-Epoch Sanity Result

`behavior_moe_sanity_001` выполнен как plain Behavior-MoE без load balancing, Optuna, hard routing, adaptive loss и test access. Запуск: Slurm job `4276720`, partition `test`, constraint `type_e`, node `cn-045`, GPU `NVIDIA A100-SXM4-80GB`, elapsed `00:09:35`, batch MaxRSS `3050872K`.

Validation trajectory:

| epoch | HR@10 | NDCG@10 | NDCG@20 | NDCG@50 |
|---:|---:|---:|---:|---:|
| 1 | 0.0867 | 0.0482 | 0.0603 | 0.0815 |
| 2 | 0.0954 | 0.0525 | 0.0672 | 0.0895 |
| 3 | 0.0987 | 0.0546 | 0.0700 | 0.0949 |
| 4 | 0.1027 | 0.0562 | 0.0717 | 0.0976 |
| 5 | 0.0992 | 0.0546 | 0.0713 | 0.0974 |

Best validation `NDCG@10=0.0562` на epoch 4. Epoch 5 `NDCG@10=0.0546`, что ниже `multitask_tim4rec_sanity_001` epoch 5 (`0.0557`) на `-0.0011`, но выше epoch 1 и не является collapse/fatal degradation.

Routing diagnostics:

- expert collapse: `false`;
- dead expert: `false`;
- shared expert domination: `false`;
- все experts и routers получают gradients на diagnostic epochs `1`, `3`, `5`;
- normalized entropy снижается для ranking с `0.5281` до `0.3095`, для click с `0.2316` до `0.3822`, для long_view остаётся около `0.54`, для like около `0.42-0.45`, для profile около `0.49-0.60`;
- mean required-pair L1 specialization не усиливается: `0.9368 -> 0.6816`.

Итоговое routing на epoch 5:

| task | interest | consumption | positive | shared |
|---|---:|---:|---:|---:|
| ranking | 0.0154 | 0.1610 | 0.8065 | 0.0172 |
| click | 0.1291 | 0.0979 | 0.7344 | 0.0386 |
| long_view | 0.1021 | 0.2073 | 0.6061 | 0.0845 |
| like | 0.0353 | 0.7166 | 0.0294 | 0.2187 |
| profile | 0.0288 | 0.3257 | 0.4453 | 0.2002 |

Semantic labels mostly do not align with learned routes by epoch 5: click, long_view and like do not choose their named expert as top route; profile partially aligns with `positive`. This does not prove architecture failure, but it means expert names should still be treated as interpretive labels, not supervised semantics.

Decision after sanity: pipeline is technically valid, but full plain Behavior-MoE should not be launched immediately. The next step should be analysis of routing architecture or initialization before a full run; load balancing is not needed now because there is no collapse/dead expert/shared domination.

## Источники related work

- HM2Rec: https://ojs.aaai.org/index.php/AAAI/article/view/34441
- HM4SR: https://arxiv.org/abs/2505.13036
- TriSSR: https://www.sciencedirect.com/science/article/abs/pii/S0020025524003984
- HyMoERec: https://arxiv.org/abs/2505.21024
- FAME: https://arxiv.org/abs/2504.10230
