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

- HM2Rec: MoE-Mamba sequential recommender, где FFN заменяется MoE-модулем.
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

## Почему generic Behavior-MoE не специализировался

### Observed facts

- В `behavior_moe_sanity_001` большинство задач к epoch 5 стало использовать один `positive` expert: `ranking=0.8065`, `click=0.7344`, `long_view=0.6061`, `profile=0.4453`.
- Expert labels не совпали с фактической specialization: `click` не выбрал `interest`, `long_view` не выбрал `consumption`, `like` ушёл в `consumption`, а не в `positive`.
- На старте smoke routing был почти равномерным и высокоэнтропийным, но это не предотвратило последующее схождение нескольких задач к одному expert.
- Формального collapse нет: dead expert и shared domination не обнаружены, gradients приходят во все experts/router, но semantic routing не возник.
- Средний required-pair L1 distance снизился с `0.9368` до `0.6816`, то есть выбранные task routings стали ближе, а не дальше.

### Hypotheses

- Свободный router имеет мало причин сохранять заданную behavior semantics: labels experts являются только именами, а не constraints или supervision.
- Одинаковые expert architectures симметричны; слабый initial semantic bias `0.05` не обязан пережить оптимизацию.
- Ranking loss доминирует по масштабу и может делать один expert универсально полезным для нескольких задач.
- Auxiliary tasks используют общий shared representation `h` и похожие MLP experts, поэтому optimization может предпочитать reuse одного expert вместо чистой поведенческой декомпозиции.
- Load balancing мог бы сделать usage более ровным, но это не гарантировало бы, что `interest/consumption/positive/shared` получат нужную семантику.

Эти пункты являются интерпретацией наблюдений, а не доказанной причиной failure mode.

## Почему load balancing не основной фикс

Load-balancing loss выравнивает среднюю загрузку experts и полезен против collapse/dead experts. В `behavior_moe_sanity_001` такой проблемы нет: все experts используются и получают gradients.

Для текущей ошибки важнее semantic mismatch: router может равномерно использовать experts, но всё равно не соответствовать behavior groups. Поэтому следующий probe проверяет structural inductive bias через constrained soft routing, а load-balancing остаётся выключенным. Если он понадобится, это должна быть отдельная ablation.

## Связь с MMoE и PLE

Literature check выполнен 2026-08-25 по MMoE, PLE/CGC, structured/expert masking и sequential/multi-behavior MoE recommenders.

Generic `BehaviorMoETiM4Rec` является близким вариантом MMoE: несколько shared experts доступны всем tasks, а каждый task имеет свой gate/router. Отличие в том, что MoE стоит как residual-надстройка после sequential/time-aware TiM4Rec representation, а не как основной bottom-layer recommender tower.

Structured `BehaviorMoE` ближе к PLE/CGC, чем к чистому MMoE: auxiliary task смешивает свой specialist expert и shared expert, а ranking gate видит все specialists + shared. Это не полная PLE: нет multi-level progressive extraction, нет отдельных private expert pools на каждый task, и behavior groups задаются вручную (`click`, `long_view`, `like/profile`) поверх sequential representation. Но общий принцип shared-specific experts уже известен, поэтому такую схему нельзя заявлять как самостоятельную архитектурную новизну.

В просмотренных работах по sequential recommendation есть MoE с shared/specialized branches и behavior-aware/multi-behavior MoE, но этот probe не должен формулироваться как "новый MoE". Для реальной научной новизны нужен дополнительный вклад: например, time-aware behavior grouping, learnable behavior-to-expert constraints, устойчивый anti-degradation criterion, или честный baseline против PLE/CGC в том же Protocol B.

## Structured Behavior-MoE probe design

`StructuredBehaviorMoE` оставляет четыре experts одинаковой мощности:

| expert | initial interpretation |
|---|---|
| `interest` | `is_click` |
| `consumption` | `long_view` |
| `engagement` | `is_like + is_profile_enter` |
| `shared` | residual/general |

Allowed expert masks:

| task | allowed experts |
|---|---|
| `ranking` | `interest`, `consumption`, `engagement`, `shared` |
| `click` | `interest`, `shared` |
| `long_view` | `consumption`, `shared` |
| `like` | `engagement`, `shared` |
| `profile` | `engagement`, `shared` |

Formula:

```text
logits_allowed = mask(router_task(h), allowed_experts)
p_task = softmax(logits_allowed / temperature)
h_task = h + residual_scale * sum_e p_task[e] * E_e(h)
```

Запрещённые experts маскируются до softmax и должны иметь exact-zero weights в expanded 4-expert diagnostics. Current behavior labels не используются как router input; task context задаётся отдельным router head per task. Loss, LR, weight decay, dropout, head LR multiplier, task weights и pos-weight policy остаются из `multitask_tim4rec_optuna_v1` trial `110`.

## Structured Behavior-MoE smoke result

`structured_behavior_moe_smoke_001` выполнен как architecture probe: 5 real train batches, no epochs, no full validation, no test. Засчитанный Slurm job `4278026`: partition `test`, constraint `type_e`, node `cn-044`, GPU `NVIDIA A100-SXM4-80GB`, elapsed `00:05:32`, batch MaxRSS `3014080K`.

Expanded routing после smoke:

| task | interest | consumption | engagement | shared |
|---|---:|---:|---:|---:|
| ranking | 0.2282 | 0.2470 | 0.2784 | 0.2465 |
| click | 0.5070 | 0.0000 | 0.0000 | 0.4930 |
| long_view | 0.0000 | 0.5362 | 0.0000 | 0.4638 |
| like | 0.0000 | 0.0000 | 0.4639 | 0.5361 |
| profile | 0.0000 | 0.0000 | 0.4873 | 0.5127 |

Forbidden paths имеют exact-zero weights для всех auxiliary tasks. Local weights суммируются к 1 с max deviation не выше `1.19e-7`.

Specialist shares:

| metric | value |
|---|---:|
| `click_interest_share` | 0.5070 |
| `long_consumption_share` | 0.5362 |
| `like_engagement_share` | 0.4639 |
| `profile_engagement_share` | 0.4873 |
| `average_specialist_share` | 0.4986 |
| `uniform_specialist_share_baseline` | 0.5000 |
| `specialization_above_uniform` | -0.0014 |

Shared domination отсутствует: shared shares `click=0.4930`, `long_view=0.4638`, `like=0.5361`, `profile=0.5127`, none `>0.9`.

Gradient connectivity соответствует structural design:

- `interest <- ranking + click`;
- `consumption <- ranking + long_view`;
- `engagement <- ranking + like + profile`;
- `shared <- ranking + click + long_view + like + profile`;
- each task router receives gradients only from its own objective.

Parameter count совпадает с generic Behavior-MoE:

| model | params |
|---|---:|
| TiM4Rec | 593498 |
| Tuned MultitaskTiM4Rec | 593758 |
| Generic Behavior-MoE | 628338 |
| Structured Behavior-MoE | 628338 |

Compute smoke cost on A100 for structured: raw mean step `0.1285s`, trimmed mean `0.0707s`, peak allocated VRAM `1847309312` bytes. Historical generic smoke was on H200, so its raw `0.0387s` / trimmed `0.0359s` step time is not hardware-comparable. Structured routing is not actually cheaper in this implementation because all four expert outputs are still computed once and ranking uses all experts.

Decision: structured routing is technically correct, but not yet ready for 5-epoch sanity. The reason is not a pipeline failure: forward/backward/optimizer, gradients, updates, masks and test safety are correct. The issue is diagnostic: `average_specialist_share=0.4986` is slightly below the structural uniform baseline `0.5`, so this smoke does not yet show a specialist preference beyond the mask itself. Before a 5-epoch sanity, the better next step is to compare against a proper PLE-style baseline or add a small structured-initialization/regularization ablation, still without claiming novelty from the PLE/CGC-like masking alone.

## PLE baseline

Аудит выполнен по первичной статье Tang et al., RecSys 2020, DOI `10.1145/3383313.3412236`, и MMoE Ma et al., KDD 2018, DOI `10.1145/3219819.3220007`. Авторский официальный PLE repository в доступных источниках не найден; публичные реализации вроде DeepCTR используются только как sanity reference для общей схемы, не как первоисточник.

### Что фиксирует первоисточник PLE/CGC

Shared-bottom / hard parameter sharing даёт всем задачам один общий нижний слой и task towers сверху. Это уменьшает число параметров, но может усиливать negative transfer, потому что разные objectives вынуждены использовать одну и ту же shared representation.

MMoE заменяет single shared-bottom на набор shared experts и отдельный gate для каждой task. Все experts в MMoE общие: каждая task может выбрать любую смесь experts. Это уже sample-dependent routing, но без явного понятия task-specific expert.

CGC вводит явное разделение:

- shared experts отвечают за task-common patterns;
- task-specific experts доступны только своей task;
- gate каждой task выбирает смесь из `own task-specific experts + shared experts`;
- параметры shared experts получают gradients от всех tasks;
- параметры task-specific experts получают gradients только от своей task.

PLE обобщает CGC на несколько extraction networks. На промежуточных уровнях кроме task gates появляется shared gate: shared module смешивает все shared и task-specific experts текущего уровня и передаёт общий сигнал на следующий extraction level. В верхних уровнях separation становится сильнее, поэтому authors называют это progressive separation routing.

### Чем это отличается от StructuredBehaviorMoE

`StructuredBehaviorMoE` был diagnostic probe с четырьмя вручную заданными semantic experts (`interest`, `consumption`, `engagement`, `shared`) и hard masks по behavior groups. В нём `ranking` видит все experts, а некоторые behavior tasks делят один grouped specialist, например `is_like` и `is_profile_enter` оба используют `engagement`.

PLE/CGC baseline устроен иначе:

- у каждой task есть свой private task-specific expert, включая `ranking`;
- behavior tasks не делят один grouped specialist;
- каждая task gate выбирает только из своего private expert и shared experts;
- нет handcrafted behavior masks поверх общего 4-expert пространства;
- нет residual `h + adapter`, потому что формула CGC/PLE задаёт task representation как weighted sum выбранных expert outputs;
- one-level вариант не использует shared gate, потому что shared output нужен для следующего extraction level, которого в CGC-1level нет.

### Реализованный baseline

Реализация называется `PLETiM4Rec`, но методологически это минимальный one-level CGC / PLE-style baseline поверх tuned TiM4Rec representation:

```text
history + time
  -> TiM4Rec backbone
  -> shared representation h
  -> experts:
       ranking_specific
       click_specific
       long_view_specific
       like_specific
       profile_specific
       shared_0
       shared_1
  -> task gate k over [k_specific, shared_0, shared_1]
  -> task representation g_k(h)
  -> ranking score / behavior heads
```

Gate input - только shared TiM4Rec representation `h`; current labels не используются. Loss, Protocol B, targets, task weights, pos-weight policy, learning rate, weight decay, dropout и head LR multiplier сохранены из tuned fixed MultitaskTiM4Rec trial `110`.

Capacity control: прямой вариант `5 specific + 2 shared` с `hidden_size=64` дал бы существенно больше параметров, чем Behavior-MoE. Поэтому expert MLP оставлен того же типа, но с bottleneck `expert_hidden_size=37`:

```text
Linear(64, 37) -> GELU -> Dropout -> Linear(37, 64)
```

Так PLE overhead остаётся того же порядка, что generic/structured Behavior-MoE: 7 experts и 5 gates почти parameter-matched с 4 full-size experts и 5 gates у Behavior-MoE. Это capacity-control decision, а не новая архитектурная идея.

### Диагностика PLE

Для PLE старый `average_specialist_share` из `StructuredBehaviorMoE` напрямую не используется. Основной diagnostic:

```text
specific_share(task) = p(task_specific_expert | h)
shared_total_share(task) = p(shared_0 | h) + p(shared_1 | h)
mean_specific_share = mean_task specific_share(task)
```

При `1 specific + 2 shared` uniform baseline для каждой task равен `1/3`. Сравнение specific share идёт с этим baseline. Дополнительно фиксируются entropy, gradients, updates, collapse на shared-only и own-only режимы, parameter count, step time, peak VRAM и test safety.

PLE baseline не является нашей новизной. Его задача - честно проверить, объясняет ли классический shared/specific gating эффект, который мы хотели получить от behavior-specialized MoE.

## Источники related work

- HM2Rec: https://ojs.aaai.org/index.php/AAAI/article/view/38567
- HM4SR: https://arxiv.org/abs/2501.14269
- TriSSR: https://www.sciencedirect.com/science/article/abs/pii/S0925231226009707
- HyMoERec: https://arxiv.org/abs/2511.06388
- FAME: https://arxiv.org/abs/2411.01457
- MMoE: https://www.kdd.org/kdd2018/accepted-papers/view/modeling-task-relationships-in-multi-task-learning-with-multi-gate-mixture-
- PLE/CGC: https://dl.acm.org/doi/10.1145/3383313.3412236
- DeepCTR PLE implementation, third-party reference only: https://deepctr-doc.readthedocs.io/en/latest/_modules/deepctr/models/multitask/ple.html
- Multi-behavior SR survey: https://www.sciengine.com/doi/10.1007/s11432-024-4568-7
- HyMoERec shared/specialized sequential MoE: https://arxiv.org/abs/2511.06388
- MEMBER multi-behavior MoE: https://arxiv.org/abs/2508.19507
- FAME sequential MoE: https://arxiv.org/abs/2411.01457
