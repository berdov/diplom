# Закрытие старого экспериментального этапа

Canonical docs merged fast-forward в main: `93053d86eb3b7516ecf748fe060baa179d4e678b`.

## MOO representatives

Fairness operating-point selection MosT vs GradHV закрыта полным post-hoc replay: на epochs 5/10/15/20/25 max VALID NDCG@10 выбирает то же solution_index=1; best epoch=10, stop epoch=25, NDCG@10=0.0522. Новый rerun не нужен. [Проверка](evidence/most_gradhv_selection_replay.json). Это устраняет конкретное различие selection, не все различия методов и бюджетов.

Рабочий representative set можно зафиксировать: **STCH, FAMO, PCGrad, EPO, MosT-style, PHN-HVI-adapter, COSMOS-style, PaLoRA**. Это выбор для следующего этапа по имеющимся VALID экспериментам, без заявления о статистически доказанном превосходстве семейств.

Historical EPO+MoE M0/M2/M4/M8: **technical failure / no scientific result**, jobs 4300861–4300864 FAILED (1:0). Не являются отрицательными научными результатами; не перезапускались.

## MTL multi-seed confirmation

Completeness **3/9**. Seeds 2026/2027/2028 общие для трёх вариантов; 2026 переиспользован из screening. Fixed protocol, без tuning и TEST. std — выборочное стандартное отклонение (ddof=1). До 3/3 seeds средние описывают только доступную часть.

| Combination | Seeds | NDCG@10 mean ± std |
| --- | ---: | ---: |
| primary_only | 1/3 | 0.058800 ± — |
| like | 1/3 | 0.059000 ± — |
| click + long_view + like | 1/3 | 0.058900 ± — |

## Все canonical metrics

| Combination | Metric | Mean | Sample std |
| --- | --- | ---: | ---: |
| primary_only | HR@5 | 0.065700 | — |
| primary_only | HR@10 | 0.107600 | — |
| primary_only | HR@20 | 0.175600 | — |
| primary_only | HR@50 | 0.315300 | — |
| primary_only | NDCG@5 | 0.045400 | — |
| primary_only | NDCG@10 | 0.058800 | — |
| primary_only | NDCG@20 | 0.075900 | — |
| primary_only | NDCG@50 | 0.103400 | — |
| primary_only | Recall@5 | 0.065700 | — |
| primary_only | Recall@10 | 0.107600 | — |
| primary_only | Recall@20 | 0.175600 | — |
| primary_only | Recall@50 | 0.315300 | — |
| like | HR@5 | 0.066900 | — |
| like | HR@10 | 0.107500 | — |
| like | HR@20 | 0.176900 | — |
| like | HR@50 | 0.315100 | — |
| like | NDCG@5 | 0.046000 | — |
| like | NDCG@10 | 0.059000 | — |
| like | NDCG@20 | 0.076400 | — |
| like | NDCG@50 | 0.103700 | — |
| like | Recall@5 | 0.066900 | — |
| like | Recall@10 | 0.107500 | — |
| like | Recall@20 | 0.176900 | — |
| like | Recall@50 | 0.315100 | — |
| click + long_view + like | HR@5 | 0.065500 | — |
| click + long_view + like | HR@10 | 0.107800 | — |
| click + long_view + like | HR@20 | 0.175100 | — |
| click + long_view + like | HR@50 | 0.316600 | — |
| click + long_view + like | NDCG@5 | 0.045400 | — |
| click + long_view + like | NDCG@10 | 0.058900 | — |
| click + long_view + like | NDCG@20 | 0.075800 | — |
| click + long_view + like | NDCG@50 | 0.103600 | — |
| click + long_view + like | Recall@5 | 0.065500 | — |
| click + long_view + like | Recall@10 | 0.107800 | — |
| click + long_view + like | Recall@20 | 0.175100 | — |
| click + long_view + like | Recall@50 | 0.316600 | — |

## Paired NDCG@10 differences

| Comparison | Seed deltas | Mean |
| --- | --- | ---: |
| like minus primary_only | 2026: +0.0002 | +0.000200 |
| click + long_view + like minus primary_only | 2026: +0.0001 | +0.000100 |
| click + long_view + like minus like | 2026: -0.0001 | -0.000100 |

## Можно ли заморозить MTL target set?

Пока нет: confirmation не завершён. Нельзя выдавать screening winner за multi-seed подтверждение.

[Машиночитаемая сводка](../experiments/stage_confirmation/summary.json). Исходные screening/challenger результаты не изменялись.
