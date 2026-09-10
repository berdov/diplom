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
| like + profile_enter | 1/3 | 0.059500 ± — |
| click + like + profile_enter | 1/3 | 0.059500 ± — |

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
| like + profile_enter | HR@5 | 0.065800 | — |
| like + profile_enter | HR@10 | 0.109300 | — |
| like + profile_enter | HR@20 | 0.178400 | — |
| like + profile_enter | HR@50 | 0.318700 | — |
| like + profile_enter | NDCG@5 | 0.045600 | — |
| like + profile_enter | NDCG@10 | 0.059500 | — |
| like + profile_enter | NDCG@20 | 0.076800 | — |
| like + profile_enter | NDCG@50 | 0.104400 | — |
| like + profile_enter | Recall@5 | 0.065800 | — |
| like + profile_enter | Recall@10 | 0.109300 | — |
| like + profile_enter | Recall@20 | 0.178400 | — |
| like + profile_enter | Recall@50 | 0.318700 | — |
| click + like + profile_enter | HR@5 | 0.066500 | — |
| click + like + profile_enter | HR@10 | 0.109200 | — |
| click + like + profile_enter | HR@20 | 0.178000 | — |
| click + like + profile_enter | HR@50 | 0.320200 | — |
| click + like + profile_enter | NDCG@5 | 0.045800 | — |
| click + like + profile_enter | NDCG@10 | 0.059500 | — |
| click + like + profile_enter | NDCG@20 | 0.076800 | — |
| click + like + profile_enter | NDCG@50 | 0.104800 | — |
| click + like + profile_enter | Recall@5 | 0.066500 | — |
| click + like + profile_enter | Recall@10 | 0.109200 | — |
| click + like + profile_enter | Recall@20 | 0.178000 | — |
| click + like + profile_enter | Recall@50 | 0.320200 | — |

## Paired NDCG@10 differences

| Comparison | Seed deltas | Mean |
| --- | --- | ---: |
| like + profile_enter minus primary_only | 2026: +0.0007 | +0.000700 |
| click + like + profile_enter minus primary_only | 2026: +0.0007 | +0.000700 |
| click + like + profile_enter minus like + profile_enter | 2026: +0.0000 | +0.000000 |

## Можно ли заморозить MTL target set?

Пока нет: confirmation не завершён. Нельзя выдавать screening winner за multi-seed подтверждение.

[Машиночитаемая сводка](../experiments/stage_confirmation/summary.json). Исходные screening/challenger результаты не изменялись.

## Отправленные confirmation jobs

Массив **4317564_0–5**, CPU-сводка **4317565**; code SHA `8cca489f8112a425f3c03956183fd785e7fdda5c`. На 10 сентября 18:09 МСК — PENDING (Priority), оценка старта недоступна. Все шесть preflight прошли. [Job ledger](../experiments/stage_confirmation/submissions/pipeline.json).

Первоначальные 4316786/4316787 отменены до начала выполнения из-за ошибки ordinal mapping; выбор исправлен на точные наборы таргетов и покрыт проверкой. Новые результаты ещё не получены; full confirmation и окончательное закрытие MTL остаются незавершёнными.
