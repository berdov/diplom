# Centered Gap-Trap: подтверждение на четырёх новых seeds

Цель — проверить единственный primary contrast `centered_gap_trap − fixed_replay`
по VALID NDCG@10 на seeds2027–2030. Пилот2026 известен заранее: 0.0635 против
0.0633, +0.316%, но first27 ниже. Он не входит в primary aggregate.

## Зафиксированный дизайн

Используются опубликованные centered model/modulation/trainer/state без изменений:
`q=(g−R0)/(g+R0)`, `T'=T+alpha*q`, R0=838393 мс. Alpha — один fp32 scalar,
общий для heads/layers, init ровно0, bounds[0,1], прежний Adam с projection.
First event/padding дают q0; active real zero-gap даёт q−1. Target timestamp
не используется. Старый endpoint guard, dtype и upstream kernel сохранены.

Основа: MIMO dual fixed-reference, rank4/chunk8, две temporal heads, два слоя,
history50/padding56, hidden64/state128/expand2/headdim64/ngroups1,
rope_fraction0.5, dropout0.2, FFN256. Counts715020/715021.
Adam0.001, batch2048, max300epochs, stopping_step10. Выбор — последняя эпоха
с максимальным округлённым VALID NDCG@10, прежняя RecBole semantics.

Данные и preprocessing прежние: KuaiRand-Pure,23951users,7111realitems,
1134420interactions, TRAIN1086518interactions/1062567examples, VALID23951.
Хронологический user-wise leave-one-out, full catalog, padding excluded,
seen items not excluded, повторные targets допустимы. Global point-in-time
сопоставимость не заявляется. TEST не оценивается, count0.

Порядок:2027fixed→centered,2028fixed→centered,2029fixed→centered,
2030fixed→centered. Ровно8freshfits в отдельных процессах на одной A100.
Новых seeds, tuning, изменений формулы/alpha/R0/precision нет.

## Проверки и происхождение

Основа — main `e8f8f1ecce0646c5815b524d0bd9683c423b4775`.
Centered execution `e4e31370e29c2a34c5f0f1071046fccebd77c289`,
source hash `638b836780af7b6f9065546c128cb8ac2451c5d8af3581855029eea515c2ca10`.
Полная dependency closure включена в новый manifest; drift запрещает запуск.
Plan входит в manifest, поэтому итоговый source hash хранится в manifest
и связывается с plan SHA в reservation/results без самоссылки.

Наследуются job4371876 gate6/158 и smoke, прежние one-sided4/133,
head-scale9/228 и MIMO45/2342 по SHA. GPU gate/smoke повторно не запускаются.
CPU preflight проверяет четыре seeds, counts, config, RNG, initial state,
пути/порядок, replay/schema mapping, deadline, partial summary и reservation.
Опубликованные21CPU regression tests также выполняются без CUDA.

Между двумя вариантами каждого seed совпадают common backbone, calibrators
с reference buffers, все RNG, DataLoader generator, первый фактически
потреблённый batch, данные, optimizer, precision и selection. Проверка batch
использует существующий trainer hook без дополнительного прохода loader.
Отдельный pair JSON связывает result SHA и общие состояния.

Fresh fixed обязателен. После него, до centered, проверяется точный replay
исторического MIMO dual job4358583, seeds2027–2030: complete history/loss/
metrics, best epoch, first27, initial state/RNG/first batch и checkpoint SHA.
`initial_calibrator_hashes` явно сопоставляется с `initial_common_calibrator_hashes`.
Из config исключены только checkpoint path и новый gap_trap_mode; seed должен
совпасть. Timing/memory не входят в scientific replay identity.

## Ресурсы и остановка

Основной Slurm job: rocky/proj_1833/type_e, A100x1, CPU4, mem0,8h,no-requeue.
Не начинать fit, если до внутреннего deadline осталось меньше90мин;
внутренний deadline на10мин раньше hard deadline. Исторические fixed занимали
10.51–35.45мин TRAIN+VALID, centered pilot24.41мин. Запас консервативный,
но не гарантирует завершение любой траектории до300эпох.

Штатная остановка между fits по этому guard — `PAUSED_DEADLINE`, exit0,
scientific summary INCOMPLETE. Это исчерпание доступного безопасного бюджета,
без ожидания оставшихся90мин на GPU. Разрешён максимум один continuation
с теми же8h limits и тем же execution/source/config, только для never-started
suffix. Общий предел —8starts, максимум2allocations. Continuation требует
terminal COMPLETED0:0, доказанный deadline guard, preservation всех прежних
результатов/metadata/logs/pair JSON/SHA, immutable admission и отдельную reservation.
При TIMEOUT с прерванным fit продолжение не допускается. Любой technical FAIL,
OOM, replay mismatch, неизвестный start или started-incomplete блокирует его.
Автоматических retries нет; завершённые fits никогда не повторяются.

Runtime namespace выделен по study; один physical path на каждый logical fit.
Отдельные allocation locks/reservations не позволяют повторить sbatch.
Неоднозначный ответ sbatch сохраняется до разбора, повторная отправка запрещена.
Git используется только на login при проверке опубликованных blobs; compute
проверяет содержимое без Git. Existing untracked artifacts не удаляются.

## Анализ и публикация

Primary — только четыре новые полные пары: mean±sample std(ddof1) для моделей
и paired delta, знаки+/−/0, относительная разница средних. Никаких p-values,
post-hoc thresholds или выбора лучшего seed. Все5seeds включая exploratory
pilot показываются отдельно, вторично. HR/Recall — descriptive.

First27 — полное окно0–26 и общий available paired subset. Отдельно показать
best/actual epochs; поздний выигрыш не означает причинный эффект extra epochs.
Для alpha сохранить best/max-at-epoch-boundaries/final/zero count/trajectory,
q/shift/odds/sigmoid и BF16 на прежней10точечной сетке. Только аналитические
вычисления из сохранённых diagnostics; никаких новых dataset forwards.
Alpha — глобальная сила поправки Trap, не пользовательская шкала времени.

Публикация в main только после8/8fits,4/4pairs, полного terminal audit и TEST0.
При partial — только evidence/handoff в ветке. Registry112→120 только после
фактического подсчёта и сохранения старого byte prefix. Сохранить compact logs,
raw JSON, plan/index/manifest, checkpoint path/SHA/bytes; weights остаются на cluster.
Для завершённой серии подготовить paired_delta.svg и valid_table.tex в reports/assets.

Сохранена оговорка Mag-Mamba: gap-conditioned updates уже исследовались;
эта серия проверяет конкретную controlled parameterization, без claims о первенстве.
VALID не сравнивается с внешним TEST как доказанный апгрейд. Статья/Overleaf
не меняются. По результатам дать рекомендацию fixed или centered как provisional
основы; layer-specific temporal functions в этой работе не реализуются и не запускаются.
