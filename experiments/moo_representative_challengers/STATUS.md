> Повторная отправка: FERERO **4313101**, MosT **4313102**, PHN-HVI **4313103**.
> Код jobs: `1a98ef966a2923a8f234b71602291b1c71ab82c3`, committed/pushed до sbatch.
> Submission records: submissions/*_smoke_002.json; снимок очереди:
> deployment/smoke_002_snapshot.json. Старые попытки сохранены; shared data/env
> и научные параметры не менялись. Sanity/convergence ещё не отправлены.

> Обновление 8 сентября, после 23:00 МСК: все smoke_001 завершились FAILED до
> запуска Python: на compute node отсутствовал Git в PATH. Подтверждения сохранены
> в deployment/smoke_001_failures.json. Исправлена загрузка shared Git module до
> provenance checks; повторные smoke используют frozen attempt 002. Gate sanity
> проверяет smoke_002. Локально 28/28 checks passed. Ниже сохранён исходный снимок
> до первых запусков; его PENDING и 27 checks относятся к прежнему состоянию.

# Статус representative challengers — 8 сентября 2026

Снимок очереди: 08:20 UTC / 11:20 МСК. Реализация и локальные проверки завершены;
три smoke отправлены, но ещё не исполнялись. Sanity и convergence не отправлены:
их обязательные предыдущие этапы пока не прошли. Это отчёт о реализации и запуске,
а не о полученных научных результатах.

## A. Git

- Проверенная база main: `7bc8f5acbf513de5dc71a8dcc0eef0635f56f802`.
- Ветка: `exp/moo-representative-challengers`.
- Начальный implementation commit: `eb33ef08e0283b66f761efe87e09c4a8f89051bc`.
- Финальный implementation/job commit: `eeaab591311851d61fac738843adecc53152261d`.
- Код committed и pushed до sbatch; локальный и кластерный checkout были clean.
- Этот отчёт, deployment evidence и submission records публикуются отдельным
  commit. Его SHA можно получить через `git log -1 --format=%H -- STATUS.md`
  из этой директории. Он не заменяет SHA кода уже отправленных jobs.
- Кластерный checkout оставлен на job commit; не обновлять его до завершения
  отправленных задач: Slurm проверяет точный HEAD.
- Canonical README, reports, experiments/results.csv и moo_8families не изменены.
  Контрольные суммы 71 исторического файла совпали.

Полный список файлов относительно base приведён в [deployment/changed_files.txt](deployment/changed_files.txt).

## B. Методы и fidelity

Все три реализации явно помечены как adaptations; exact reproduction не заявляется.
Полная аргументация: [DESIGN.md](DESIGN.md), источники: [provenance.yaml](provenance.yaml).

| Метод | Upstream commit | Лицензия pinned upstream | Параметризация | Parity |
|---|---|---|---|---|
| FERERO-adapter, method-level | `e350248c4e558173b740bc2ca1e02feb5e448253` | LICENSE не найден; исходники не vendored | 6 независимых TiM4Rec по старым preferences | passed |
| MosT-style, method-level | `726f20179a9cefd3a0f0d82e7b2bf915d50914a1` | LICENSE не найден; исходники не vendored | 3 независимых TiM4Rec | passed |
| PHN-HVI-adapter, family-level | `63e818a8bc780ceb5deeae28d56360d884b8c923` | MIT | FiLM 5→64→128, 8704 параметра, scale 0.1; 8 simultaneous rays | passed |

Официальные repositories: [FERERO](https://github.com/lisha-chen/FERERO),
[MosT](https://github.com/tianyi-lab/MosT),
[MultiSample-Hypernetworks](https://github.com/longhp1618/MultiSample-Hypernetworks).
Оригинальный PHN: `AvivNavon/pareto-hypernetworks`, commit
`6355fd28a05560806ba88ec898db4e6b46ba5ed9`, MIT.

FERERO: официальный dual subproblem перенесён на 5 задач; shared-backbone Gram,
внешний Adam и замороженный spectral step cap — адаптации. Равенства задают отношения
losses по inverse-preference ray, без придуманных performance bounds. Parity с toy
solver проверена в режиме неактивного cap; для cap отдельно проверена устойчивость.

MosT: minibatch gradients заменяют federated client deltas, внешний optimizer — Adam.
Сохранены IPOT, marginals, EMA, masking, conditional MGDA, официальное сочетание
коэффициентов с raw gradients и фактически одна итерация upstream Frank–Wolfe.
Нет warmup и preference в оптимизаторе. Проверена полная GlobalUpdate для OT/MGDA.

PHN-HVI: генерируется adapter, не все веса TiM4Rec. Сохранены multifront normalized
HV gradients, duplicates/ties, dynamic reference и negative cosine. HVI получает
batch-mean losses, как официальный Jura trainer. Preference input сохраняет прежний
смысл importance; cosine использует inverse-priority loss ray — явно объявленное
отличие. Нет дополнительного 30-epoch warmup. RNG-exact replay проверен против
совместного backward; проверены чувствительность к preference и gradients hypernet.
Full-weight PHN технически возможен на A100, но оценён в 38 594 654 параметра generator
и около 589 MiB для FP32 weights/grads/Adam; причина adapter — контролируемое сравнение,
а не заявление о невозможности full PHN.

Локально **27/27 проверок passed**, без dataset и GPU: parity, реальный общий training
loop на синтетической модели с пятью heads, gradients, selection, file guards и
исторические hashes. [verification.json](verification.json) привязан к source digest
`44340ac96c4a0e0f2080c7a5e46ab8ef4c452209383846cd984757a3bf1eee0d`.
Поле git_commit_at_verification отражает HEAD до commit проверенных изменений;
содержимое проверяется отдельным digest. GPU smoke пока не пройден.

## C. Scientific freeze

Protocol B fingerprint:
`954d8abff424b5a57daa74f361ab0f8309cf93121fcc12ef10569d2df11144c7`.
23 951 users, 7 111 items, 1 134 420 interactions; TRAIN 1 086 518,
VALID 23 951. Размер TEST 23 951 известен только из metadata.
Sequential TRAIN examples: 1 062 567.

Seed 2026. Порядок objectives: rank, is_click, long_view, is_like, is_profile_enter.
Исходные tuned parameters зафиксированы в [config.yaml](config.yaml), без нового tuning.
TRAIN diagnostic normalization: 8 batches. Preferences:

| Preference | Вектор в порядке objectives |
|---|---|
| balanced | [0.2, 0.2, 0.2, 0.2, 0.2] |
| rank_heavy | [0.6, 0.1, 0.1, 0.1, 0.1] |
| click_heavy | [0.2, 0.5, 0.1, 0.1, 0.1] |
| long_heavy | [0.2, 0.1, 0.5, 0.1, 0.1] |
| like_heavy | [0.2, 0.1, 0.1, 0.5, 0.1] |
| profile_heavy | [0.2, 0.1, 0.1, 0.1, 0.5] |

Smoke: 3 training batches. Sanity: 5 полных epochs, validation каждый epoch.
Convergence: максимум 100 epochs, validation каждые 5, minimum 20,
early stopping patience 3 validation checks, min_delta 0.

FERERO и PHN-HVI: фиксированная rank_heavy точка. MosT: минимизация
sum(r_j*z_ij), r=rank_heavy, z=(1−NDCG@10, четыре BCE)/(1,2,2,2,2),
при равенстве — индекс solution. Выбор checkpoint по NDCG@10 выбранного solution.
MosT не выбирает solution через oracle max NDCG.

## D. TEST hygiene

TEST dataset loaded: **NO**. TEST dataloader created: **NO**.
TEST evaluation count: **0**. Optuna studies/trials: **0**.
Зависимость Optuna присутствует в старом зафиксированном окружении, но tuning не запускался.
Локальные unit/parity tests не являются оценкой на TEST split.
Read-only preflight проверил TRAIN, VALID и item SHA256, не открывая TEST.

## E. Cluster

Partition **rocky**, constraint **type_e**, GPU A100×1, CPU×4, mem=0,
time limit 24 часа, requeue=0. type_e — constraint, а не partition.

| Метод/этап | Job ID | State | Exit code | Partition | Назначенный node | Runtime | Причина |
|---|---|---|---|---|---|---|---|
| FERERO smoke | 4311825 | PENDING | 0:0* | rocky | нет | 00:00:00 | Priority |
| FERERO sanity | — | NOT_SUBMITTED | — | — | — | — | smoke gate не пройден |
| FERERO convergence | — | NOT_SUBMITTED | — | — | — | — | sanity gate не пройден |
| MosT smoke | 4311826 | PENDING | 0:0* | rocky | нет | 00:00:00 | Priority |
| MosT sanity | — | NOT_SUBMITTED | — | — | — | — | smoke gate не пройден |
| MosT convergence | — | NOT_SUBMITTED | — | — | — | — | sanity gate не пройден |
| PHN-HVI smoke | 4311827 | PENDING | 0:0* | rocky | нет | 00:00:00 | Priority |
| PHN-HVI sanity | — | NOT_SUBMITTED | — | — | — | — | smoke gate не пройден |
| PHN-HVI convergence | — | NOT_SUBMITTED | — | — | — | sanity gate не пройден |

*0:0 — текущее поле sacct для ещё не запущенной задачи, не успешное завершение.
Предварительные start estimates Slurm: FERERO/MosT 2026-09-08 14:03:11,
PHN-HVI 2026-09-09 03:08:48 (время кластера). SchedNodeList: cn-043, cn-046,
cn-044 соответственно; это прогноз, не выделение. Оценки могут меняться.
Partition up, sinfo -R не показывает DOWN/DRAIN reasons в этом снимке.
Полные read-only ответы: [cluster_snapshot.json](deployment/cluster_snapshot.json).

Изолированный checkout: `/home/daryumin/iberdov/diplom_exp_moo_challengers`.
Private Python: `experiments/moo_representative_challengers/runtime/env/bin/python3.10`
относительно checkout. Существующий shared env не изменялся.
90 исходных package specs совпали по PEP 440, pip check и imports прошли.
PyTorch 2.3.0+cu118, NumPy 1.26.4, RecBole 1.2.0, mamba_ssm 2.2.2,
causal_conv1d 1.2.2.post1. Native wheels взяты из официальных releases;
URLs, SHA256 и установленный freeze сохранены в deployment/.

При подготовке обнаружена недоступность Lustre OST0: часть старого Python env
и validation IDs sidecar не читалась. Только новый checkout/runtime создан на
OST7; старые файлы и stripe settings не изменялись. Неудачный первый clone сохранён
как `/home/daryumin/iberdov/diplom_exp_moo_challengers.failed_clone_20260907`.
VALID IDs извлекаются в памяти из уже существующего VALID inter с проверкой
полного SHA256 и исходного sidecar SHA256; все 23 951 IDs совпали.
Dataset/sidecar не пересоздавался и не копировался. Это обход конкретной проблемы
чтения, не подтверждение исправления всего Lustre. Evidence: [data_preflight.json](deployment/data_preflight.json).

## F. Results

Completed convergence отсутствует для всех трёх методов. HR/NDCG, best epoch и
objective vector пока **не получены**. Подставлять historical metrics запрещено.
После smoke → sanity → convergence требуется собрать small JSON и notes отдельным
results commit; текущий evidence commit не является results commit.

## G. Historical references

Stage-1 validation NDCG@10: EPO **0.0584**, GradHV **0.0486**,
PHN-adapter **0.0423**. Это прежние reference значения, не новые результаты.
Stage-2 EPO 0.0588 и GradHV 0.0488 — только дополнительный контекст.
Исторический GradHV использовал max-NDCG выбор solution; selection policy отличается
от замороженного scalarized MosT, поэтому сравнение требует этой оговорки.
Решение «заменить / оставить representative» не принято.

## H. Старые EPO+MoE jobs

| Job | Вариант | Последний state | Exit | Runtime последней попытки | Node |
|---|---|---|---|---|---|
| 4300861 | M0 | FAILED | 1:0 | 00:00:11 | cn-044 |
| 4300862 | M2 | FAILED | 1:0 | 00:00:06 | cn-044 |
| 4300863 | M4 | FAILED | 1:0 | 00:00:06 | cn-044 |
| 4300864 | M8 | FAILED | 1:0 | 00:00:06 | cn-044 |

Все финальные попытки завершились 5 сентября 2026, 22:43–22:44 по кластеру.
До этого sacct -D фиксирует PREEMPTED; после автоматического requeue старый runner
завершился на защите от перезаписи существующих artifacts. Ранее проверенные stderr
содержали этот отказ. У M0/M2/M4 были частичные epoch-1 checkpoints без validation;
у M8 checkpoint не найден. Успешных итоговых result JSON нет.
Старые jobs **не отменялись, не менялись и не перезапускались нами**.

## I. Границы текущего результата

PR не создан, merge не выполнен. TEST, Optuna, clustering, Mamba-3 и новые MoE
архитектуры не запускались. Новые jobs не дублировать из-за PENDING.
Продолжение: проверить завершение smoke и его scientific gate; только затем отправить
sanity соответствующего метода, далее аналогично convergence. До результатов
научный выбор представителей остаётся открытым.
