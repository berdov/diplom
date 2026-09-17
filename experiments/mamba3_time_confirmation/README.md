# Подтверждение shared/separate Mamba3

Серия завершена: сохранены 9/9 новых VALID JSON и две исторические ссылки seed 2026. [Итоговые таблицы, график и ограничения](../../reports/MAMBA3_TIME_MECHANISMS_RESULTS.md#confirmation).

KuaiRand: хронологический leave-one-out, оценка по полному каталогу.
**Только TRAIN → VALID. Новый TEST запрещён.**

[Versioned study plan](study_plan.json) фиксирует 5 matched seeds (2026–2030).
2026 shared/separate переиспользуются побайтно с original execution commits:
`e81129a66c818a570ab589b4e255da08aac22422` и
`9334bd93c8fa30fbacabaeb220cef17e336c261b`. Эти runs не переобучаются и не оцениваются.
Проверены settings, .inter/stats SHA и порядок seed → dataset → loaders → model.
Единственная нормализация сериализации: YAML GPU ID `"0"` равен JSON `0` после Config.
Научных различий конфигурации нет; shared использует свой frozen RT class,
separate — frozen mechanism class. Original JSON не переписываются.

## Зафиксированный план

| Task | Mode | Seed | Run ID |
|---:|---|---:|---|
| 0 | shared | 2027 | mamba3_confirm_shared_seed2027_001 |
| 1 | separate | 2027 | mamba3_confirm_separate_seed2027_001 |
| 2 | shared | 2028 | mamba3_confirm_shared_seed2028_001 |
| 3 | separate | 2028 | mamba3_confirm_separate_seed2028_001 |
| 4 | shared | 2029 | mamba3_confirm_shared_seed2029_001 |
| 5 | separate | 2029 | mamba3_confirm_separate_seed2029_001 |
| 6 | shared | 2030 | mamba3_confirm_shared_seed2030_001 |
| 7 | separate | 2030 | mamba3_confirm_separate_seed2030_001 |
| 8 | separate_constant_gap | 2026 | mamba3_confirm_separate_constant_gap_seed2026_001 |

Все новые обучения с нуля: Adam .001, CE, max300, stopping_step10,
eval_step1, batch2048/eval4096, history50. Сохранена RecBole tie/early-stop
семантика: update при `>=`, stop после `cur_step>10`. Нет фиксирования epochs=15/51.

## Constant-gap

Тонкий wrapper создаёт timestamps `position * 838393 ms` только по items/lengths
и вызывает неизменённый `MechanismMamba3Rec.forward`. Реальные timestamps
вообще не читаются этим wrapper. Каждый active gap, включая исходный нулевой,
становится reference, tau=log(2). First/padding остаются neutral=1.
Два независимых calibrator, 610572 параметра, те же zero-init и bounds [0.5,2].
Scales обучаются с первого шага и не обязаны равняться друг другу или единице.
Это один exploratory control, не пятисидовый результат.

## Защиты и smoke

- Frozen Python/YAML и GPU evidence не изменены. [Собственный source manifest](source_manifest.json)
  включает новый harness/config/plan/tests/launcher; compute-node проверяет Python hashes, не git.
- GPU PASS старых компонентов не выдаётся за проверку нового harness.
- Перед scientific training каждая allocation проверяет настоящие модели:
  paired backbone init, counts, synthetic forward/backward, finite gradients,
  обновление calibrator, constant eval invariance после update, checkpoint roundtrip.
- Затем все smoke objects отбрасываются, RNG сбрасывается и создаются свежие
  dataset/loaders/model/optimizer в reference-порядке. Диагностика имеет отдельный RNG.
- CPU tests проверяют temporal count deltas и wrapper с явно обозначенным CPU receiver;
  это не GPU evidence. Login preflight дополнительно считает полные CPU-модели и init SHA
  без forward/dataset; реальные GPU gates выполняются только внутри array allocation.
- Result или атомарный lock запрещают overwrite/automatic retry. FAIL сохраняется,
  lock не удаляется. Повтор после ручного расследования требует отдельного решения.
- `runs/<run_id>.json` атомарно записывается; runtime/checkpoint/TensorBoard/cache
  уникальны в ignored `slurm_logs/<run_id>/`. Slurm stdout/stderr: `%A_%a`.
- Только один конкретный VALID loader допускается guard в Trainer. TEST loader не создаётся.

## Агрегация

`python -m experiments.mamba3_time_confirmation.aggregate` только читает JSON.
Все полные matched pairs: mean/sample std (ddof=1), каждая paired delta,
её mean/std и количества +/-/0; `n_expected=5`, `n_available`, incomplete flag.
Плохие successful runs не исключаются. Неполные/ошибочные runs перечислены отдельно.
Best within first27 показывается отдельно только при полных histories всех доступных пар;
при n<5 явно остаётся incomplete study. Control не смешивается с matched seeds.
Ни aggregator, ни jobs не пишут общий results.csv или Git.

## Кластерные entrypoints завершённой серии

[Launcher](../../slurm/mamba3_time_confirmation.sh): rocky/proj_1833/type_e,
1 A100, mem0, array `0-8%1`, no-requeue; существующий envs/mamba3.
[Preflight](preflight.py) не загружает dataset и не вызывает model forward.
[Submit helper](submit.py) создаёт durable `slurm_logs/submission_001.json` до
единственного sbatch и сохраняет ID немедленно. После ID мониторинга нет.

Исторический тест `test_frozen_files_byte_stable` в старой ветке проверял также
CSV/документы относительно pre-results main; после разрешённой канонизации он
не применим. Он сохранён как historical source. Новый gate проверяет непосредственно
старый core fingerprint/evidence и reuse SHA, а не требует отката новых таблиц.
