# Контекстная калибровка времени Mamba3

**Эксперимент завершён:** GPU gates и все пять запусков прошли проверки. На seed2026 routed и uniform получили VALID NDCG@10 **0.0628** против **0.0633** у separate replay; dense12 получил **0.0635**, без многосидового подтверждения. Экспертный вариант пока не выбран основной моделью. [Таблица, график и диагностика](../../reports/MAMBA3_TIME_MECHANISMS_RESULTS.md#context-time-pilot).

Один seed=2026, только TRAIN → full-ranking VALID на frozen KuaiRand chronological split. TEST loader не создаётся. Это адаптация/контроль известного класса механизмов, не заявление новизны или SOTA.

## Вопрос и формулы

Нужен ли item-local контекст дополнительным temporal calibrators и помогает ли экспертная факторизация сверх обычного dense calibrator? Базовые A и DT Mamba уже content-dependent; доказанного дефекта backbone не заявляем. [Обзор и ограничения доступа](../../reports/EXPERTS_REVIEW.md): учтены HM2Rec, Swimba, STM3, MoM, KVAE. Full Methods TriSSR/TMTRec пока недоступны; совпадение не исключено.

Вход: история item IDs `[B,L]`, lengths `[B]`, исторические timestamps float64 `[B,L]`; `L<=50`. `u:[B,L,64]` берётся из embedding до dropout/norm. Это обучаемые координаты текущего исторического item, не атрибуты видео и не полный user state. Нет target timestamp/item во входе encoder, idle gap, t_score, user features или будущих событий.

```text
tau = log1p(gap / 838393 ms)                 [B,L,1]
q[e,p] = log(2)*tanh(MLP_1->16->2[e,p](tau)) [B,L,4,2 paths,2 heads]
pi = softmax(Linear_65->4(concat(u,tau)))    [B,L,4]
ell[p,h] = sum_e pi[e]*q[e,p,h]
s[p,h] = exp(ell[p,h])                      [B,L,2 paths,2 heads]
ADT = A*(DT_base*s_decay)
DT = DT_base*s_scan
```

Один router выбирает пару decay/scan функций. Bounded log-scales смешиваются до exp, а не после. Два heads не являются двумя layers: scales вычисляются один раз и используются обеими layers. DT влияет также на input weighting. Bounds `[0.5,2]` сохраняют положительность DT и знак A, но не доказывают устойчивость всей нелинейной модели. Все эксперты активны; новых scans/states нет, sparse speedup не обещается.

Безопасный tau повторяет float64 logaddexp frozen calibrator, затем cast в float32. Router/scales fp32, прежний mixer bf16. First/padding нейтральны; настоящий zero gap активен. Новых losses/нормализаций/autocast нет.

## Фиксированные режимы

| mode | temporal module | temporal params | total params |
|---|---|---:|---:|
| separate_replay | frozen separate | 132 | 610572 |
| dense11 | 65→11→4, SiLU | 774 | 611214 |
| dense12 | 65→12→4, SiLU | 844 | 611284 |
| uniform | четыре пары 1→16→2, pi=1/4, без router | 528 | 610968 |
| routed | тот же bank + Linear(65,4), softmax T=1 | 792 | 611232 |

Counts проверяются на реальных temporal parameters локально и полных моделях в login preflight/GPU gate. Dense widths заранее фиксированы по обе стороны routed budget, не точно равны по capacity/compute/optimization. Никаких других режимов в ночном бюджете.

Исторический `test_frozen_files_byte_stable` отдельно воспроизведён: FAIL относительно старого `de0a137`, поскольку уже в исходном `origin/main` изменились три README, общий results.csv и reports/RESULTS.md. Эти файлы в данном задании не менялись. Это не скрывается как PASS и не обходится правкой frozen manifests; core и confirmation hashes проверяются отдельно.

Последние Linear zero-init; первые слои экспертов/путей независимы. Router weights Normal(0,.01), bias=0. Uniform/routed имеют одинаковый исходный bank. Identity относится к необученной temporal части при одном backbone, не к trained separate checkpoint. На первом backward нулевой router gradient допустим; ненулевой synthetic fixture проверяет дальнейшую обучаемость. Различие экспертов не навязывается loss или названиями.

RNG-порядок: seed → dataset → loaders → frozen separate construction → замена temporal module внутри `fork_rng` → trainer → fit. Старый calibrator не остаётся зарегистрированным. Логируются backbone/bank hashes, CPU/CUDA/Python/NumPy RNG перед fit и hash фактически потреблённого первого train batch. Последний не извлекается отдельным итератором. Исторический JSON не содержит исходного RNG hash: точное равенство его траектории не обещаем. Новый replay является основным comparator, не шестым независимым seed.

## План и остановки

[План](study_plan.json) фиксирует порядок: separate_replay → dense11 → dense12 → uniform → routed. Все с нуля: Adam .001, CE, weight_decay=0, batches 2048/4096, maxlen=50, epochs=300, eval_step=1, stopping_step=10, selection VALID NDCG@10. Сохранён RecBole fit с его last-equal-maximum tie semantics. Data/stat/source hashes обязательны; реальные TRAIN reference stats не пересчитываются для калибровки.

Один Slurm job: rocky/proj_1833/type_e, 1×A100, 4 CPU, mem=0, 8 часов, no-requeue. Перед fits обязательны runtime/source preflight и synthetic CUDA gate eval/train × L50/L64, atol=1e-6/rtol=1e-5. Gates: vanilla identity всех режимов, separate wrapper, E=1 nonzero reduction с gradients, uniform/routed reduction, finite backward/router signal/expert update, safe roundtrip. CPU mocks не являются GPU evidence.

Каждый stage отдельный процесс. Внутренний deadline 7ч40м; при остатке менее 90 минут следующий fit не начинается. Max epochs не уменьшается. Технический FAIL останавливает цепочку, остальные NOT_RUN, строится partial summary, exit nonzero. Низкая VALID-метрика не останавливает остальные заданные режимы. Retries и дополнительные allocations отсутствуют.

Checkpoint: чистый `state_dict` и отдельный metadata JSON (epoch/mode/config/source/metric/SHA). Best diagnostics соответствуют моменту сохранения, не последней эпохе. Только `weights_only=True` для synthetic roundtrip; исторические trained checkpoints не загружаются. VALID adapter отвергает любой другой loader и загрузку best model внутри evaluate.

## Команды и артефакты

Локально в существующем CPU-окружении, без установки пакетов:

```bash
python -m compileall -q experiments/mamba3_context_time
python -m pytest experiments/mamba3_context_time/tests -q
bash -n slurm/mamba3_context_time.sh
git diff --check
```

До deploy обязательна разовая проверка активных jobs/WorkDir/Command: занятый canonical checkout не переключать. Предыдущие constant-gap runs не отправлять повторно. Сохранить unrelated untracked и научные JSON. После безопасного checkout **точного опубликованного SHA**:

```bash
envs/mamba3/bin/python -m experiments.mamba3_context_time.preflight
export RUN_COMMIT="$(git rev-parse HEAD)"
export EXPECTED_CORE_HASH="$(envs/mamba3/bin/python -c 'from experiments.mamba3_context_time.config import CORE; print(CORE)')"
export EXPECTED_STUDY_HASH="$(envs/mamba3/bin/python -c 'from experiments.mamba3_context_time.provenance import verify; print(verify()["source_hash"])')"
envs/mamba3/bin/python -m experiments.mamba3_context_time.submit
```

`submit` создаёт durable record **до единственного** `sbatch --parsable`; потерянный ответ даёт SUBMISSION_UNKNOWN без retry. После Job ID остановиться, не опрашивать очередь. Compute-node не требует git и ничего не коммитит/публикует.

Относительно этой папки:
- `source_manifest.json`: новые исходники; frozen manifests не меняются.
- `runs/gpu_checks_001.json`: только synthetic CUDA evidence.
- `runs/mamba3_context_<mode>_seed2026_001.json`: raw VALID, epoch history и best diagnostics.
- `slurm_logs/<run_id>/`: отдельные lock, checkpoints, stdout/stderr, TensorBoard, caches.
- `slurm_logs/submission_001.json`, `slurm_logs/pipeline_status.json`: отправка и stages.
- `runs/pilot_summary.json`, `.md`, `.svg`: сводка, raw links/hashes и воспроизводимый график.

SVG строится стандартной библиотекой Python, без matplotlib и внешних ресурсов. JSON/Markdown обязательны и сохраняются до рисования. Ошибка только renderer отмечается `plot_status=FAILED`/`plot_error` без ссылки на изображение и без изменения научного статуса; ошибки научной валидации или записи обязательных файлов по-прежнему останавливают pipeline. FAIL/NOT_RUN показаны статусами, не нулевыми метриками.

Сводка заранее определена: full horizon и первые 27 эпох, deltas относительно replay, historical separate отдельно, runtime/epochs/counts. При неполных/ошибочных runs победитель не выбирается. Dense не хуже routed: преимущество MoE не показано; uniform не хуже routed: адаптивный выбор не подтверждён. Даже выигрыш routed на одном seed не означает statistical significance, временную семантику или превосходство над внешними paper numbers.

Утром одна read-only команда, без обучения:

```bash
ssh hse-karizma 'cat /home/daryumin/iberdov/diplom/experiments/mamba3_context_time/slurm_logs/pipeline_status.json /home/daryumin/iberdov/diplom/experiments/mamba3_context_time/runs/pilot_summary.md'
```
