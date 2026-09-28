# Подтверждающая серия MIMO base / dual / triple

KuaiRand: хронологический leave-one-out, полный каталог. Только TRAIN → VALID;
TEST loader не создаётся, TEST=NOT_RUN и test_evaluation_count=0.

## Заранее фиксированный вопрос

Первичный контраст: triple − dual на четырёх новых seeds2027–2030.
Дополнительные: dual − base, triple − base. [Study plan](study_plan.json)
фиксирует 12 fresh fits в порядке base → dual → triple внутри каждого seed,
seeds по возрастанию. Seed2026 повторно не обучается и не оценивается.
Низкая метрика triple не останавливает серию и не меняет план.

Новые четыре тройки анализируются отдельно от всех пяти с exploratory pilot2026.
Pilot уже повлиял на решение продолжить исследование и не является независимым
подтверждением. [Завершённый пилот](../../../reports/MAMBA3_TIME_MECHANISMS_RESULTS.md#mimo-time-pilot):
base=.0590, dual=.0633, triple=.0618. Автоматического выбора backbone нет.

## Неизменная процедура

`config` и `effective_config` берутся из успешных JSON каждого режима
[attempt003](../runs/attempt_003/pilot_summary.json). Разрешены только seed,
run_id, output paths и provenance новой серии. Собственные config/runner не
подменяют globals старого пилота. Сохраняются dataset, модель, kernels,
calibrators, neutral length adapter, trainer, fit/ties и detached VALID collector.

MIMO rank4/chunk8, история50, kernel length56, layers2, temporal heads2,
backendupstream, counts714888/715020/715086. TRAIN reference838393ms,
bounds[.5,2]. Adam .001, CE, weight_decay0, train2048/eval4096, max300 epochs,
eval_step1, stopping_step10. Последний равный максимум округлённого VALID NDCG@10;
никакого изменения правил ради заполнения окна27 или бюджета времени.
Mixer bf16; AMP/scaler/TF32 flags, data hashes, каталог, masking и repeated targets
сверяются с сохранёнными значениями пилота до каждого fit.

Setup order: runtime/config → seed → dataset/loaders → model → trainer → fit.
Reserved TEST из dataset.build удаляется без loader/evaluation. Каждый fit
начинается в новом процессе с нуля. Checkpoint: pure state_dict + metadata;
исторические веса не загружаются. Diagnostics только в штатных VALID passes,
с отдельным RNG; first-batch hash фиксируется внутри обычного train loop.

## Парность и допуск

Внутри seed проверяются backbone, общие calibrators, независимость write/phase
storage, Python/NumPy/CPU/CUDA/DataLoader RNG, canonical Adam settings, precision,
данные и первый потреблённый batch. У base calibrators нет. Между seeds равенство
инициализации не требуется. RNG считывается без расходования.

CPU-preflight создаёт модели, но не делает forward, optimizer step, fit или
evaluation. На seed2026 сверяет сохранённые initial backbone/calibrators и
Python/NumPy/CPU hashes. CUDA/loader/first-batch на CPU повторно не воспроизводятся:
в evidence явно NOT_REPLAYED_CPU/NOT_REPLAYED_NO_FIT, не ложный PASS.
На новых fits все эти поля обязательны и проверяются внутри каждой тройки.

Наследуются admission45/45,2342/2342 и smokePASS из job4358147,
execution `c1dd31eec7907c67348769b6a1811b89aa4011c0`:

- admission SHA256: `c673524fda1a41d472c6cafebca7e25dad5cd7bd5df99fa772dfdcd4ee8bd5bc`;
- smoke SHA256: `ac17c67b3fc411fce5cbdd53a855e75392d5bfd953b29a434e7237f135e7db9b`;
- policy SHA256: `216fbde21dcd5d349857bef8ef963e535d92aefee1e971e0de91d67385de43b7`.

45 GPU cases и smoke не повторяются. Content hashes всех исходных зависимостей
старого manifest, реальные required leaves, dtype/rank/chunk/config и installed
Mamba pin проверяются заново без изменения исходного evidence/job_id.
Собственный `runs/inherited_admission_001.json` связывает новый allocation с
историческим допуском. Любое изменение математики/policy блокирует наследование.
Legacy exact-zero FAIL остаются историческими FAIL; допуски не расширяются.
Hook/no_grad проверяются по сохранённым 20 реальным GPU phases и неизменности
production helper. На login они не переисполняются новым model forward.

## Проверки и запуск

Локально: unittest, compileall, bash -n и diff-check. В существующем frozen env
на кластере: те же CPU regressions, реальный Adam JSON roundtrip, parameterized
construction всех seeds, effective Config/reference Config, временные fixtures,
production subprocess masks, no-Git launcher/preflight, tampering/ownership,
no-overwrite, duplicate/ambiguous submit, missing/negative aggregation.
Ничего не устанавливается. Все synthetic records тестов только в temporary dirs.

Login проверяет exact published commit и его Git blobs; compute использует
content hashes/manifest/reservation, Git ему не нужен. `submit.py` проверяет
сохранённые CPU/preflight/ownership records, атомарно создаёт login/reservation
до единственного sbatch. Неопределённый ответ блокирует повторный submit.

Один job: rocky/proj_1833/type_e, A100×1, CPU4, mem0, 08:00:00, no-requeue.
Runtime/inherited-admission → 12 fresh subprocess fits → JSON/Markdown summary.
Внутренний deadline: минимум Slurm end и start+8h, минус600s на запись.
Новый fit не начинается, если осталось меньше5400s. Это консервативный порог
из пилота, не гарантия завершения12 fits. Epochs/early stopping не сокращаются.
Timeout сохраняет INCOMPLETE; технический FAIL останавливает следующие fits.
Retry/requeue/дополнительные seeds/TEST запрещены.

Каждый result содержит identity, runtime/data/config hashes, initialization,
first batch, полную history, best epoch/diagnostics, checkpoint SHA, timing и
измеренную память. Время включает JIT/cache effects, не является чистой latency.
`runs/confirmation_summary.json` и `.md` сохраняют обе выборки, три контраста,
mean/sample std(ddof1), знаки, n_expected/n_available, конкретные paired seed sets.
Пропуски не заменяются нулями; средние моделей используют одинаковые seeds.
First27 показывает наблюдаемые эпохи и incomplete window, без заполнения.

После Job ID никаких ожиданий или опросов. Незавершённая серия остаётся в этой
ветке и не добавляется в main/results.csv до отдельной проверки результатов.
