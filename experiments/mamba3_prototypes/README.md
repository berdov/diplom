# Proto-Mamba3 v1

Изолированный prototype baseline поверх vanilla Mamba3, без RT/time conditioning,
MTL/MOO, experts, routing, auxiliary losses или prototype supervision.
Reference: [vanilla Mamba3](../mamba3_baseline/README.md), VALID NDCG@10 **0.0584**.
Результатов этого эксперимента пока нет. TEST запрещён.

## Архитектура

[Модель](model.py) наследует frozen `Mamba3Rec` без изменения внутренних Mamba kernels,
embeddings, двух blocks, FFN, final valid hidden state, tied scorer и CE.
[PrototypeFusion](prototypes.py) добавляется после `h [B,64]`:

```text
P: learnable [8,64]
alpha = softmax(normalize(h) @ normalize(P).T / 1.0)  [B,8]
z = alpha @ P                                        [B,64]
gate = sigmoid(Linear(128,64)(concat(h,z)))           [B,64]
h_proto = h + tanh(residual_strength) * gate * z       [B,64]
scores = h_proto @ item_embedding.T
```

`residual_strength` является одним learnable scalar с начальным значением 0.
Поэтому `h_proto == h` точно при initialization. P и gate получают нулевой gradient
на первом backward, scalar начинает обучаться сразу; после его изменения gradients
достигают prototypes/gate. Тесты проверяют это и изменение P после двух steps.
K=8 и temperature=1.0 зафиксированы до результатов, никакого tuning.
Argmax применяется только в detached diagnostics, не в forward assignment.

## Инициализация и TRAIN-only

Backbone научного run инициализируется **с нуля**, как vanilla, seed=2026.
Frozen pretrained checkpoint используется **только как encoder для KMeans**, не
загружается в обучаемый Proto-Mamba3. Это сохраняет тренировочный бюджет backbone;
при этом centroids получены в пространстве другого, обученного encoder, совпадение
координат с новым backbone не гарантируется. Identity residual защищает initial
representation. Нельзя интерпретировать результат как доказанное превосходство
без дальнейших контролей; warm-start/continuation в этом run не используется.

Exact source checkpoint (при отсутствии запуск завершается, vanilla не переобучается):

`/home/daryumin/iberdov/diplom_exp_mamba3_baseline/experiments/mamba3_baseline/checkpoints/Mamba3Rec-Sep-12-2026_17-55-38.pth`

SHA256: `d0bc3bb504daf5df068b6fd5bd635d6454aa223da01c006c63c78da193bb9dbe`.
Проверяются frozen VALID JSON, score 0.0584, checkpoint epoch 15, архитектура,
training settings, pinned Mamba `e9594ce1c732d97440f0332fdc43170a2294dbfa` и
Protocol B `.inter` SHA `e275ded0b330c2827b49ccf567d6784452d6dcbf8cd719dc3009d36eadc2e2cc`.

[Инициализатор](prototype_init.py) передаёт encoder только `item_id_list` и
`item_length` из TRAIN dataset. Encoder работает в eval/no_grad; timestamps и target
items не подаются в forward. Порядок/split остаются штатными chronological Protocol B;
timestamp нужен dataset только для порядка, не является входом Proto-Mamba3.
VALID/TEST examples не передаются в KMeans, TEST loader вообще не создаётся.

L2-normalized h передаётся batch-by-batch в `MiniBatchKMeans.partial_fit`:
K=8, random_state=2026, batch_size=2048, n_init=3, reassignment_ratio=0.01,
один проход по 1062567 TRAIN histories. Это фиксированные initial settings,
не подбор по VALID. Хранятся только текущий batch и 8 centroids, без hidden dump.
Будущий `runs/prototype_init_001.json` включает centroids 8x64, provenance,
число histories, source SHA и параметры KMeans. Он появится внутри job, не заранее.

## Один allocation

[Config loader](config.py) наследует [frozen YAML](../mamba3_baseline/config_kuairand.yaml)
и применяет только [prototype overlay](config_kuairand.yaml). Main/RT ветки не меняются.

[Runner](run.py) и [launcher](../../slurm/mamba3_prototypes_validation.sh):

1. Preflight: imports, frozen checkpoint/config/pin/protocol checks.
2. TRAIN-only streaming KMeans, компактный artifact.
3. Smoke: exact identity на реальном Mamba backbone, два optimizer steps,
   finite gradients, prototype update, 64 VALID histories с full-ranking,
   checkpoint save/load. Ошибка прерывает job до full training.
4. Smoke-модель/optimizer отбрасываются; seed, scientific model и loader generators
   создаются заново. Один TRAIN->VALID: Adam .001, batch 2048/4096, max epochs 300,
   patience 10, eval every epoch, CE, max history 50, full-ranking VALID NDCG@10.
5. Best checkpoint выбирается только по VALID. TEST=NOT_RUN, test_evaluation_count=0.

Имя результата: `runs/mamba3_prototypes_validation_001.json`.
Checkpoint/raw logs/cache находятся в игнорируемом `slurm_logs`, не в Git.
Один persistent lock и существующие artifacts запрещают автоматический retry.
Никакого автоматического второго sbatch или final-test runner нет.

## Diagnostics

Каждый VALID epoch записывает mean assignment probabilities, entropy mean/std,
argmax usage share, mean/max off-diagonal prototype cosine similarity,
raw/effective residual strength и mean gate activation. Сохраняются diagnostics
для best checkpoint и последнего epoch. Это detached monitoring, не loss:
ни collapse, ни entropy не меняют training/selection автоматически.

## Проверки и submit

CPU dependencies: PyTorch, pytest, PyYAML, NumPy, scikit-learn (кластер: 1.5.2).
Существующий Mamba3 environment не изменяется. Из корня repo:

```bash
python -m pytest experiments/mamba3_prototypes/tests -q
python -m compileall -q experiments/mamba3_prototypes
git diff --check
```

Синтетические CPU-тесты не заменяют реальный Mamba smoke; он выполняется в allocation.
До submit mkdir для `experiments/mamba3_prototypes/slurm_logs` и экспорт `RUN_COMMIT`.
Ресурсы: rocky/proj_1833/type_e/A100 x1/mem=0, environment `diplom/envs/mamba3`.
После единственного submit агент останавливается сразу после Job ID:
никаких squeue/sacct/log reads/polling. Получение результатов отдельным запросом.
`experiments/results.csv` на этом этапе не меняется.
## Random-init control (подготовлен, не запущен)

Ветка `exp/mamba3-prototypes-controls` основана на exact commit
`bbb9cd954e929df0e4ebc737be41b8440ecdb0bc`. Исходная ветка и её scientific job
не изменяются. `run_random.py` выбирает explicit mode `random`, а обычный
`run.py` по умолчанию выбирает `kmeans`. Модель, smoke, TRAIN→VALID,
diagnostics, tied scorer и CE переиспользуются без второй реализации модели.

Random control задаёт `P ~ Normal(0, 0.02)` отдельным CPU generator с seed 2026.
Он не использует frozen checkpoint, hidden-state extraction, KMeans или
статистики TRAIN для инициализации. Общий runner импортирует KMeans initializer
только внутри KMeans-ветки. K=8, temperature=1.0, scratch backbone,
нулевой residual strength и все training settings сохранены.

`assert_control_parity` разрешает только различия initialization mode/std и
пути checkpoints. Размер prototype module одинаков: 8769 trainable parameters;
с неизменным vanilla backbone 610440 ожидается 619209. Runner проверяет этот
размер до полного обучения. Smoke отбрасывается, seed/model/loaders создаются
заново; подстановка P не сдвигает RNG обучения. Форма и масштаб начального P
являются частью сравниваемого способа инициализации, не дополнительным tuning.

Будущий run: `mamba3_prototypes_random_validation_001`; отдельные init/smoke/result
JSON и lock исключают перезапись KMeans-run. Метаданные включают
`prototype_initialization=random_normal`, std=.02, seed=2026, K=8,
temperature=1.0, `test_evaluation_count=0`, `TEST=NOT_RUN` и reference commit.
Scientific JSON появится только при отдельно разрешённом запуске, не сейчас.
Launcher `slurm/mamba3_prototypes_random_validation.sh` только подготовлен:
rocky, proj_1833, type_e, 1 A100, mem=0, прежнее Mamba3 environment.

## Coordinate-space caveat

KMeans uses TRAIN examples only, encoded by a validation-selected frozen vanilla checkpoint.

Checkpoint encoder был выбран по VALID, а scientific backbone стартует с нуля.
Поэтому centroids первоначально находятся в пространстве другого, уже обученного
encoder. Это не TEST leakage и не использование VALID examples в KMeans.
Random-init control проверяет пользу такой инициализации сверх самой архитектуры.

## Capacity control: только дизайн

Возможный следующий контроль без prototypes: residual bottleneck
`h + tanh(s) * (W2 * GELU(W1*h + b1) + b2)`, где `s=0`,
`W1: 64→68`, `W2: 68→64`. Это 8837 дополнительных параметров против 8769
(разница 68, около 0.78% от добавленного блока), без assignment и прототипов.
Такой почти capacity-matched контроль сохраняет identity initialization;
он не является exact parameter match. Его размер и постановку надо отдельно
утвердить. Здесь он не реализован и не запускается.
