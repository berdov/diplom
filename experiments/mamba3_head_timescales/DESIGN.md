# Обучаемая нормировка исторического интервала

План зафиксирован до новых метрик. Основа — опубликованный MIMO dual,
rank4/chunk8, две temporal heads, два слоя. Это консервативный comparator,
не доказанное превосходство над SISO или triple. Уже существуют отдельные
выходные коэффициенты heads и content-dependent внутренний шаг d Mamba3.
Новое изменение относится только к нормировке observed gap перед log1p.

Для механизма m ∈ {decay, scan}, головы h ∈ {0,1} и TRAIN reference
R0=838393 ms:

```
q_mh = log(4) tanh(alpha_mh)
R_mh = R0 exp(q_mh)
z_mh(g) = log1p(g / R_mh)
u_mh = SiLU(W1_m z_mh + b1_m)
raw_mh = W2_m[h,:] u_mh + b2_m[h]
s_mh = exp(log(2) tanh(raw_mh))
```

First Linear1→16 общий для heads; каждая голова использует свою прежнюю
строку последнего Linear16→2. Новых MLP и признаков нет. Scan совместно
управляет write и phase, функции общие для двух слоёв.

| Режим | Alpha | Новые параметры | Всего параметров |
|---|---|---:|---:|
| fixed | отсутствует | 0 | 715020 |
| shared_tau | одна на механизм | 2 | 715022 |
| head_tau | одна на механизм/head | 4 | 715024 |

Fixed оставляет исходные объекты TimeCalibrator и их вычислительный путь.
Learned wrappers сохраняют reference/first/last state keys без переименований;
единственные новые keys: times.calibrators.{decay,scan}.alpha. Alpha создаются
детерминированными нулями, не расходуя RNG; на старте R=R0. Общие веса,
буферы и training RNG сравниваются явно, без исключения всех calibrators.

Fixed контролирует сам факт обучения нормировки; shared_tau отделяет его от
добавления индивидуальных шкал heads. Основной контраст head_tau−shared_tau;
дополнительные shared_tau−fixed и head_tau−fixed. Исторический dual2026=.0633
служит reference, не порогом допуска и не новым независимым seed.

## Числа, маски и интерпретация

R ограничен [R0/4,4R0]. Это инженерное ограничение первого пилота, выбранное
заранее, не найденный по VALID диапазон интересов. Выходные scales по-прежнему
в [.5,2]; это другие bounds. Оба диапазона не меняются по итогам эксперимента.

Вход считается в float64: log(g)−log(R0)−q и logaddexp с нулём, затем
прежний dtype MLP. Нулевой gap даёт z=0 и нулевую производную z по alpha,
но active zero-gap сохраняет обученный scale. Первое событие и padding дают
строго1 и нулевой вклад в gradient alpha. NaN/Inf/negative gaps отклоняются.
Порядок, timestamp precision, active mask и отсутствие target timestamp прежние.

При z0=log1p(g/R0) и a=R0/R:
zR=log(1+a(exp(z0)−1)). Его вторая производная равна
a(1−a)exp(z0)/(1+a(exp(z0)−1))²: при a≠1 это в общем случае не affine
преобразование z0. При малых gaps zR≈a z0, при больших zR≈z0+log(a);
эффект может приближённо компенсироваться весами/bias MLP. Поэтому R не
объявляется идентифицируемым или новым в литературе. Это глобальный параметр
модели, не личный период пользователя. Различия R не доказывают специализацию
heads или короткие/длинные интересы; смотреть нужно и на scale(g).

## Проверки и фиксированный бюджет

CPU: исходная функция fixed; alpha0 с ненулевой MLP; tied shared/head outputs
и сумма derivatives; head-locality; float64 reference и gradcheck alpha/весов;
маски/zero/extreme gaps; bounds/finite gradients/roundtrip; RNG, counts и
config parity. Первый alpha-gradient при zero-init last может быть нулевым;
обучаемость проверяется на ненулевой MLP и после коротких synthetic steps.

Старое MIMO admission45/2342 и smoke job4358147 используются только как
evidence неизменённых kernels. Новый отдельный targeted registry проверяет
fixed parity, alpha0 parity и общие model gradients, calibrator derivatives,
tied derivatives, причинность (suffix items/timestamps, positional и
cross-user gradients), neutral masks и weights_only roundtrip. Состав
required leaves записан в study_plan.json до allocation. Structural
atol1e-6/rtol1e-5; исходная MIMO policy неизменна, bitwise записывается отдельно.
Legacy exact-zero FAIL остаются FAIL; gradients не обнуляются ради допуска.
Каждый check сохраняется до assertion, capture hooks удаляются до no_grad.

После PASS: synthetic smoke3 modes, batch2048/history50/kernel56, три Adam
steps для проверки alpha после zero-init; затем три свежих процесса
fixed→shared_tau→head_tau, seed2026. Все исходные scientific settings MIMO
dual сохранены, alpha обучаются тем же Adam. Reserved TEST split не используется;
TEST loader не создаётся. Максимум300 epochs, прежний VALID selection/last ties,
никаких warm-start, нового tuning или автоматического выбора backbone.

Один job, A100×1, rocky/proj_1833/type_e, CPU4, mem0, 6h, no-requeue.
Internal deadline с запасом600s; новый fit начинается только при ≥5400s.
Technical FAIL блокирует последующие fits, низкий VALID — нет. Неизменяемая
reservation до единственного sbatch. После Job ID работа останавливается.

Best diagnostics включают alpha, R/R0, Rms, saturation и scale(g) на сетке
[0,.01,.1,.25,.5,1,2,4,10,100] в штатном VALID callback, без RNG и дополнительной
оценки каталога. JSON/Markdown summary учитывают NOT_RUN/FAIL и отрицательные
контрасты; first27 только для полного окна0–26. Один seed не доказывает
устойчивость. Основная ветка, реестр93 строк, environment и данные неизменны.
