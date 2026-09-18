# Attempt002: алгебра и границы утверждений

Первый execution commit: `1bff0863e13c90161b6828c1905b2f5a0ab62a14`.
Его FAIL и исходный план сохранены в [attempt001](attempt_001/README.md).
Причина future-gradient `8.477159252340272e-11` пока **не установлена**.
Ни малость остатка, ни его наличие сами по себе не доказывают безопасность
или использование будущих событий.

## Угловой backward

Для градиента накопленной фазы g в одном блоке:

```text
sum_j g[j] - sum_{j<=i} g[j] + g[i] = sum_{j>=i} g[j]
```

Локальный `stable_angle.py` заменяет только total-minus-prefix на
`tl.cumsum(g, axis=0, reverse=True)`. Chunk sum для carry, маски valid/partial,
градиент состояния, dtype, tanh/sech2 approximations и forward неизменны.
Документация [Triton cumsum](https://triton-lang.org/main/python-api/generated/triton.language.cumsum.html)
определяет `reverse=True`; наличие параметра проверяется в установленном Triton
3.5.1 на login без запуска kernel.

## ADT backward

Пусть r = `dM_rev_vector`, v = `dM_vector`, c = `dM_scalar` ДО присваивания.
Для i от 0 до chunk_size-1 исходная формула равна:

```text
r[i] + sum_j r[j] + c + sum_{j<=i}(v[j]-r[j]) - v[i]
= sum_{j>=i}r[j] + sum_{j<i}v[j] + c.
```

В `stable_adt.py` первая сумма inclusive reverse, вторая exclusive prefix:
inclusive prefix сдвигается на одну позицию через gather; индекс0 получает0.
Не применяется вычитание текущего v из его prefix. На последней позиции r
включает себя, v исключает себя; на первой prefix(v)=0. CPU tests проверяют
эти индексы, signed значения, partial blocks и ненулевой межблочный carry.
Маски, carry `d_ssm_states_acc`, все остальные производные и dtype сохранены.
Это точное тождество над вещественными числами, не обещание bitwise backward.

Скопированы только backward kernel и его wrapper из каждого pinned файла;
не весь Mamba package. Pin, copyright и Apache-2.0 сохранены. Forward не скопирован
и не меняется. Installed upstream и глобальные функции не модифицируются.

## Фиксированный диагностический план

Original upstream, angle-only, ADT-only и combined stable-scan сравниваются
на одних fixtures L17/P7, L65/P31, L97/P65 и loss multipliers1/16.
Loss ровно прежний scalar_loss, только prefix outputs, без final-state loss.
Градиенты измеряются по позиционным embedding outputs, не общей embedding table.
Отдельно сохраняются kernel-input и промежуточные backward производные,
изолированный angle backward и direct reverse-cumsum reference.

Official base и frozen separate служат контролями на тех же весах и входах.
Не объявляется локализация, если исходный остаток не воспроизведён или
промежуточные производные не показывают источник. Источников может быть несколько.
Строгий exact-zero check не ослабляется. Clipping, detach, отсечение по prefix
в backward и special-case длин отсутствуют.

Все A-H/I и initialization повторяются для candidate; дополнительно сравниваются
original/stable outputs (exact) и все gradients (старый structural tolerance).
Reference сначала калибруется official baseline. Его широкие допуски не меняются.
Оригинальный exact-zero FAIL отражается отдельно от candidate verdict.
Default backend остаётся upstream до ручного разбора GPU evidence; автоматического
переключения или научного обучения нет. При принятии правки base/dual/triple
должны использовать одинаковый backend.

## MIMO

Новый профиль rank4/chunk8 меняет только вычислительный chunk. Counts фиксированы:
714888/715020/715086. Предыдущий rank4/chunk16 FAIL не переписывается.
Pinned public API допускает chunk>=8 и рекомендует уменьшать chunk при smem
over-allocation. Сначала настоящий official forward/backward; отдельно фиксируются
стадии, capability, доступные limits и traceback. Per-block shared memory не HBM.
Если metadata native launch не доступны через wrapper, это явно указано, не угадано.
SISO и MIMO имеют независимые subprocesses. Полный MIMO suite возможен только
после native PASS. Никаких переборов chunk/rank, изменений TileLang или второго job.
