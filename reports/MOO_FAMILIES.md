# Семейства MOO в Stage 1

Stage 1 — первичный отбор представителей и адаптаций восьми семейств многокритериальной оптимизации (MOO) на KuaiRand в рамках протокола B. Все строки ниже относятся только к валидационной выборке; TEST не использовался.

Таблица не утверждает, что выбранный representative является лучшим методом внутри своего семейства. Это текущий экспериментальный выбор для дипломного проекта. Дополнительный literature audit по внутри-семейному выбору остаётся открытой задачей.

| Семейство | Representative / adaptation | Run | HR@10 | HR@20 | HR@50 | NDCG@10 | NDCG@20 | NDCG@50 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Балансировка losses | [STCH](https://proceedings.mlr.press/v235/lin24y.html) | `stch_convergence_001` | 0.0749 | 0.1163 | 0.2082 | 0.0424 | 0.0528 | 0.0709 |
| Взвешивание градиентов | [FAMO](https://papers.neurips.cc/paper_files/paper/2023/hash/b2fe1ee8d936ac08dd26f2ff58986c8f-Abstract-Conference.html) | `famo_convergence_001` | 0.0719 | 0.1102 | 0.1935 | 0.0412 | 0.0508 | 0.0672 |
| Коррекция конфликтующих градиентов | [PCGrad](https://proceedings.neurips.cc/paper_files/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html) | `pcgrad_convergence_001` | 0.0790 | 0.1259 | 0.2253 | 0.0444 | 0.0562 | 0.0757 |
| Конечный набор решений с preference vectors | [EPO](https://proceedings.mlr.press/v119/mahapatra20a.html) | `epo_convergence_001` | 0.1078 | 0.1767 | 0.3171 | 0.0584 | 0.0756 | 0.1033 |
| Конечный набор решений без явных preference vectors | [GradHV-style](https://link.springer.com/chapter/10.1007/978-3-319-54157-0_44) | `gradhv_convergence_001` | 0.0874 | 0.1382 | 0.2440 | 0.0486 | 0.0613 | 0.0820 |
| Hypernetwork-based приближение множества решений | [PHN](https://openreview.net/forum?id=NjF772F4ZZR) adapter | `phn_convergence_001` | 0.0746 | 0.1155 | 0.2027 | 0.0423 | 0.0526 | 0.0698 |
| Preference-conditioned сеть | [COSMOS](https://ieeexplore.ieee.org/document/9679014/) style | `cosmos_convergence_001` | 0.0810 | 0.1257 | 0.2252 | 0.0453 | 0.0565 | 0.0761 |
| Комбинация моделей | [PaLoRA](https://proceedings.iclr.cc/paper_files/paper/2025/hash/384c7fe3f5c377efb5f9d1282ae98b81-Abstract-Conference.html) | `palora_convergence_001` | 0.0750 | 0.1159 | 0.2080 | 0.0422 | 0.0525 | 0.0706 |

## Как читать эту таблицу

EPO показал лучший validation NDCG@10 в Stage 1: `0.0584`. Поэтому EPO, GradHV, COSMOS и PCGrad были переданы в Stage 2 для настройки гиперпараметров. Это рабочее решение по итогам первичного отбора, а не доказательство, что остальные семейства не могут дать лучший результат при другой реализации или большем бюджете.

GradHV-style и COSMOS-style обозначены как адаптации под текущую MTL-постановку. PHN-adapter также является адаптацией: он проверяет идею hypernetwork-based представления, но не претендует на точное воспроизведение всех деталей конкретной опубликованной реализации.
