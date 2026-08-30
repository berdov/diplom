# Paper Results / опубликованный benchmark: KuaiRand Protocol B

Эта страница содержит только опубликованные paper rows. Наши reproductions, validation-only MOO screening и tuning должны жить отдельно и иметь явные поля `source`, `stage` и `split`.

Primary key для machine-readable таблицы: `Benchmark source` + `Method`. Поэтому `SASRec / SSD4Rec Table 4` и `SASRec / TiM4Rec Table 3` являются разными published rows, а не duplicate.

## Canonical reproduction target

Для исторической reproduction target проекта используется `SSD4Rec arXiv v1 Table 4`: `SSD4Rec NDCG@10=0.0602`, `NDCG@20=0.0759`, `HR@10=0.1076`, `HR@20=0.1704`. Текущий arXiv v2 содержит обновлённую published row, но v1 не считается ошибочной.

## Compact published rows

| Benchmark source | Method | NDCG@10 | NDCG@20 | NDCG@50 | MRR@10 | MRR@20 | MRR@50 | HR@10 | HR@20 | HR@50 | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SSD4Rec arXiv v1 Table 4 | Caser | 0.0545 | 0.0692 |  | 0.0414 | 0.0453 |  | 0.0982 | 0.1571 |  | Historical reproduction target for this project. Baseline method original paper is not the KuaiRand result source. |
| SSD4Rec arXiv v1 Table 4 | GRU4Rec | 0.0563 | 0.0722 |  | 0.0426 | 0.0470 |  | 0.1017 | 0.1653 |  | Historical reproduction target for this project. Baseline method original paper is not the KuaiRand result source. |
| SSD4Rec arXiv v1 Table 4 | BERT4Rec | 0.0534 | 0.0683 |  | 0.0404 | 0.0444 |  | 0.0968 | 0.1563 |  | Historical reproduction target for this project. Baseline method original paper is not the KuaiRand result source. |
| SSD4Rec arXiv v1 Table 4 | SASRec | 0.0567 | 0.0733 |  | 0.0426 | 0.0471 |  | 0.1040 | 0.1705 |  | Historical reproduction target for this project. Baseline method original paper is not the KuaiRand result source. |
| SSD4Rec arXiv v1 Table 4 | Mamba4Rec | 0.0558 | 0.0710 |  | 0.0427 | 0.0468 |  | 0.0994 | 0.1601 |  | Historical reproduction target for this project. Baseline method original paper is not the KuaiRand result source. |
| SSD4Rec arXiv v1 Table 4 | SSD4Rec | 0.0602 | 0.0759 |  | 0.0460 | 0.0503 |  | 0.1076 | 0.1704 |  | Historical reproduction target for this project. |
| SSD4Rec current arXiv v2 Table 4 | Caser | 0.0545 | 0.0692 |  | 0.0414 | 0.0453 |  | 0.0982 | 0.1571 |  | Updated SSD4Rec paper version. Kept as published version, not treated as an error versus v1. |
| SSD4Rec current arXiv v2 Table 4 | GRU4Rec | 0.0563 | 0.0722 |  | 0.0426 | 0.0470 |  | 0.1017 | 0.1653 |  | Updated SSD4Rec paper version. Kept as published version, not treated as an error versus v1. |
| SSD4Rec current arXiv v2 Table 4 | BERT4Rec | 0.0534 | 0.0683 |  | 0.0404 | 0.0444 |  | 0.0968 | 0.1563 |  | Updated SSD4Rec paper version. Kept as published version, not treated as an error versus v1. |
| SSD4Rec current arXiv v2 Table 4 | SASRec | 0.0567 | 0.0733 |  | 0.0426 | 0.0471 |  | 0.1040 | 0.1705 |  | Updated SSD4Rec paper version. Kept as published version, not treated as an error versus v1. |
| SSD4Rec current arXiv v2 Table 4 | Mamba4Rec | 0.0558 | 0.0710 |  | 0.0427 | 0.0468 |  | 0.0994 | 0.1601 |  | Updated SSD4Rec paper version. Kept as published version, not treated as an error versus v1. |
| SSD4Rec current arXiv v2 Table 4 | SIGMA | 0.0572 | 0.0727 |  | 0.0433 | 0.0475 |  | 0.1036 | 0.1655 |  | Updated SSD4Rec paper version. Kept as published version, not treated as an error versus v1. |
| SSD4Rec current arXiv v2 Table 4 | SSD4Rec | 0.0593 | 0.0757 |  | 0.0448 | 0.0493 |  | 0.1075 | 0.1731 |  | Updated SSD4Rec paper version. Kept as published version, not treated as an error versus v1. |
| TiM4Rec arXiv v3 Table 3 | Caser | 0.0395 | 0.0531 | 0.0770 | 0.0273 | 0.0310 | 0.0347 | 0.0801 | 0.1344 | 0.2561 | KuaiRand row is from TiM4Rec Table 3, not from the original method paper. |
| TiM4Rec arXiv v3 Table 3 | GRU4Rec | 0.0564 | 0.0724 | 0.0911 | 0.0428 | 0.0471 | 0.0513 | 0.1020 | 0.1659 | 0.3017 | KuaiRand row is from TiM4Rec Table 3, not from the original method paper. |
| TiM4Rec arXiv v3 Table 3 | SASRec | 0.0584 | 0.0747 | 0.1016 | 0.0443 | 0.0487 | 0.0529 | 0.1055 | 0.1704 | 0.3074 | KuaiRand row is from TiM4Rec Table 3, not from the original method paper. |
| TiM4Rec arXiv v3 Table 3 | BERT4Rec | 0.0510 | 0.0660 | 0.0923 | 0.0382 | 0.0422 | 0.0464 | 0.0938 | 0.1537 | 0.2873 | KuaiRand row is from TiM4Rec Table 3, not from the original method paper. |
| TiM4Rec arXiv v3 Table 3 | TiSASRec | 0.0590 | 0.0753 | 0.1019 | 0.0450 | 0.0494 | 0.0536 | 0.1057 | 0.1710 | 0.3060 | KuaiRand row is from TiM4Rec Table 3, not from the original method paper. |
| TiM4Rec arXiv v3 Table 3 | LRURec | 0.0570 | 0.0727 | 0.1005 | 0.0431 | 0.0473 | 0.0517 | 0.1036 | 0.1663 | 0.3078 | KuaiRand row is from TiM4Rec Table 3, not from the original method paper. |
| TiM4Rec arXiv v3 Table 3 | Mamba4Rec | 0.0608 | 0.0777 | 0.1050 | 0.0461 | 0.0508 | 0.0552 | 0.1094 | 0.1768 | 0.3154 | KuaiRand row is from TiM4Rec Table 3, not from the original method paper. |
| TiM4Rec arXiv v3 Table 3 | SSD4Rec* | 0.0588 | 0.0754 | 0.1024 | 0.0449 | 0.0494 | 0.0536 | 0.1055 | 0.1717 | 0.3088 | SSD4Rec* is the TiM4Rec authors own replicated variant, not the original SSD4Rec published row. |
| TiM4Rec arXiv v3 Table 3 | TiM4Rec | 0.0611 | 0.0779 | 0.1060 | 0.0463 | 0.0508 | 0.0552 | 0.1109 | 0.1774 | 0.3202 | TiM4Rec Table 3 same-pipeline published row. |

## TiM4Rec Table 3: compact same-source table

`R@K` в TiM4Rec Table 3 интерпретируется как recall; для leave-one-positive evaluation это сопоставимо с HR@K в наших таблицах. `SSD4Rec*` - это replicated variant authors of TiM4Rec, а не оригинальная SSD4Rec published row.

| Benchmark source | Method | NDCG@10 | NDCG@20 | NDCG@50 | MRR@10 | MRR@20 | MRR@50 | HR@10 | HR@20 | HR@50 | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TiM4Rec arXiv v3 Table 3 | Caser | 0.0395 | 0.0531 | 0.0770 | 0.0273 | 0.0310 | 0.0347 | 0.0801 | 0.1344 | 0.2561 | KuaiRand row is from TiM4Rec Table 3, not from the original method paper. |
| TiM4Rec arXiv v3 Table 3 | GRU4Rec | 0.0564 | 0.0724 | 0.0911 | 0.0428 | 0.0471 | 0.0513 | 0.1020 | 0.1659 | 0.3017 | KuaiRand row is from TiM4Rec Table 3, not from the original method paper. |
| TiM4Rec arXiv v3 Table 3 | SASRec | 0.0584 | 0.0747 | 0.1016 | 0.0443 | 0.0487 | 0.0529 | 0.1055 | 0.1704 | 0.3074 | KuaiRand row is from TiM4Rec Table 3, not from the original method paper. |
| TiM4Rec arXiv v3 Table 3 | BERT4Rec | 0.0510 | 0.0660 | 0.0923 | 0.0382 | 0.0422 | 0.0464 | 0.0938 | 0.1537 | 0.2873 | KuaiRand row is from TiM4Rec Table 3, not from the original method paper. |
| TiM4Rec arXiv v3 Table 3 | TiSASRec | 0.0590 | 0.0753 | 0.1019 | 0.0450 | 0.0494 | 0.0536 | 0.1057 | 0.1710 | 0.3060 | KuaiRand row is from TiM4Rec Table 3, not from the original method paper. |
| TiM4Rec arXiv v3 Table 3 | LRURec | 0.0570 | 0.0727 | 0.1005 | 0.0431 | 0.0473 | 0.0517 | 0.1036 | 0.1663 | 0.3078 | KuaiRand row is from TiM4Rec Table 3, not from the original method paper. |
| TiM4Rec arXiv v3 Table 3 | Mamba4Rec | 0.0608 | 0.0777 | 0.1050 | 0.0461 | 0.0508 | 0.0552 | 0.1094 | 0.1768 | 0.3154 | KuaiRand row is from TiM4Rec Table 3, not from the original method paper. |
| TiM4Rec arXiv v3 Table 3 | SSD4Rec* | 0.0588 | 0.0754 | 0.1024 | 0.0449 | 0.0494 | 0.0536 | 0.1055 | 0.1717 | 0.3088 | SSD4Rec* is the TiM4Rec authors own replicated variant, not the original SSD4Rec published row. |
| TiM4Rec arXiv v3 Table 3 | TiM4Rec | 0.0611 | 0.0779 | 0.1060 | 0.0463 | 0.0508 | 0.0552 | 0.1109 | 0.1774 | 0.3202 | TiM4Rec Table 3 same-pipeline published row. |

## Source checks

- SSD4Rec v1 Table 4 подтверждает KuaiRand rows для Caser, GRU4Rec, BERT4Rec, SASRec, Mamba4Rec и SSD4Rec: https://arxiv.org/html/2409.01192v1.
- SSD4Rec current v2 Table 4 подтверждает updated SSD4Rec row и добавляет SIGMA baseline: https://arxiv.org/pdf/2409.01192.
- TiM4Rec Table 3 подтверждает Caser, GRU4Rec, SASRec, BERT4Rec, TiSASRec, LRURec, Mamba4Rec, SSD4Rec* и TiM4Rec на KuaiRand: https://arxiv.org/html/2409.16182v3.
- Machine-readable provenance каждого числа лежит в [paper_results_provenance.csv](paper_results_provenance.csv).
