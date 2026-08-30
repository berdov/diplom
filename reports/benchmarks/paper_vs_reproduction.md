# Paper vs наши canonical reproductions

Эта таблица использует только canonical TEST reproductions из repository artifacts. MOO validation-only runs сюда не входят.

Источник нашего SSD4Rec: [experiments/ssd4rec_baseline/runs/ssd4rec_001.json](../../experiments/ssd4rec_baseline/runs/ssd4rec_001.json).  
Источник нашего TiM4Rec: [experiments/tim4rec_baseline/runs/tim4rec_001.json](../../experiments/tim4rec_baseline/runs/tim4rec_001.json).

## SSD4Rec

| Metric | Paper | Ours | Absolute delta | Relative delta |
| --- | --- | --- | --- | --- |
| HR@10 | 0.1076 | 0.1032 | -0.0044 | -4.09% |
| HR@20 | 0.1704 | 0.1683 | -0.0021 | -1.23% |
| NDCG@10 | 0.0602 | 0.0576 | -0.0026 | -4.32% |
| NDCG@20 | 0.0759 | 0.0739 | -0.0020 | -2.64% |
| MRR@10 | 0.0460 | — | — | — |
| MRR@20 | 0.0503 | — | — | — |

## TiM4Rec

| Metric | Paper | Ours | Absolute delta | Relative delta |
| --- | --- | --- | --- | --- |
| HR@10 | 0.1109 | 0.1053 | -0.0056 | -5.05% |
| HR@20 | 0.1774 | 0.1696 | -0.0078 | -4.40% |
| HR@50 | 0.3202 | 0.3031 | -0.0171 | -5.34% |
| NDCG@10 | 0.0611 | 0.0598 | -0.0013 | -2.13% |
| NDCG@20 | 0.0779 | 0.0759 | -0.0020 | -2.57% |
| NDCG@50 | 0.1060 | 0.1022 | -0.0038 | -3.58% |
| MRR@10 | 0.0463 | — | — | — |
| MRR@20 | 0.0508 | — | — | — |
| MRR@50 | 0.0552 | — | — | — |

MRR у наших reproduction artifacts не измерялся, поэтому MRR deltas не вычисляются.
