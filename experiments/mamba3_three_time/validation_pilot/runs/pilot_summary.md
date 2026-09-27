# SISO Dual / Triple Pilot

KuaiRand Protocol B, VALID full-ranking. TEST NOT RUN. Один seed, exploratory; статистическая значимость не установлена.

| mode | parameters | VALID NDCG@10 | HR@10 | first27 best | best_epoch (0-based) | actual_epochs | TRAIN/VALID seconds | status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| dual | 610572 | 0.0615 | 0.1142 | 0.0615 | 15 | 27 | 593.2322801131522 / 11.176218414097093 | PASS |
| triple | 610638 | 0.0623 | 0.1148 | 0.0623 | 15 | 27 | 600.2216197031667 / 10.499554241949227 | PASS |

triple - dual NDCG@10: {"absolute": 0.0008000000000000021, "relative": 0.013008130081300848}

Исторический dual seed2026: VALID NDCG@10 = 0.0633, только контекст; это не дополнительный независимый seed.

Старые exact-zero FAIL остаются FAIL. Допуск ограничен текущим SISO-пилотом; MIMO не авторизован.
