# Adaptive sanity comparison 001

Сгенерировано: `2026-08-25T09:34:12.181519+00:00`.

Все adaptive sanity runs обучались 5 эпох на полном train split и оценивались только на full-ranking validation. TEST не использовался.

Reference rows ниже являются уже существующими full/reference runs, поэтому они не равны по бюджету 5-epoch sanity.

## Main comparison

| Method | Run type | Best epoch | HR@10 | HR@20 | HR@50 | NDCG@10 | NDCG@20 | NDCG@50 | epoch time | peak VRAM |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TiM4Rec reference validation | full reference | 12 | 0.1086 | 0.1740 | 0.3126 | 0.0593 | 0.0757 | 0.1030 | 26.5 | 1.95 GB |
| Fixed Multitask reference validation | full reference | 14 | 0.1061 | 0.1733 | 0.3115 | 0.0580 | 0.0749 | 0.1022 | 28.6 | 1.96 GB |
| Tuned fixed reference validation | full tuned reference | 16 | 0.1069 | 0.1769 | 0.3195 | 0.0589 | 0.0765 | 0.1046 | 26.2 |  |
| PCGrad sanity | 5-epoch sanity | 5 | 0.1036 | 0.1656 | 0.3007 | 0.0568 | 0.0723 | 0.0990 | 80.3 | 2.47 GB |
| MetaBalance sanity | 5-epoch sanity | 5 | 0.0951 | 0.1511 | 0.2665 | 0.0518 | 0.0659 | 0.0886 | 94.5 | 5.27 GB |

## Rank-aux diagnostic cosines

| Method | Diagnostic epoch | Rank-vs-click cosine | Rank-vs-long cosine | Rank-vs-like cosine | Rank-vs-profile cosine | rank-aux conflict fraction |
| --- | --- | --- | --- | --- | --- | --- |
| PCGrad | 1 | 0.002822 | 0.005418 | -0.002681 | 0.013976 | 0.2500 |
| PCGrad | 3 | 0.070678 | 0.124876 | -0.013719 | 0.060680 | 0.2500 |
| PCGrad | 5 | 0.004221 | 0.034950 | -0.002634 | -0.032694 | 0.5000 |
| MetaBalance | 1 | 0.002822 | 0.005418 | -0.002681 | 0.013976 | 0.2500 |
| MetaBalance | 3 | 0.007276 | 0.039199 | 0.010419 | 0.022221 | 0.0000 |
| MetaBalance | 5 | 0.045849 | 0.031129 | 0.022746 | -0.010778 | 0.2500 |

## Auxiliary-auxiliary conflicts

| Method | aux negatives before | aux pairs before | aux fraction before | aux negatives after | aux pairs after | aux fraction after | rank-aux negatives before | rank-aux pairs before | rank-aux fraction before |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PCGrad | 3801 | 15570 | 0.2441 | 3745 | 15570 | 0.2405 | 2884 | 10380 | 0.2778 |
| MetaBalance | 3637 | 15570 | 0.2336 | 2666 | 15570 | 0.1712 | 2471 | 10380 | 0.2381 |

## is_like diagnostics

| Method | Diagnostic epoch | like raw loss | like effective contribution | like shared grad norm | like cosine with ranking |
| --- | --- | --- | --- | --- | --- |
| PCGrad | 1 | 0.822490 | 0.352060 | 0.128600 | -0.002681 |
| PCGrad | 3 | 0.468092 | 0.200363 | 0.111358 | -0.013719 |
| PCGrad | 5 | 0.329796 | 0.141167 | 0.112511 | -0.002634 |
| MetaBalance | 1 | 0.822490 | 0.352060 | 0.128600 | -0.002681 |
| MetaBalance | 3 | 0.352206 | 0.150759 | 0.163360 | 0.010419 |
| MetaBalance | 5 | 0.316262 | 0.135373 | 0.294754 | 0.022746 |
