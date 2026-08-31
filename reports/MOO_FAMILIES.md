# MOO Family Benchmark

KuaiRand Protocol B, validation-only. TEST не использовался.

| Family | Representative | HR@10 | HR@20 | HR@50 | NDCG@10 | NDCG@20 | NDCG@50 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Loss Balancing | [STCH](https://proceedings.mlr.press/v235/lin24y.html) | .0749 | .1163 | .2082 | .0424 | .0528 | .0709 |
| Gradient Weighting | [FAMO](https://papers.neurips.cc/paper_files/paper/2023/hash/b2fe1ee8d936ac08dd26f2ff58986c8f-Abstract-Conference.html) | .0719 | .1102 | .1935 | .0412 | .0508 | .0672 |
| Gradient Manipulation | [PCGrad](https://proceedings.neurips.cc/paper_files/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html) | .0790 | .1259 | .2253 | .0444 | .0562 | .0757 |
| Finite set + preference vectors | [EPO](https://proceedings.mlr.press/v119/mahapatra20a.html) | .1078 | .1767 | .3171 | .0584 | .0756 | .1033 |
| Finite set without preference vectors | [GradHV-style](https://link.springer.com/chapter/10.1007/978-3-319-54157-0_44) | .0874 | .1382 | .2440 | .0486 | .0613 | .0820 |
| Hypernetwork-based infinite set | [PHN](https://openreview.net/forum?id=NjF772F4ZZR) | .0746 | .1155 | .2027 | .0423 | .0526 | .0698 |
| Preference-conditioned network | [COSMOS](https://ieeexplore.ieee.org/document/9679014/) | .0810 | .1257 | .2252 | .0453 | .0565 | .0761 |
| Model combination | [PaLoRA](https://proceedings.iclr.cc/paper_files/paper/2025/hash/384c7fe3f5c377efb5f9d1282ae98b81-Abstract-Conference.html) | .0750 | .1159 | .2080 | .0422 | .0525 | .0706 |

*GradHV-style is a family-level adaptation rather than an exact reproduction of a single published implementation.*
