# MOO Family Benchmark

KuaiRand Protocol B, validation-only. TEST не использовался.

| Family | Representative | HR@10 | HR@20 | HR@50 | NDCG@10 | NDCG@20 | NDCG@50 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Loss Balancing | STCH | .0749 | .1163 | .2082 | .0424 | .0528 | .0709 |
| Gradient Weighting | FAMO | .0719 | .1102 | .1935 | .0412 | .0508 | .0672 |
| Gradient Manipulation | PCGrad | .0790 | .1259 | .2253 | .0444 | .0562 | .0757 |
| Finite set + preference vectors | EPO | .1078 | .1767 | .3171 | .0584 | .0756 | .1033 |
| Finite set without preference vectors | GradHV | .0874 | .1382 | .2440 | .0486 | .0613 | .0820 |
| Hypernetwork-based infinite set | PHN | .0746 | .1155 | .2027 | .0423 | .0526 | .0698 |
| Preference-conditioned network | COSMOS | .0810 | .1257 | .2252 | .0453 | .0565 | .0761 |
| Model combination | PaLoRA | .0750 | .1159 | .2080 | .0422 | .0525 | .0706 |
