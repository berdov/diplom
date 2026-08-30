# Supervisor Summary

Короткая версия состояния benchmark для обсуждения с руководителем.

## 1. Published same-protocol benchmark

- SSD4Rec arXiv v1 Table 4 остаётся historical reproduction target проекта: `NDCG@10=0.0602`, `HR@10=0.1076`.
- SSD4Rec current v2 - отдельная published version: `NDCG@10=0.0593`, `HR@10=0.1075`.
- TiM4Rec Table 3 даёт лучший published KuaiRand row среди проверенных источников: `NDCG@10=0.0611`, `HR@10=0.1109`.

## 2. Our reproduction quality

- Our SSD4Rec locked test: `NDCG@10=0.0576`, `HR@10=0.1032`.
- Our TiM4Rec locked test: `NDCG@10=0.0598`, `HR@10=0.1053`.
- Оба результата ниже published rows, но находятся достаточно близко для дальнейшего controlled multitask/MOO анализа.

## 3. Multitask baseline

- Tuned MultitaskTiM4Rec validation control: `NDCG@10=0.0589`, `HR@10=0.1069`.
- Tuned MultitaskTiM4Rec locked test: `NDCG@10=0.0598`, `HR@10=0.1071`.

## 4. Eight-family screening

| Method | Validation NDCG@10 | HR@10 | Best epoch | Status |
| --- | ---: | ---: | ---: | --- |
| STCH | 0.0424 | 0.0749 | 80 | screened out for now |
| FAMO | 0.0412 | 0.0719 | 15 | screened out for now |
| PCGrad | 0.0444 | 0.0790 | 25 | tuning candidate |
| EPO | 0.0584 | 0.1078 | 15 | tuning candidate |
| HV-Gradient / GradHV-style | 0.0486 | 0.0874 | 50 | tuning candidate |
| PHN-adapter | 0.0423 | 0.0746 | 60 | screened out for now |
| COSMOS-style | 0.0453 | 0.0810 | 25 | tuning candidate |
| PaLoRA | 0.0422 | 0.0750 | 35 | screened out for now |

## 5. Current tuning candidates

- Tuning candidates: EPO, GradHV, COSMOS, PCGrad.
- Не тюним сейчас STCH, FAMO, PHN, PaLoRA.
- PCGrad tuning допускается только как current MOO-protocol tuning, а не как ретроактивная подмена current default historical run.

## 6. Next scientific step

- Запустить controlled validation-only Optuna tuning на A100-SXM4-80GB/type_e с примерно равным budget: 20-24 EPO trials и 24 trials для GradHV, COSMOS, PCGrad.
- TEST остаётся закрыт. Следующий этап после tuning - multi-seed validation confirmation максимум двух выбранных families.
