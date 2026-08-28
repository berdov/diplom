# References

Основной taxonomy source:

- Weiyu Chen, Xiaoyuan Zhang, Baijiong Lin, Xi Lin, Han Zhao, Qingfu Zhang, James T. Kwok. `Gradient-Based Multi-Objective Deep Learning: Algorithms, Theories, Applications, and Beyond`. arXiv:2501.10945. URL: https://arxiv.org/abs/2501.10945

## 1. Loss Balancing - STCH

- Paper: Xi Lin, Xiaoyuan Zhang, Zhiyuan Yang, Fei Liu, Zhenkun Wang, Qingfu Zhang. `Smooth Tchebycheff Scalarization for Multi-Objective Optimization`. ICML 2024. URL: https://arxiv.org/abs/2402.19078
- Official code: https://github.com/Xi-L/STCH
- License: MIT.
- Implementation note: используется Smooth Tchebycheff scalarization, не linear weighted sum. Defaults взяты из official LibMTL config: `STCH_mu=1.0`, `STCH_warmup_epoch=4`.

## 2. Gradient Weighting - FAMO

- Paper: Bo Liu, Yihao Feng, Peter Stone, Qiang Liu. `FAMO: Fast Adaptive Multitask Optimization`. NeurIPS 2023. URL: https://arxiv.org/abs/2306.03792
- Official code: https://github.com/Cranial-XIX/FAMO
- License: MIT.
- Implementation note: сохранены state vector `w`, Adam update для task weights, `gamma=0.01`, `w_lr=0.025`, detached normalizer `c` и detached loss progress signal.

## 3. Gradient Manipulation - PCGrad

- Paper: Tianhe Yu, Saurabh Kumar, Abhishek Gupta, Sergey Levine, Karol Hausman, Chelsea Finn. `Gradient Surgery for Multi-Task Learning`. NeurIPS 2020. URL: https://arxiv.org/abs/2001.06782
- Official code: https://github.com/tianheyu927/PCGrad
- License: no license file observed in the official repository at review time.
- Implementation note: новый run не запускается автоматически. Используется исторический `pcgrad_001`, если JSON подтверждает validation-only protocol и `test_evaluation_count=0`.

## 4. Finite Set With Preference Vectors - EPO

- Paper: Debabrata Mahapatra, Vaibhav Rajan. `Multi-Task Learning with User Preferences: Gradient Descent with Controlled Ascent in Pareto Optimization`. ICML 2020. URL: https://proceedings.mlr.press/v119/mahapatra20a.html
- Extended EPO Search reference: Debabrata Mahapatra, Vaibhav Rajan. `Exact Pareto Optimal Search for Multi-Task Learning and Multi-Criteria Decision-Making`. URL: https://arxiv.org/abs/2108.00597
- Official code: https://github.com/dbmptr/EPOSearch
- License: MIT.
- Implementation note: LP constraints follow the official `EPO_LP` formulation. Для 5 tasks используется small exact vertex-enumeration LP solver, чтобы не зависеть от `cvxpy/cvxopt` в cluster env.

## 5. Finite Set Without Preference Vectors - GradHV

- Paper: Hao Wang, Andre Deutz, Thomas Baeck, Michael Emmerich. `Hypervolume Indicator Gradient Ascent Multi-Objective Optimization`. EMO 2017. DOI: 10.1007/978-3-319-54157-0_44
- Author implementation reference: https://github.com/wangronin/HIGA-MO
- License: no explicit license file observed.
- Implementation note: для малого числа решений и 5 objectives используется exact dominated-hypervolume inclusion-exclusion objective; reference point фиксируется по train-only diagnostics.

## 6. Infinite Set Hypernetwork - PHN

- Paper: Aviv Navon, Aviv Shamsian, Gal Chechik, Ethan Fetaya. `Learning the Pareto Front with Hypernetworks`. ICLR 2021. URL: https://arxiv.org/abs/2010.04104
- Official code: https://github.com/AvivNavon/pareto-hypernetworks
- License: MIT.
- Implementation note: полный PHN не подменяется молча. В этой ветке реализован `PHN-adapter`, который генерирует compact representation adapter parameters conditioned on preference.

## 7. Infinite Set Preference-Conditioned Net - COSMOS

- Paper: Michael Ruchte, Josif Grabocka. `Scalable Pareto Front Approximation for Deep Multi-Objective Learning`. ICDM 2021. URL: https://arxiv.org/abs/2103.13392
- Official code: https://github.com/ruchtem/cosmos
- License: MIT.
- Implementation note: preference vector напрямую подается в recommender через encoder/fusion, это не hypernetwork.

## 8. Infinite Set Model Combination - PaLoRA

- Paper: Nikolaos Dimitriadis, Pascal Frossard, Francois Fleuret. `Pareto Low-Rank Adapters: Efficient Multi-Task Learning with Preferences`. ICLR 2025. URL: https://arxiv.org/abs/2407.08056
- Official code: https://github.com/nik-dim/palora
- License: MIT.
- Implementation note: LoRA contribution is combined as preference-weighted sum over per-task low-rank adapters, `W + alpha/r * sum_t lambda_t B_t A_t`.
