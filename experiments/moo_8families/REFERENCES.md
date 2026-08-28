# References

Основной taxonomy source:

- Weiyu Chen, Xiaoyuan Zhang, Baijiong Lin, Xi Lin, Han Zhao, Qingfu Zhang, James T. Kwok. `Gradient-Based Multi-Objective Deep Learning: Algorithms, Theories, Applications, and Beyond`. arXiv:2501.10945. URL: https://arxiv.org/abs/2501.10945

## 1. Loss Balancing - STCH

- Paper: Xi Lin, Xiaoyuan Zhang, Zhiyuan Yang, Fei Liu, Zhenkun Wang, Qingfu Zhang. `Smooth Tchebycheff Scalarization for Multi-Objective Optimization`. ICML 2024. URL: https://arxiv.org/abs/2402.19078
- Official code: https://github.com/Xi-L/STCH
- License: MIT.
- Implementation note: используется Smooth Tchebycheff scalarization, не linear weighted sum. Defaults взяты из official LibMTL config: `STCH_mu=1.0`, `STCH_warmup_epoch=4`.
- Audit note: benchmark preference - fixed `balanced` vector из `preferences.yaml`; reference point - zero vector in normalized loss space; nadir treatment - train-time log-normalized objective as in STCH-style implementation. Stable detached-max form subtracts `terms.detach().max()` before `logsumexp`; this changes only a detached constant and preserves the gradient of the original `mu * logsumexp(lambda_i f_i / mu)` objective. Synthetic smoke compares gradients with direct `torch.logsumexp`.

## 2. Gradient Weighting - FAMO

- Paper: Bo Liu, Yihao Feng, Peter Stone, Qiang Liu. `FAMO: Fast Adaptive Multitask Optimization`. NeurIPS 2023. URL: https://arxiv.org/abs/2306.03792
- Official code: https://github.com/Cranial-XIX/FAMO
- License: MIT.
- Implementation note: сохранены state vector `w`, Adam update для task weights, `gamma=0.01`, `w_lr=0.025`, detached normalizer `c` и detached loss progress signal.
- Audit note: conceptual line-by-line mapping to official formulation is: `w=zeros(requires_grad=True)`, effective weights `softmax(w)`, `D=losses-min_losses+eps`, `c=(z/D).sum().detach()`, weighted loss `(log(D)*z/c).sum()`, post-step recomputation of current losses, detached progress signal `log(prev-min)-log(curr-min)`, and Adam update on `w` with `gamma=0.01`, `w_lr=0.025`. Synthetic smoke checks equality with this formula.

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
- Audit note: preference adjustment follows official `adjustments(l,r)`: `rl=r*l`, `l_hat=rl/rl.sum()`, `mu=sum(l_hat*log(l_hat*m))`, `a=r*(log(l_hat*m)-mu)`, `mu_rl=rl.sum()*mu`. Balance branch solves `max alpha @ (C@a)` with simplex, non-negativity and `C@alpha >= C@a`; dominance branch uses the official dominance objective and constraints, then retries the official relaxed `C@alpha >= 0` LP if the primary dominance LP is infeasible. Final alpha is multiplied by `n_tasks`, matching official training code. Synthetic smoke includes 2-task regressions where balanced preference puts larger coefficient on the larger loss coordinate.

## 5. Finite Set Without Preference Vectors - GradHV

- Paper: Hao Wang, Andre Deutz, Thomas Baeck, Michael Emmerich. `Hypervolume Indicator Gradient Ascent Multi-Objective Optimization`. EMO 2017. DOI: 10.1007/978-3-319-54157-0_44
- Author implementation reference: https://github.com/wangronin/HIGA-MO
- License: no explicit license file observed.
- Implementation name: `HV-Gradient / GradHV-style`.
- Representative fidelity: `family-level adaptation`, not exact HIGA-MO/GradHV reproduction.
- Audit note: for minimization points, the implementation computes exact dominated hypervolume relative to a worse reference point by inclusion-exclusion over all solution subsets and optimizes `-HV`. Overlapping hyperrectangles are handled exactly by the inclusion-exclusion signs. Dominated solutions receive zero gradient in the synthetic dominated 2D case. The autograd gradient is the exact gradient of this implemented hypervolume objective and is checked by finite differences. This is a hypervolume-gradient representative of the family, but not a verified reproduction of the original HIGA-MO algorithmic update.

## 6. Infinite Set Hypernetwork - PHN

- Paper: Aviv Navon, Aviv Shamsian, Gal Chechik, Ethan Fetaya. `Learning the Pareto Front with Hypernetworks`. ICLR 2021. URL: https://arxiv.org/abs/2010.04104
- Official code: https://github.com/AvivNavon/pareto-hypernetworks
- License: MIT.
- Strict audit answer A: no, this is not an exact PHN reproduction. Official PHN generates network parameters from a preference ray; this branch does not generate the whole TiM4Rec, item embeddings or SSD blocks.
- Strict audit answer B: yes, this is only `PHN-inspired / PHN-adapter representative of the hypernetwork-based family`.
- Implementation note: полный PHN не подменяется молча. В этой ветке реализован `PHN-adapter`, который генерирует compact representation adapter parameters conditioned on preference and logs `representative_fidelity = family-level adaptation`.
- Preference sampling: official PHN trainer samples rays from `np.random.dirichlet([alpha] * K)` with default `alpha=0.2`; benchmark PHN-adapter uses the same Dirichlet alpha during training.

## 7. Infinite Set Preference-Conditioned Net - COSMOS

- Paper: Michael Ruchte, Josif Grabocka. `Scalable Pareto Front Approximation for Deep Multi-Objective Learning`. ICDM 2021. URL: https://arxiv.org/abs/2103.13392
- Official code: https://github.com/ruchtem/cosmos
- License: MIT.
- Implementation note: preference vector напрямую подается в recommender через encoder/fusion, это не hypernetwork.
- Preference sampling: official COSMOS samples Dirichlet alpha vectors in `COSMOSMethod.step`; benchmark COSMOS-style uses `Dirichlet(alpha=1.2)` and logs sampled preference diagnostics.

## 8. Infinite Set Model Combination - PaLoRA

- Paper: Nikolaos Dimitriadis, Pascal Frossard, Francois Fleuret. `Pareto Low-Rank Adapters: Efficient Multi-Task Learning with Preferences`. ICLR 2025. URL: https://arxiv.org/abs/2407.08056
- Official code: https://github.com/nik-dim/palora
- License: MIT.
- Implementation note: LoRA contribution is combined as preference-weighted sum over per-task low-rank adapters, `W + alpha/r * sum_t lambda_t B_t A_t`.
- Preference sampling: official PaLoRA includes Dirichlet and annealing-Dirichlet ray samplers; benchmark PaLoRA uses `Dirichlet(alpha=1.0)` during training and reserves the fixed grid for validation operating points.

## Common Pre-Run Audit Decisions

- PHN-adapter, COSMOS-style and PaLoRA train on continuous simplex samples from deterministic Dirichlet samplers with frozen seeds. Diagnostics record sampled mean, min/max, simplex sum error, coverage fraction and deterministic reproduction error.
- `preferences.yaml` is not a training cycle for continuous methods anymore. It is used for validation, Pareto plots and reproducible operating-point evaluation.
- All trainable families receive normalized train objectives unless a method explicitly requires otherwise. Raw and normalized losses are both persisted in run JSON.
