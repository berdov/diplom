# Frozen stage confirmation

Authorized: merge canonical docs, close selection fairness, confirm three target sets on three common seeds. No tuning or TEST. Main fast-forwarded to 93053d86eb3b7516ecf748fe060baa179d4e678b.

MosT: full stored trajectory replay shows identical choices for max VALID NDCG@10 at every check, best/stop epoch 10/25; no rerun required. Original JSON remains immutable; derivative audit explicitly labels the oracle selection.

MTL: common seeds **2026, 2027, 2028**, subsets **primary-only, like+profile_enter, click+like+profile_enter**. Reuse the three completed 2026 screening records; submit exactly six additional full runs. No subset-specific tuning. Original 80-epoch cap, patience 5, validation each epoch, batch sizes 2048/4096, fixed parameters, full ranking and uniform mean auxiliary losses are unchanged.

The seven files in experiments/target_combination_analysis are byte-identical to screening code SHA 599bcdb6e50dceea73233169b8834eaceef083a8. The wrapper calls its original execute function; only training.seed changes. 59 frozen scientific source/config hashes are checked, including imported Stage 3 code. Reuse the completed all-four smoke (4315279) and prior exact dataset audit; original execute rechecks TRAIN/VALID data hashes on each run. No new smoke/training algorithm is introduced. New runtime paths and run IDs prevent overwrites.

All 12 canonical HR/Recall/NDCG @5/10/20/50 metrics are aggregated as mean and sample std (ddof=1). Report paired NDCG@10 differences at the same seeds. The descriptive freeze decision is declared before new results: require improvement over primary-only on all three seeds; among eligible subsets select higher mean NDCG@10, prefer the smaller pair if pair/triple mean difference ≤0.0001 (screening near-tie threshold). If no subset meets this rule, do not claim stable auxiliary improvement. This is not a significance test; seeds are only training randomness, not independent data splits.

One six-task Slurm array, rocky / A100 / 4 CPU / mem=0 / 3h / no-requeue, same resource class as screening. CPU summary runs afterany array; missing/failed cells remain explicit, never averaged as zeros. No automatic reruns. Historical EPO+MoE M0/M2/M4/M8 are closed as technical failures without scientific results, never resubmitted.
