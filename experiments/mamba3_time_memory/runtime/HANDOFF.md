# Time-addressed memory pilot: attempt002, job4373393

Phase: SUBMITTED_PENDING. **2026-10-03 18:58:00 MSK:** PENDING (Priority), node UNKNOWN, elapsed 00:00:00. Scientific fits started UNKNOWN, completed UNKNOWN/3. GPU gate UNKNOWN; smoke UNKNOWN. Next explicit poll no earlier than 2026-10-03 19:08:00 MSK (2026-10-03T16:08:00.722692+00:00). No background process.
[Exact snapshot](status_4373393.json), [state](state.json).

## Canonical execution

Local /Users/berdov/diplom; cluster /home/daryumin/iberdov/diplom, SSH hse-karizma.
Branch exp/mamba3-time-addressed-memory; main7ba5e45ed110914b354c89d4377836910b55408c.
Execution **53bcc76752d0d7fd06c8ba0866709a64cd978165**;
source **ad4565d171b78c83831ef84c139a4912af2022a04fdb39fdfd5b4a858461f5e6**,412files,
source_manifest_002.json. Cluster tracked clean on this exact pushed execution; do not change
source while its job uses it. Other active jobs have unrelated WorkDir/Command and were untouched.

Original task attachment (read on resumption):
/Users/berdov/.codex/attachments/4de68a7a-1c0e-4e3e-89d7-b6b8a1d03f62/Вставленный текст.txt

## Scope and budget

Three fresh TRAIN→VALID fits in order no_memory,index_memory,time_memory, seed2026;
parameters715020/715021/715021. Existing MIMOdual fixedR0=838393ms, shared functions,
rank4/chunk8,2layers/2heads,L50. Archive original causal output_norm H, not native SSM S_t.
K4,anchors1/4/16/32,strict j<t,greedy log1p nearest/latest tie,unique positions,sorted slots.
FP32 dot/sqrt64 reader, signed lambda=tanh(beta),singlefp32beta EXACT0,no newQKV/norm/cache.
Primary time−index; both vsfresh no_memory secondary. No temporal methods frompriornegativepilots.
No TEST/continuation/refit/extra seeds/confirmation/other memory/history extension/article/messages.
Current job: A100x1,CPU4,mem0,rocky/proj_1833/type_e,no-requeue,**4h maximum**.
**Submits2/2 used. NO further submit or automatic refit under any outcome.**

## Original attempt and exact correction

Job4373262,execution55d812bf55b1dffbbab6a7b0da86e616a6227b8b,source410
4e26434f8bf084d6acd562e6859c24699013510d8f2ffcaaf2653435d3698e8d.
Cancelled at18:30:29MSK whilePENDING, beforeallocation: StartNone,elapsed00:00:00,nodeNoneassigned;
no pipeline/GPUgate/smoke/scientificfit. Cancellation only pending, prompted by proven wrapper
failure, not queue duration.13compactfiles preservedSHA beforecorrection in commit3b2832b.
[Parent audit](../evidence/job4373262/pre_fit_cancellation_audit.json).
Progress writer used update before initialcreate. A second reproducedreportfailure erasedunknown
start when lock existed without result. Corrected4wrapperfiles and added18regressions;406oldsource
files and method/tolerances unchanged;train/pipeline run+scientificvalidationASTs identical.
[Retry review](retry_review.json), [reproduced progress failure](progress_initialization_repro.json),
[reproduced unknown counter](report_unknown_start_repro.json), [regressions](wrapper_regression.json).

## Pre-submit evidence

Exactexecution CPU85/85 and noGit85/85 PASS,zeroerrors/failures/skips,CUDAuninitialized.
RepeatedTRAIN-only10000inputhistory coveragePASS: inputs and diagnostics exactlyequalattempt001,
only6expectedsource/attempt/timestamp fields differ. No targets/loaders/models/forwards/CUDA.
89.02% differentselectedsets,91.27% have>=4past events,2.11%empty. Anchorsunchanged.
Spanmean206036133.1549ms,max1026592333ms;reach1/4/16/32R0=97.21/96.31/94.12/91.08%.
CoverageSHA d19a9ebdef16c6cfa9c535148096b0fd9b4cee5e818de335a3dbdbbaa21ec40c.
GPUgate frozen11cases/429leaves,smoke3modesx3steps,scientific3fits await actual execution.
InheritedMIMOkernel45cases/2342checks boundbySHA. Negative-control raw failedchecks areexpected
inside negative_evidence;randomsmoke beta movement not numericalPASScriterion.

## Next safe actions

One compact read-onlypoll; helper enforces>=600s since previouspoll:
`python3 experiments/mamba3_time_memory/runtime/capture_status.py 4373393 53bcc76752d0d7fd06c8ba0866709a64cd978165 --attempt 002`
It reads compactprogress,not epochhistories/weights,and synchronizesstate/HANDOFF. No duplicate.
If queue>4h orsessionends saveactualhandoff;do notcancel/requeue forqueue. No background promise.

On observedterminal:
`python3 experiments/mamba3_time_memory/runtime/preserve_terminal.py 4373393 53bcc76752d0d7fd06c8ba0866709a64cd978165 --attempt 002`
StreamscheckpointSHA/bytes;neverdeserializes/downloadsweights. Preservesretry_review too.
Then stdlib-only independent fullaudit:
`PYTHONNOUSERSITE=1 python3 -B -m experiments.mamba3_time_memory.runtime.audit_saved experiments/mamba3_time_memory/evidence/job4373393 --execution 53bcc76752d0d7fd06c8ba0866709a64cd978165 --cpu-tests 85`
Check everymetriccell/log,selection/lastties/earlystop,first27,checkpointmetadata/SHA,
control exacthistoricalhistory+weightSHA,pairing,source/runtime/coverageownership,TEST0.
Quantiles areapproximate256queryreservoir;only8examplespublished,notfullreconstruction.

After PASS+3/3complete: preserve→commit; createexperiments/mamba3_time_memory/RESULTS.md;
update reports/MAMBA3_TIME_MECHANISMS_RESULTS.md#time-addressed-memory-pilot and reports/RESULTS.md;
appendexact3newregistryrows122→125 with old56821bytes/header unchanged
(oldregistrySHA073435a00d73646b4ed2c442a2538cd60338090a0bf00ae18902ca8a9181e79e).
Negativecompletepilot publishedtoo. Thenreviewdiff,normalmerge/pushmainauthorized.
Reportlimited50eventcausalrepresentationreadoutpilot,notalllifelongmemory.
FinalA–K perattachment;shortlowercasepoliteRussian draftmessagewithfullGitHubURLs,not sent.
Ifincomplete: preservepartialbranch/report,do notclaimpoint5complete,NOnewsubmit.
