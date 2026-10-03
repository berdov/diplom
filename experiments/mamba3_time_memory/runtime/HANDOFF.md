# Time-addressed memory pilot: job4373262

Phase: SUBMITTED_PENDING. **3 октября 2026, 18:06:24 MSK:** PENDING (Priority), node not assigned,
elapsed 00:00:00. Scientific fits 0/3, TEST 0. Next scheduler poll no earlier than 18:16:24 MSK
(2026-10-03T15:16:24.816908+00:00). No background process.
[Exact snapshot](status_4373262.json), [state](state.json),
[submission preservation](../evidence/job4373262/submission_preservation.json).

## Canonical execution

Local /Users/berdov/diplom; cluster /home/daryumin/iberdov/diplom, hse-karizma.
Branch exp/mamba3-time-addressed-memory, base main7ba5e45ed110914b354c89d4377836910b55408c.
Exact execution **55d812bf55b1dffbbab6a7b0da86e616a6227b8b**;
source **4e26434f8bf084d6acd562e6859c24699013510d8f2ffcaaf2653435d3698e8d**,410files.
Cluster checkout remains this execution with tracked clean; do not modify while job uses it.
Other running/pending jobs had unrelated WorkDir/Command, none cancelled. Existing untracked
artifacts preserved;40files adopted only after exact byte comparison to published Git blobs,
including7 historical layer-temporal raw JSON and33 new development files.

Original task attachment:
/Users/berdov/.codex/attachments/4de68a7a-1c0e-4e3e-89d7-b6b8a1d03f62/Вставленный текст.txt
Read it on resumption. Registry actually122. No reports/main edits before complete audit.

## Design and completed pre-submit checks

Order no_memory/index_memory/time_memory, seed2026; counts715020/715021/715021.
MIMO dual fixed-reference, shared maps, rank4/chunk8, history50. K4, anchors1/4/16/32,
R0=838393ms. Archive original causal output_norm H, not native SSM S_t; no persistence.
Single fp32 beta EXACT0, lambda=tanh(beta), signed residual, no Q/K/V or extra norm.
Primary time−index; secondary both vs no_memory. Design/source frozen beforecoverage.

CPU67/67 and no-Git67/67 PASS,0failures/errors/skips,CUDAuninitialized. These are exact
execution/source proofs, not just developer tests. Gate frozen11cases/429leaves; not run yet.
TRAIN coverage10000fixed input histories only, no model/forward/loaders/targetfields:
PASS, selected sets equal10.98%, different89.02%;91.27% have≥4past events.
Window spans reach anchors1/4/16/32R0 in97.21%/96.31%/94.12%/91.08% ofsample.
No anchors changed. Full coverage evidence copied with submission.

Durable reservation and submission on cluster under slurm_logs/attempt_001.
One job6hA100,CPU4,mem0,rocky/proj_1833/type_e,no-requeue. Submission budget used1/2;
scientific fit budget0/3. Second allocation allowed only for proven infrastructure/wrapper
failure BEFORE any scientific fit, terminalparent+preservation+regression, unchangedmethod
and tolerances, at most4h. No retry for numericalparity/quality/OOMsmoke or interruptedfit.
No continuation/confirmation/newseeds/TEST/historyextension/othermemory/articlechanges/messages.

## Next safe actions

One compact explicit poll after due time:
`python3 experiments/mamba3_time_memory/runtime/capture_status.py 4373262 55d812bf55b1dffbbab6a7b0da86e616a6227b8b`
Helper enforces≥600seconds, reads pipeline/gate/smoke summaries and small progress.json,
not full scientific histories or checkpoint weights. Do not queryscheduler separately moreoften.
If queue exceeds4h or sessionends, save actualhandoff and stop; do not cancel/resubmit.

When terminal:
`python3 experiments/mamba3_time_memory/runtime/preserve_terminal.py 4373262 55d812bf55b1dffbbab6a7b0da86e616a6227b8b`
It streams SHA/bytes ofweights, neverdeserializes/downloadsthem; preservescompactraw/logs.
Then independent runtime/audit_saved.py (being prepared) mustvalidate all3fits/logmetrics,
controlreplay, pairing, ownershipinclcoverageSHA, gate11/429,smoke3×3, TEST0, checkpointmetadata,
selection/ties/earlystop, first27, diagnostics andsummary. No newforwards forpublication.

At completePASS: preserve→commit; report/registry122→125exactoldprefix→publicationcommit;
merge/pushmain. Negative completepilot also published. Point5 wording restricted to current
50-event causal representation readout pilot, not all possible memory. Final A–K asattachment;
short lowercase draft message, politeaddress, fullGitHubURLs, notsent. Stopafterreport.
