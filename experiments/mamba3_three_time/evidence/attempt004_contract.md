# Attempt004: limited diagnostic contract

No recommender TRAIN/VALID/TEST, optimizer fitting, data loaders or automatic promotion.
training_authorized is always false. DIAGNOSTICS_COMPLETE means records were
collected, never that every correctness/legacy check passed. Profiles from
attempt003 are unchanged; its 16 original artifacts are preserved verbatim.

SISO carries an invocation-owned collector from forward ctx into backward.
Inputs are independent snapshots; strides/storage offsets and any layout change
are recorded. Thread IDs and forward/backward call counts accompany each record.
A two-layer trace on/off/RNG smoke precedes only two drift fixtures (dual train
L50 and base eval L64). Instrumentation failure stops SISO, not MIMO.
Autotuned and fixed supported launch (4 warps, 2 stages, maxnreg 128) replays
are diagnostics, never a launch search. Hybrid is conditional and not production.
Frozen upstream kernels, default upstream arithmetic, calibrators and padding
adapter are unchanged. Original-arithmetic suites and positional prefix residuals
are reported separately from legacy exact-zero status.

MIMO investigates L15 independent triple, L50 official tied and a D-only L7
fixture. D denotes the skip weight, not DT_write/DT_phase. Signed and nonnegative
cotangents are fixed independently of outputs. All input derivatives are retained
as comparisons; full D, outputs, losses and incoming gradients are recorded.
The independent D coefficient follows pinned forward order: bf16(Psi),
bf16(V*Psi), D skip, bf16(SiLU(Z*Zeta)), bf16(Phi), rank reduction.
Rounded coefficients are continuous/STE diagnostics, not derivatives of a
discontinuous quantizer. A separate pinned-backward cast transcription is labeled
as such, and not used to assert forward correctness. Ideal FP64 coefficient has
an independent CPU autograd check. No candidate derivative is its oracle.

For L50 the unchanged objective mean(y*w)+.03*mean(y^2) is decomposed in FP64.
The output-error bound explains propagation, not replacement acceptance.
Mixed atol/rtol and the old relative cap both remain visible. Cancellation and
reference scale are recorded; no threshold is chosen from measured triple errors.

Separate SISO/MIMO subprocesses: 40 minutes each, one A100 allocation of 90
minutes, no retries. Exclusive submission/execution files forbid reuse.
After one sbatch --parsable response, persist Job ID and stop without monitoring.
