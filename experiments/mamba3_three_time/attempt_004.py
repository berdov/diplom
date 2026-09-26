"""Disjoint immutable attempt004 ownership; never reuses older files."""

from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
ATTEMPT='004'
PARENT='3a9771f91bbdc8233eee83bb326a0731f3830373'
PLAN=HERE/'test_plan_004.json'
SOURCE_MANIFEST=HERE/'source_manifest_004.json'
LOGS=HERE/'slurm_logs/attempt_004'
RUNS=HERE/'runs'
SUMMARY=RUNS/'technical_summary_004.json'
SUBMISSION=LOGS/'submission_004.json'
LOCK=LOGS/'execution_004.lock'
PREFLIGHT=LOGS/'login_preflight_004.json'
PIPELINE=LOGS/'pipeline_status.json'


def evidence(arch):
    if arch not in ('SISO','MIMO'):raise ValueError(arch)
    return RUNS/f'{arch.lower()}_diagnostics_004.json'


def require_unused():
    paths=[LOCK,PIPELINE,SUMMARY,evidence('SISO'),evidence('MIMO')]
    paths += [LOGS/f'{a}_{s}.log' for a in ('siso','mimo') for s in ('stdout','stderr')]
    occupied=[str(p) for p in paths if p.exists() or p.is_symlink()]
    if occupied:raise FileExistsError('Attempt004 already owned: '+repr(occupied))
