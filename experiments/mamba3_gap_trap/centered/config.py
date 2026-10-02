"""Separate one-shot namespace; immutable historical scientific settings."""
from pathlib import Path
from .. import config as parent
from .reuse import bind

HERE = Path(__file__).resolve().parent
ROOT = parent.ROOT
BRANCH = 'exp/mamba3-gap-trap-centered'
PUBLICATION = 'f533a49038a5a055a19837bb2dfde4518f10e3f4'
STUDY = 'mamba3_gap_trap_centered_001'
EXECUTION_ATTEMPT = '001'
MANIFEST = HERE/'source_manifest.json'
MODES = ('fixed_replay', 'centered_gap_trap')
COUNTS = dict(fixed_replay=715020, centered_gap_trap=715021)
PILOT = parent.PILOT
CORE, PIN, POLICY, POLICY_SHA = parent.CORE, parent.PIN, parent.POLICY, parent.POLICY_SHA
ADMISSION, ADMISSION_SHA = parent.ADMISSION, parent.ADMISSION_SHA
OLD_SMOKE, SMOKE_SHA = parent.OLD_SMOKE, parent.SMOKE_SHA
PILOT_COMMIT, PILOT_SOURCE = parent.PILOT_COMMIT, parent.PILOT_SOURCE
LOGS, RUNS = HERE/'slurm_logs/attempt_001', HERE/'runs/attempt_001'
LOGIN, RESERVATION, SUBMISSION = (LOGS/n for n in ('login_verification_001.json','reservation_001.json','submission_001.json'))
PIPELINE = LOGS/'pipeline_status.json'
INHERITED, GATE, SMOKE, SUMMARY = (RUNS/n for n in ('inherited_kernel_001.json','targeted_gate_001.json','smoke_001.json','pilot_summary.json'))
LAUNCHER = ROOT/'slurm/mamba3_gap_trap_centered.sh'
PARENT_EXECUTION = '8ee54cf1faee22bb3abad3f31aa9268d77a125d1'


def paths(variant):
    if variant not in MODES:
        raise ValueError('Unknown centered study variant')
    name = f'mamba3_gaptrap_centered_{variant}_seed2026_001'
    runtime = LOGS/name
    return dict(run_id=name, runtime=runtime, result=RUNS/(name+'.json'),
                lock=runtime/'run.lock', checkpoint=runtime/'checkpoints/best_state_dict.pth',
                metadata=runtime/'checkpoints/best_metadata.json')


_bound = bind(parent, {k:v for k,v in globals().items() if k.isupper()} | {'paths': paths})
plan, settings, unused = (_bound[k] for k in ('plan','settings','unused'))
