"""Separate one-shot namespace; immutable historical scientific settings."""
from pathlib import Path
from .. import config as parent
from .reuse import bind

HERE = Path(__file__).resolve().parent
ROOT = parent.ROOT
BRANCH = 'exp/mamba3-gap-trap-centered'
PUBLICATION = 'f533a49038a5a055a19837bb2dfde4518f10e3f4'
STUDY = 'mamba3_gap_trap_centered_001'
EXECUTION_ATTEMPT = '002'
MANIFEST = HERE/'source_manifest_002.json'
PLAN = HERE/'study_plan_002.json'
MODES = ('fixed_replay', 'centered_gap_trap')
COUNTS = dict(fixed_replay=715020, centered_gap_trap=715021)
PILOT = parent.PILOT
CORE, PIN, POLICY, POLICY_SHA = parent.CORE, parent.PIN, parent.POLICY, parent.POLICY_SHA
ADMISSION, ADMISSION_SHA = parent.ADMISSION, parent.ADMISSION_SHA
OLD_SMOKE, SMOKE_SHA = parent.OLD_SMOKE, parent.SMOKE_SHA
PILOT_COMMIT, PILOT_SOURCE = parent.PILOT_COMMIT, parent.PILOT_SOURCE
LOGS, RUNS = HERE/'slurm_logs/attempt_002', HERE/'runs/attempt_002'
LOGIN, RESERVATION, SUBMISSION = (LOGS/n for n in ('login_verification_002.json','reservation_002.json','submission_002.json'))
PIPELINE = LOGS/'pipeline_status.json'
INHERITED, GATE, SMOKE, SUMMARY = (RUNS/n for n in ('inherited_kernel_002.json','targeted_gate_002.json','smoke_002.json','pilot_summary.json'))
LAUNCHER = ROOT/'slurm/mamba3_gap_trap_centered_002.sh'
PARENT_EXECUTION = '8ee54cf1faee22bb3abad3f31aa9268d77a125d1'


def paths(variant):
    if variant not in MODES:
        raise ValueError('Unknown centered study variant')
    name = f'mamba3_gaptrap_centered_{variant}_seed2026_{EXECUTION_ATTEMPT}'
    runtime = LOGS/name
    return dict(run_id=name, runtime=runtime, result=RUNS/(name+'.json'),
                lock=runtime/'run.lock', checkpoint=runtime/'checkpoints/best_state_dict.pth',
                metadata=runtime/'checkpoints/best_metadata.json')


_bound = bind(parent, {k:v for k,v in globals().items() if k.isupper()} | {'paths': paths})
settings, unused = (_bound[k] for k in ('settings','unused'))


def plan():
    value = parent.read(PLAN)
    tasks = [dict(gap_trap_mode=mode,mode='dual',seed=2026,run_id=paths(mode)['run_id']) for mode in MODES]
    expected = dict(study_id=STUDY,execution_attempt=EXECUTION_ATTEMPT,modes=list(MODES),
                    parameter_counts=COUNTS,tasks=tasks,seed=2026,max_scientific_fits=2,
                    max_jobs=1,test_evaluations=0,automatic_retries=0)
    if any(value.get(k)!=v for k,v in expected.items()):
        raise ValueError('Attempt002 plan scope drift')
    if value['retry_authorization']['previous_job_id']!='4371302' or value['retry_authorization']['previous_scientific_fits']!=0:
        raise ValueError('Expected one explicitly authorized retry after zero-fit failure')
    return value
