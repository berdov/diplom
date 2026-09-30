"""Exact saved MIMO dual settings, independent variant/output namespace."""
import copy
from pathlib import Path
from experiments.mamba3_mimo_time.records import read
from experiments.mamba3_mimo_time.confirmation.config import CORE, PIN, POLICY, POLICY_SHA, ADMISSION, ADMISSION_SHA, SMOKE as OLD_SMOKE, SMOKE_SHA, PILOT_COMMIT, PILOT_SOURCE

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BRANCH = 'exp/mamba3-head-timescales'
PUBLICATION = 'ce6da46099fcc15f74f61c28ed946f78210ac053'
STUDY = 'mamba3_head_timescales_001'
EXECUTION_ATTEMPT = '002'
MANIFEST = HERE/'source_manifest_002.json'
PRESERVATION = HERE/'evidence/job4361071/preservation_manifest.json'
MODES = ('fixed', 'shared_tau', 'head_tau')
COUNTS = dict(fixed=715020, shared_tau=715022, head_tau=715024)
PILOT = ROOT/'experiments/mamba3_mimo_time/runs/attempt_003/mamba3_mimo_dual_seed2026_001.json'
LOGS, RUNS = HERE/'slurm_logs/attempt_002', HERE/'runs/attempt_002'
LOGIN, RESERVATION, SUBMISSION = (LOGS/n for n in ('login_verification_001.json', 'reservation_001.json', 'submission_001.json'))
PIPELINE = LOGS/'pipeline_status.json'
INHERITED, GATE, SMOKE, SUMMARY = (RUNS/n for n in ('inherited_kernel_001.json','targeted_gate_001.json','smoke_001.json','pilot_summary.json'))
LAUNCHER = ROOT/'slurm/mamba3_head_timescales.sh'


def plan():
    p = read(HERE/'study_plan.json')
    if p['study_id'] != STUDY or p['modes'] != list(MODES) or p['parameter_counts'] != COUNTS or p['max_scientific_fits'] != 3:
        raise ValueError('Plan scope drift')
    return p


def paths(variant):
    if variant not in MODES:
        raise ValueError('Unknown variant')
    name = f'mamba3_headtime_{variant}_seed2026_001'
    runtime = LOGS/name
    return dict(run_id=name,runtime=runtime,result=RUNS/(name+'.json'),lock=runtime/'run.lock',
                checkpoint=runtime/'checkpoints/best_state_dict.pth',metadata=runtime/'checkpoints/best_metadata.json')


def settings(variant, device=None, checkpoint_dir=None):
    p = paths(variant)
    values = copy.deepcopy(read(PILOT)['config'])
    values.update(time_scale_mode=variant,checkpoint_dir=str(checkpoint_dir or p['checkpoint'].parent))
    if device == 'cpu':
        values.update(use_gpu=False,device='cpu',gpu_id='')
    elif device == 'cuda':
        values.update(use_gpu=True,device='cuda')
    elif device is not None:
        raise ValueError(device)
    return values


def unused():
    candidates = [LOGIN,RESERVATION,SUBMISSION,PIPELINE,INHERITED,GATE,SMOKE,SUMMARY,SUMMARY.with_suffix('.md'),LOGS/'pipeline.lock',LOGS/'gate',LOGS/'smoke']
    for variant in MODES:
        candidates += [v for k,v in paths(variant).items() if k in ('result','runtime','lock','checkpoint','metadata')]
    occupied = [str(p) for p in candidates if p.exists() or p.is_symlink()]
    if occupied:
        raise FileExistsError('One-shot study already owned: '+repr(occupied))
