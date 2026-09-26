"""Private paths and frozen scientific settings."""
import json
from pathlib import Path
from experiments.mamba3_three_time.config import settings as frozen_settings

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PARENT = HERE.parent
STUDY = 'mamba3_three_time_siso_validation_001'
MODES = ('dual', 'triple')
COUNTS = dict(dual=610572, triple=610638)
LOGS = HERE / 'slurm_logs'
RUNS = HERE / 'runs'
GATE = RUNS / 'admission_001.json'
SMOKE = RUNS / 'smoke_001.json'
SUMMARY = RUNS / 'pilot_summary.json'
SUBMISSION = LOGS / 'submission_001.json'
LOCK = LOGS / 'pipeline.lock'
ACCEPTED = 'ACCEPTED_FOR_SISO_PILOT_WITH_DOCUMENTED_NUMERICAL_RESIDUAL'
EXECUTION004 = '176f72a206eb48ddc4de492b87bc3273b75e1cae'
SOURCE004 = '795022880fe1459e3674e298638202bd10ddae2f04f3e737d0b4d90dbef035ca'


def plan():
    p = json.loads((HERE / 'study_plan.json').read_text())
    if p['tasks'] != [dict(mode=m, seed=2026, run_id=f'mamba3_three_time_siso_{m}_seed2026_001') for m in MODES]:
        raise ValueError('Two frozen scientific tasks only')
    return p


def paths(mode):
    if mode not in MODES:
        raise ValueError('Only SISO dual/triple')
    run_id = f'mamba3_three_time_siso_{mode}_seed2026_001'
    runtime = LOGS / run_id
    return dict(runtime=runtime, result=RUNS / (run_id + '.json'), lock=runtime / 'run.lock',
                checkpoint=runtime / 'checkpoints/best_state_dict.pth',
                metadata=runtime / 'checkpoints/best_metadata.json')


def settings(mode):
    result = frozen_settings('SISO', mode)
    result.update(seed=2026, checkpoint_dir=str(paths(mode)['checkpoint'].parent))
    return result


def unused(include_submission=False):
    files = [GATE, SMOKE, SUMMARY, SUMMARY.with_suffix('.md'), LOCK, LOGS / 'pipeline_status.json']
    for mode in MODES:
        p = paths(mode)
        files += [p[k] for k in ('result', 'lock', 'checkpoint', 'metadata')]
    if include_submission:
        files += [SUBMISSION]
    occupied = [str(p) for p in files if p.exists() or p.is_symlink()]
    if occupied:
        raise FileExistsError('Pilot already owned; no automatic _002 or retry: ' + repr(occupied))
