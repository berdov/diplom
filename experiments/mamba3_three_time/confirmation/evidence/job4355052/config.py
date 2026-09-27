"""Explicit task allowlist and settings from recorded pilot, not new defaults."""
import copy
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PILOT = HERE.parent / 'validation_pilot'
STUDY = 'mamba3_three_time_siso_confirmation_001'
SEEDS = (2027, 2028, 2029, 2030)
MODES = ('dual', 'triple')
COUNTS = {'dual': 610572, 'triple': 610638}
LOGS, RUNS = HERE / 'slurm_logs', HERE / 'runs'
ATTEMPT = LOGS / 'attempt_002'
SUBMISSION = ATTEMPT / 'submission_002.json'
LOGIN_VERIFICATION = ATTEMPT / 'login_verification.json'
LOCK = LOGS / 'pipeline.lock'
FAILED = HERE / 'evidence/submission_001'
PARENT_COMMIT = '886ef23d0d922bea4fff5fd835834cf5daececcb'
PARENT_HASH = '1d524e1e1be74a458193a6521789b0e224ca7e038295aebf4f09d2f3db72df94'
BATCH, INIT, SUMMARY = (RUNS / p for p in ('one_batch_001.json', 'initialization_001.json', 'confirmation_summary.json'))
PILOT_COMMIT = '6b5618a769d8e3424df0b7232605426fac8c573a'
B_COMMIT = '6feb8329334fe94d894c9d79f5a412a1241d5b3f'
PILOT_HASH = '105e09559cde506d0a6de46f8731213a5a6a8af0c56b20924d5c702c9ec87b58'
POLICY_SHA = 'b40f72eae5895c379143e8ef590a66f7dab80b9b61e1f1349fcdf16efb509c50'


def task(mode, seed):
    if mode not in MODES or type(seed) is not int or seed not in SEEDS:
        raise ValueError('Only dual/triple seeds2027-2030 may fit')
    return dict(mode=mode, seed=seed, run_id=f'mamba3_three_time_confirm_siso_{mode}_seed{seed}_001')


def plan():
    value = json.loads((HERE / 'study_plan.json').read_text())
    if value['tasks'] != [task(m, s) for s in SEEDS for m in MODES]:
        raise ValueError('Frozen eight-task order changed')
    return value


def paths(mode, seed):
    t = task(mode, seed)
    runtime = LOGS / t['run_id']
    return dict(runtime=runtime, result=RUNS / (t['run_id'] + '.json'), lock=runtime / 'run.lock',
                checkpoint=runtime / 'checkpoints/best_state_dict.pth', metadata=runtime / 'checkpoints/best_metadata.json')


def pilot(mode):
    if mode not in MODES:
        raise ValueError(mode)
    return json.loads((PILOT / f'runs/mamba3_three_time_siso_{mode}_seed2026_001.json').read_text())


def settings(mode, seed, checkpoint_dir=None):
    if mode not in MODES or seed not in (2026, *SEEDS):
        raise ValueError('Unknown initialization/config task')
    result = copy.deepcopy(pilot(mode)['config'])
    result.update(seed=seed, checkpoint_dir=str(checkpoint_dir or paths(mode, seed)['checkpoint'].parent))
    return result


def effective_check(config, mode, seed, cpu=False):
    actual = json.loads(json.dumps(config.final_config_dict, default=str))
    expected = copy.deepcopy(pilot(mode)['effective_config'])
    expected.update(seed=seed, checkpoint_dir=actual['checkpoint_dir'])
    if cpu:
        expected.update(use_gpu=False, device='cpu')
    differences = [k for k in set(actual) | set(expected) if actual.get(k) != expected.get(k)]
    if differences:
        raise ValueError('Pilot effective config drift: ' + repr(differences))
    return dict(status='PASS', excluded=['seed', 'checkpoint_dir'], cpu_construction_only=cpu)


def unused(include_submission=False):
    files = [BATCH, INIT, SUMMARY, SUMMARY.with_suffix('.md'), SUMMARY.with_suffix('.svg'), LOCK,
             LOGS / 'pipeline_status.json']
    for t in plan()['tasks']:
        files.extend(paths(t['mode'], t['seed'])[k] for k in ('result', 'lock', 'checkpoint', 'metadata'))
    if include_submission:
        files.extend((SUBMISSION, LOGIN_VERIFICATION))
    occupied = [str(p) for p in files if p.exists() or p.is_symlink()]
    if occupied:
        raise FileExistsError('Existing reservation/results; no overwrite/retry: ' + repr(occupied))
