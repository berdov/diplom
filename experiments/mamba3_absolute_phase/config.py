"""One frozen phase design; explicit scientific stage, attempt and seed."""
import copy
import os
from pathlib import Path
from experiments.mamba3_mimo_time.records import read
from experiments.mamba3_mimo_time.confirmation.config import (
    CORE, PIN, POLICY, POLICY_SHA, ADMISSION, ADMISSION_SHA,
    SMOKE as OLD_SMOKE, SMOKE_SHA, PILOT_COMMIT, PILOT_SOURCE)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PUBLICATION = 'de2cddf271bd4c5ccaa4693552f005799a171526'
BRANCH = 'exp/mamba3-absolute-phase'
STUDY = 'mamba3_absolute_phase_001'
STAGE = os.environ.get('ABS_PHASE_STAGE', 'pilot')
EXECUTION_ATTEMPT = os.environ.get('ABS_PHASE_ATTEMPT', '001')
if STAGE not in ('pilot', 'confirmation') or EXECUTION_ATTEMPT not in ('001', '002'):
    raise ValueError('Only pilot/conditional confirmation and one pre-fit retry')
MODES = ('baseline_dual', 'relative_phase', 'absolute_phase')
COUNTS = dict(baseline_dual=715020, relative_phase=715148, absolute_phase=715148)
SEEDS = (2026,) if STAGE == 'pilot' else (2027, 2028, 2029, 2030)
MANIFEST = HERE / (('source_manifest.json' if EXECUTION_ATTEMPT == '001' else 'source_manifest_002.json')
                   if STAGE == 'pilot' else ('source_manifest_confirmation.json' if EXECUTION_ATTEMPT == '001'
                                            else 'source_manifest_confirmation_002.json'))
PARENT_MANIFEST = ROOT / 'experiments/mamba3_time_memory/source_manifest_002.json'
PILOT = ROOT / 'experiments/mamba3_mimo_time/runs/attempt_003/mamba3_mimo_dual_seed2026_001.json'
LOGS = HERE / 'slurm_logs' / STAGE / ('attempt_' + EXECUTION_ATTEMPT)
RUNS = HERE / 'runs' / STAGE / ('attempt_' + EXECUTION_ATTEMPT)
LOGIN, RESERVATION, SUBMISSION, PIPELINE = (LOGS / n for n in (
    'login_verification.json', 'reservation.json', 'submission.json', 'pipeline_status.json'))
COVERAGE = LOGS / 'train_coverage.json'
INHERITED, GATE, SMOKE, SUMMARY = (RUNS / n for n in (
    'inherited_kernel.json', 'targeted_gate.json', 'smoke.json', STAGE + '_summary.json'))
LAUNCHER = ROOT / 'slurm/mamba3_absolute_phase.sh'


def paths(variant, seed=2026):
    if variant not in MODES or type(seed) is not int:
        raise ValueError('Invalid mode/seed')
    name = f'mamba3_absolute_phase_{variant}_seed{seed}_001'
    runtime = LOGS / name
    return dict(run_id=name, runtime=runtime, result=RUNS / (name + '.json'),
                lock=runtime / 'run.lock', checkpoint=runtime / 'checkpoints/best_state_dict.pth',
                metadata=runtime / 'checkpoints/best_metadata.json')


def tasks():
    return [dict(variant=v, seed=s, run_id=paths(v, s)['run_id']) for s in SEEDS for v in MODES]


def historical(seed):
    if seed == 2026:
        return PILOT
    if seed not in (2027, 2028, 2029, 2030):
        raise ValueError('No historical scientific reference for this seed')
    return ROOT / f'experiments/mamba3_mimo_time/confirmation/runs/mamba3_mimo_confirm_dual_seed{seed}_001.json'


def settings(variant, seed=2026, device=None, checkpoint_dir=None):
    values = copy.deepcopy(read(PILOT)['config'])
    values.update(seed=seed, phase_mode=variant,
                  checkpoint_dir=str(checkpoint_dir or paths(variant, seed)['checkpoint'].parent))
    if device is not None:
        if device not in ('cpu', 'cuda'):
            raise ValueError(device)
        values.update(use_gpu=device == 'cuda', device=device)
        if device == 'cpu':
            values['gpu_id'] = ''
    return values


def plan():
    value = read(HERE / 'study_plan.json')
    expected = dict(study_id=STUDY, modes=list(MODES), parameter_counts=COUNTS,
                    periods_ms=[21600000, 86400000], K=2, max_scientific_fits=15,
                    max_jobs=3, max_requested_gpu_seconds=64800, test_evaluations=0,
                    pilot_seeds=[2026], confirmation_seeds=[2027, 2028, 2029, 2030])
    if any(value.get(k) != v for k, v in expected.items()):
        raise ValueError('Frozen phase plan drift')
    return value


def parameter_keys(variant):
    return plan()['common_parameter_keys'] + ([] if variant == 'baseline_dual' else ['phase_adapter.W'])


def allocation_seconds():
    return 14400 if EXECUTION_ATTEMPT == '002' else 21600 if STAGE == 'pilot' else 28800


def unused():
    candidates = [LOGIN, RESERVATION, SUBMISSION, PIPELINE, INHERITED, GATE, SMOKE,
                  SUMMARY, LOGS / 'pipeline.lock', LOGS / 'gate', LOGS / 'smoke']
    for t in tasks():
        candidates.extend(p for k, p in paths(t['variant'], t['seed']).items() if k != 'run_id')
    occupied = [str(p) for p in candidates if p.exists() or p.is_symlink()]
    if occupied:
        raise FileExistsError('Already reserved; do not duplicate: ' + repr(occupied))
