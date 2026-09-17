"""Frozen training settings with only seed/mode/output overrides."""
import json
from pathlib import Path

from experiments.mamba3_timeaware.config import load_config as frozen_config

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
STUDY_ID = 'mamba3_time_confirmation_001'
CORE = '460bd3e687d26a458cafc21960391f83905da962ca5c992d6062c7b048f0a73f'
EVIDENCE = ROOT / 'experiments/mamba3_time_mechanisms/runs/gpu_equivalence_001.json'
EVIDENCE_SHA = '6484b49cf6d69e2066cc8db6c9c7dbc12bf33a682136682fbf83798e70c9de37'
STATS = ROOT / 'experiments/mamba3_timeaware/runs/train_time_stats_001.json'
STATS_SHA = 'fa5df0e5ec97d84e5dffd157373318ebfa2cb94fe2853c2aec4898ecd5f89943'
MANIFEST = ROOT / 'outputs/data/protocol_b_manifest.json'
MANIFEST_SHA = '9f39aa12ec16f697a8bedb91eeb21c46dce514e6f414b7e47fc4064c1fcffd2c'
COUNTS = dict(shared=610506, separate=610572, separate_constant_gap=610572)


def plan():
    data = json.loads((HERE / 'study_plan.json').read_text())
    expected = [(mode, seed) for seed in range(2027, 2031) for mode in ('shared', 'separate')]
    expected.append(('separate_constant_gap', 2026))
    if [(r['mode'], r['seed']) for r in data['tasks']] != expected:
        raise ValueError('Unexpected task mapping')
    if [r['task_index'] for r in data['tasks']] != list(range(9)):
        raise ValueError('Unexpected array indices')
    if len({r['run_id'] for r in data['tasks']}) != 9 or data['study_id'] != STUDY_ID:
        raise ValueError('Study/run ID mismatch')
    return data


def paths(task):
    run_id = task['run_id']
    if not run_id or Path(run_id).name != run_id or run_id in ('.', '..'):
        raise ValueError('Invalid run ID')
    runtime = HERE / 'slurm_logs' / run_id
    return dict(result=HERE / 'runs' / (run_id + '.json'), runtime=runtime,
                lock=runtime / 'run.lock', checkpoints=runtime / 'checkpoints',
                smoke_checkpoint=runtime / 'smoke_roundtrip.pth',
                tensorboard=runtime / 'log_tensorboard', cache=runtime / 'cache')


def settings(task):
    data = frozen_config()
    data.update(seed=task['seed'], checkpoint_dir=str(paths(task)['checkpoints']))
    if task['mode'] != 'shared':
        data['time_mechanism_mode'] = 'separate'
    return data


def scientific_settings(data):
    result = {k:v for k,v in data.items() if k not in ('seed', 'checkpoint_dir', 'time_mechanism_mode')}
    # RecBole serializes YAML gpu_id="0" as integer 0 in the historical settings.
    result['gpu_id'] = str(result['gpu_id'])
    return result
