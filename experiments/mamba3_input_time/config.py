"""Frozen scientific settings, four explicit run IDs and private runtime paths."""
import json
from pathlib import Path
from experiments.mamba3_timeaware.config import load_config

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
STUDY_ID = 'mamba3_input_time_001'
MODES = ('separate_replay', 'time_add', 'attention_content', 'attention_time')
CORE = '460bd3e687d26a458cafc21960391f83905da962ca5c992d6062c7b048f0a73f'
CONFIRMATION = '307d6cf2c80a90fa8c89aa5768d9da3aafbc9cfa9f4beaea89393c201d7c0346'
RUNTIME = HERE / 'slurm_logs'
GPU_EVIDENCE = HERE / 'runs/gpu_checks_001.json'
SUMMARY = HERE / 'runs/pilot_summary.json'


def plan():
    data = json.loads((HERE / 'study_plan.json').read_text())
    expected = [dict(mode=m, seed=2026, run_id=f'mamba3_input_{m}_seed2026_001') for m in MODES]
    if data['tasks'] != expected or data['study_id'] != STUDY_ID:
        raise ValueError('Frozen four-task plan mismatch')
    return data


def paths(task):
    if task not in plan()['tasks']:
        raise ValueError('Unknown task')
    runtime = RUNTIME / task['run_id']
    return dict(runtime=runtime, result=HERE / 'runs' / (task['run_id'] + '.json'),
                lock=runtime / 'run.lock', checkpoint=runtime / 'checkpoints/best_state_dict.pth',
                metadata=runtime / 'checkpoints/best_metadata.json', cache=runtime / 'cache',
                tensorboard=runtime / 'log_tensorboard', stdout=runtime / 'stdout.log', stderr=runtime / 'stderr.log')


def settings(task):
    result = load_config()
    result.update(seed=2026, time_mechanism_mode='separate', input_time_mode=task['mode'],
                  checkpoint_dir=str(paths(task)['checkpoint'].parent))
    return result


def scientific_settings(settings):
    result = {k: v for k, v in settings.items() if k not in ('checkpoint_dir', 'time_mechanism_mode', 'input_time_mode', 'context_time_mode')}
    result['gpu_id'] = str(result['gpu_id'])
    return result
