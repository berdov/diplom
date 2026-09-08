import hashlib
import json
import math
from pathlib import Path
import subprocess
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_config():
    return yaml.safe_load((HERE / 'config.yaml').read_text())


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def source_digest():
    # Runtime/env and package caches live below HERE on the cluster. Never hash
    # installed third-party code or generated artifacts as challenger source.
    paths = sorted(list(HERE.glob('*.py')) + list((HERE/'methods').glob('*.py')) +
                   list((HERE/'tests').glob('*.py')) +
                   list(HERE.glob('*.yaml')) + list(HERE.glob('requirements*.txt')) +
                   [ROOT / 'slurm/moo_representative_challengers.sh'])
    text = '\n'.join(f'{p.relative_to(ROOT)} {digest(p)}' for p in paths if p.is_file())
    return hashlib.sha256(text.encode()).hexdigest()


def verify_historical_inputs():
    manifest = yaml.safe_load((HERE / 'historical_inputs.yaml').read_text())
    changed = [p for p, h in manifest['files'].items() if not (ROOT/p).is_file() or digest(ROOT/p) != h]
    if changed:
        raise RuntimeError(f'Historical inputs changed: {changed}')
    return {'files_verified':len(manifest['files']), 'manifest_sha256':digest(HERE/'historical_inputs.yaml')}


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(text + '\n')
    temporary.replace(path)


def select_operating_point(records, method, config):
    from experiments.moo_8families.evaluation.pareto import objective_point_from_record
    for record in records:
        vector = objective_point_from_record(record)
        if not all(math.isfinite(x) for x in vector):
            raise RuntimeError('Non-finite validation objective vector')
        record['objective_vector'] = vector
        record['normalized_objective_vector'] = [v/s for v,s in zip(vector,config['selection']['validation_coordinate_scales'])]
        record['selection_score'] = sum(r*v for r,v in zip(config['selection']['weights'],record['normalized_objective_vector']))
    if method == 'most':
        return min(records, key=lambda r:(r['selection_score'], r['solution_index']))
    points = [r for r in records if r['preference_id'] == config['selection']['preference_id']]
    if len(points) != 1:
        raise RuntimeError('Predefined ranking operating point missing or duplicated')
    return points[0]


def require_gate(method, stage):
    report = json.loads((HERE / 'verification.json').read_text())
    current = source_digest()
    if report['status'] != 'passed' or report['source_digest'] != current or report['test_evaluation_count'] != 0:
        raise RuntimeError('Unit/parity gate missing, failed, or stale')
    previous = {'sanity':'smoke','convergence_screening':'sanity'}.get(stage)
    if previous:
        path = HERE / 'runs' / f'{method}_{previous}_001.json'
        result = json.loads(path.read_text())
        if (result['status'] != 'completed' or not result['gates']['passed'] or
            result['test_evaluation_count'] != 0 or result['source_digest'] != current):
            raise RuntimeError(f'Previous stage gate failed/stale: {path}')
    return {'unit_parity_source_digest':current,'verification_sha256':digest(HERE/'verification.json'),
            'previous_stage':previous}
