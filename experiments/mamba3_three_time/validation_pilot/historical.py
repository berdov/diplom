"""Read archived raw cases, not summary flags, to establish admission prerequisites."""
import json
from .config import PARENT, EXECUTION004, SOURCE004
from .policy import from_norms
from experiments.mamba3_three_time.records_003 import required_pass


def expected_ids():
    names = ['trace_smoke', 'drift_dual_train_L50', 'drift_base_eval_L64']
    names += [f'{s}_{m}_L{n}' for s in ('A', 'B') for m in ('eval', 'train') for n in (50, 64)]
    names += ['C_leaf_gradient_sum', 'C_tied_calibrator_recovery',
              'reference_calibration_official_tied_L7', 'D_three_path_L7', 'E_causality_padding']
    for n in (1, 50, 63, 64, 65, 97):
        names += [f'F_official_full_sequence_L{n}', f'F_reference_calibration_official_tied_L{n}', f'F_D_three_path_L{n}']
    names += ['H_state_dict_roundtrip']
    names += [f'causal_{m}_L{n}_P{p}_x{x}' for m in ('base', 'dual', 'triple')
              for n, p in ((17, 7), (65, 31), (97, 65)) for x in (1, 16)]
    return ['SISO/upstream/' + name for name in names]


def audit(value=None):
    path = PARENT / 'evidence/attempt_004/runs/siso_diagnostics_004.json'
    o = json.loads(path.read_text()) if value is None else value
    if (o['attempt_id'], o['job_id'], o['execution_commit'], o['source_hash'], o['backend']) != (
            '004', '4353838', EXECUTION004, SOURCE004, 'upstream'):
        raise ValueError('Archived SISO identity mismatch')
    ids = [r['case_id'] for r in o['cases']]
    if ids != expected_ids() or ids != o['required_registry'] or len(ids) != len(set(ids)):
        raise ValueError('Archived SISO coverage mismatch')
    residuals = []
    for row in o['cases']:
        if row['name'] == 'E_causality_padding':
            checks = row['checks']
            if row['failed_keys'] != ['future_gradient_zero'] or checks['future_gradient_zero']['finite'] is not True:
                raise ValueError('Unexpected archived padding failure')
            if not required_pass({k: v for k, v in checks.items() if k != 'future_gradient_zero'}):
                raise ValueError('Other padding checks failed')
        elif not row['passed'] or not required_pass(row):
            raise ValueError('Unknown required archived failure: ' + row['case_id'])
        if row.get('category') == 'causal':
            for name, variant in row['variants'].items():
                measured = variant['report']['embedding_output_gradient']
                accepted = from_norms(measured['prefix_l2'], measured['future_l2'])
                residuals.append(dict(case_id=row['case_id'], variant=name, historical=measured, policy_v1=accepted))
                if not accepted['finite_precision_pass']:
                    raise ValueError('Archived residual exceeds policy v1')
    if len(residuals) != 30 or o['training_authorized'] is not False:
        raise ValueError('Historical scope mismatch')
    return dict(status='PASS', cases=len(ids), residuals=residuals,
                historical_exact_zero_reclassified=False, historical_training_authorized=False)
