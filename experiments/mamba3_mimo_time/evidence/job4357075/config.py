"""Prospective fixed scope, paths and explicit required-case registry."""
import copy
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MODES = ('base', 'dual', 'triple')
COUNTS = dict(base=714888, dual=715020, triple=715086)
STUDY = 'mamba3_mimo_time_pilot_001'
BRANCH = 'exp/mamba3-mimo-time'
PUBLICATION = 'a462c412ef72dc0fa8af1ec6b48d1eef47ba8e72'
HISTORICAL = '995c5cde6449ea429c1d80ca6ab276b9791041c0'
CORE = '460bd3e687d26a458cafc21960391f83905da962ca5c992d6062c7b048f0a73f'
PIN = 'e9594ce1c732d97440f0332fdc43170a2294dbfa'
PILOT = ROOT / 'experiments/mamba3_three_time/validation_pilot/runs/mamba3_three_time_siso_dual_seed2026_001.json'
EXECUTION_ATTEMPT = '002'
PARENT_JOB = '4356310'
PARENT_COMMIT = 'f4938792ab3415e431320d5ca7c2d6ce042ab35a'
PARENT_SOURCE = '52c903ac4d2212ae8d81645079e424265b30c0f0733a3b48ec70571de8f9fc12'
PARENT_EVIDENCE_SHA = '5c51d7628c109f8011d58bc3a629d3e52a1ca0aa78f2caecda1f6b242356e672'
PARENT_EVIDENCE = HERE / ('evidence/job' + PARENT_JOB)
LOGS, RUNS = (HERE / n / ('attempt_' + EXECUTION_ATTEMPT) for n in ('slurm_logs', 'runs'))
GATE, SMOKE, SUMMARY = (RUNS / n for n in ('admission_001.json', 'smoke_001.json', 'pilot_summary.json'))
LOGIN, RESERVATION = LOGS / 'login_verification_001.json', LOGS / 'reservation_001.json'
SUBMISSION, PIPELINE = LOGS / 'submission_001.json', LOGS / 'pipeline_status.json'
POLICY = HERE / 'mimo_numeric_acceptance_v1.json'
ACCEPTED = 'ACCEPTED_FOR_MIMO_PILOT_WITH_DOCUMENTED_NUMERICAL_LIMITATIONS'
LENGTHS = (1, 7, 8, 9, 15, 16, 17, 50, 65)


def paths(mode):
    if mode not in MODES:
        raise ValueError(mode)
    name = f'mamba3_mimo_{mode}_seed2026_001'
    runtime = LOGS / name
    return dict(run_id=name, runtime=runtime, result=RUNS / (name + '.json'), lock=runtime / 'run.lock',
                checkpoint=runtime / 'checkpoints/best_state_dict.pth', metadata=runtime / 'checkpoints/best_metadata.json')


def settings(mode, device=None):
    values = copy.deepcopy(json.loads(PILOT.read_text())['config'])
    values.update(three_time_mode=mode, mamba3_is_mimo=True, mamba3_mimo_rank=4,
                  mamba3_chunk_size=8, checkpoint_dir=str(paths(mode)['checkpoint'].parent))
    if device is not None:
        values.update(use_gpu=device == 'cuda', device=device)
        if device == 'cpu':
            values['gpu_id'] = ''
    return values


def case_specs():
    specs = [dict(id='runtime_initialization', suite='initialization'), dict(id='native_L16', suite='native', length=16)]
    specs += [dict(id=f'boundary_L{n}', suite='boundary', length=n) for n in LENGTHS]
    specs += [dict(id=f'structural_{m}_{t}_L{n}', suite='structural', mode=m, training=t, length=n)
              for m in ('base', 'dual') for t in (False, True) for n in (17, 50)]
    specs += [dict(id='tied_slots', suite='tied_slots'), dict(id='tied_calibrators', suite='tied_calibrators')]
    specs += [dict(id=f'reference_seed{s}_L{n}', suite='reference', seed=s, length=n)
              for s, ns in ((2026, (15, 50)), (314159, (7, 8, 9, 15, 50))) for n in ns]
    specs += [dict(id=f'prefix_{m}_L{n}_P{p}_x{x}', suite='prefix', mode=m, length=n, prefix=p, multiplier=x)
              for m in MODES for n, p in ((17, 7), (50, 25)) for x in (1, 16)]
    specs += [dict(id='D_oracle', suite='D_oracle'), dict(id='masks_zero_gaps', suite='edges')]
    specs += [dict(id=f'optimizer_{m}', suite='optimizer', mode=m) for m in ('dual', 'triple')]
    return specs + [dict(id='state_dict_roundtrip', suite='roundtrip')]


def plan():
    result = json.loads((HERE / 'study_plan.json').read_text())
    tasks = [dict(mode=m, seed=2026, run_id=paths(m)['run_id']) for m in MODES]
    specs = [dict(s, required_keys=planned_checks(s, result['parameter_keys'])) for s in case_specs()]
    if result['tasks'] != tasks or result['required_cases'] != specs:
        raise ValueError('Frozen scope/registry mismatch')
    return result


def planned_checks(spec, parameters):
    kernel = ['output', 'loss'] + ['gradient:' + k for k in
              ('q','k','v','adt','dw','trap','qb','kb','angles','d','z','mv','mz','mo')]
    model = lambda m: ['output','loss','input_gradient'] + ['gradient:'+n for n in parameters[m]]
    suite = spec['suite']
    if suite == 'initialization':
        keys = ['initialization']
    elif suite in ('native', 'tied_slots'):
        keys = kernel
    elif suite == 'boundary':
        keys = kernel + [b+':crop:'+k for b in ('official','local') for k in kernel if k != 'loss']
        keys += [b+':no_tail_loss' for b in ('official','local')]
    elif suite == 'structural':
        keys = model(spec['mode'])
    elif suite == 'tied_calibrators':
        keys = model('dual') + ['independent_parameter_objects']
    elif suite == 'reference':
        keys = []
        for tied in (True, False):
            measurements = kernel + ([] if tied else ['gradient:dp'])
            for backend in (('official','local') if tied else ('local',)):
                keys += [f'{tied}/{kind}/{backend}/{k}' for kind in ('signed','nonnegative') for k in measurements if k != 'loss']
                keys += [f'{tied}/{backend}/legacy_loss_mixed']
                keys += [f'{tied}/{backend}/legacy_finite/{k}' for k in measurements]
    elif suite == 'prefix':
        leaves = ['residual','cross_user_gradient','finite_output','prefix_intervention0',
                  'prefix_intervention1','cross_user_intervention0','cross_user_intervention1']
        keys = [b+':'+k for b in (('local',) if spec['mode']=='triple' else ('local','official')) for k in leaves]
    elif suite == 'D_oracle':
        keys = [kind+':reference_analytic_D' for kind in ('signed','nonnegative')]
        keys += [kind+'/'+b+'/'+k for kind in ('signed','nonnegative') for b in ('official','local')
                 for k in [x for x in kernel if x!='loss']+['analytic_D']]
    elif suite == 'edges':
        keys = [label+':'+k for label in ('padded','length1','zero_gap') for k in
                ['finite','neutral0','neutral1','neutral2']+['gradient:'+p for p in parameters['triple']]]
        keys += ['real_zero_gap_active','padding_timestamp_ignored','target_timestamp_ignored']
    elif suite == 'optimizer':
        keys = [f'finite_gradients_step{s}' for s in range(3)]
        keys += ['learnable:'+p[len('times.'):] for p in parameters[spec['mode']] if p.startswith('times.')]
    elif suite == 'roundtrip':
        keys = ['state_dict','output']
    else:
        raise ValueError(suite)
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate required leaf')
    return sorted(keys)


def unused():
    candidates = [LOGIN, RESERVATION, SUBMISSION, GATE, SMOKE, SUMMARY, SUMMARY.with_suffix('.md'),
                  LOGS / 'pipeline.lock', PIPELINE]
    for mode in MODES:
        candidates += [v for k, v in paths(mode).items() if k in ('result', 'lock', 'checkpoint', 'metadata')]
    occupied = [str(p) for p in candidates if p.exists() or p.is_symlink()]
    if occupied:
        raise FileExistsError('Already reserved/executed; no automatic retry: ' + repr(occupied))


def freeze_plan():
    from .records import create
    unused()
    old = json.loads((ROOT / 'experiments/mamba3_three_time/evidence/attempt_003/runs/mimo_correctness_003.json').read_text())
    parameters = {}
    for mode in MODES:
        row = next(r for r in old['cases'] if r['case_id'] == f'local_gate_{mode}/MIMO/upstream_with_adapter/L17/Pnone/x1')
        parameters[mode] = sorted(k[len('gradient:'):] for k in row['checks'] if k.startswith('gradient:'))
    value = dict(study_id=STUDY, tasks=[dict(mode=m, seed=2026, run_id=paths(m)['run_id']) for m in MODES],
                 parameter_keys=parameters, parameter_counts=COUNTS,
                 required_cases=[dict(s, required_keys=planned_checks(s, parameters)) for s in case_specs()],
                 budget=dict(allocation_seconds=28800, save_margin_seconds=600, min_remaining_to_start_fit_seconds=5400),
                 limits=dict(jobs=1, gpus=1, max_scientific_fits=3, test_evaluations=0),
                 resources=dict(partition='rocky', account='proj_1833', constraint='type_e', gpu='a100:1', cpus=4, mem=0, time='08:00:00', requeue=False),
                 initialization_order=['runtime/config','seed','dataset/loaders','model','trainer','fit'],
                 effective_config_source=str(PILOT.relative_to(ROOT)), reference_gap_ms=838393,
                 evaluation='VALID full-ranking; chronological leave-one-out; TEST NOT_RUN',
                 selection='Last tied maximum rounded VALID NDCG@10; unchanged RecBole.fit',
                 policy_version='mimo_numeric_acceptance_v1', architecture='MIMO rank4/chunk8; two layers/two temporal heads',
                 smoke=dict(batch=2048, input_length=50, kernel_length=56, optimizer_steps_per_mode=2),
                 fits_from_scratch=True, automatic_retries=False, thresholds_change_after_gate=False)
    create(HERE / 'study_plan.json', value)
    return value


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze-plan', action='store_true', required=True)
    parser.parse_args()
    print(json.dumps({'study':freeze_plan()['study_id']}))
