"""Production orchestration/JSON failure paths with expensive GPU work replaced."""
import copy
import json
import os
import signal
import subprocess
import tempfile
import unittest
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

from experiments.mamba3_absolute_phase import config as c, pipeline, provenance, report
from experiments.mamba3_absolute_phase.process_env import child_environment
from experiments.mamba3_mimo_time.records import create, read, sha, digest, Registry, accepted_cases


@contextmanager
def scratch():
    plan_text = (c.HERE / 'study_plan.json').read_text()
    with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
        root = Path(directory)
        here = root / 'study'
        here.mkdir()
        (here / 'study_plan.json').write_text(plan_text)
        values = dict(ROOT=root, HERE=here, LOGS=here / 'slurm_logs/attempt_001',
                      RUNS=here / 'runs/attempt_001', LAUNCHER=root / 'launch.sh',
                      MANIFEST=here / 'source_manifest.json', EXECUTION_ATTEMPT='001', STAGE='pilot', SEEDS=(2026,))
        for key in ('LOGIN', 'RESERVATION', 'SUBMISSION', 'PIPELINE', 'COVERAGE'):
            values[key] = values['LOGS'] / (key + '.json')
        for key in ('INHERITED', 'GATE', 'SMOKE', 'SUMMARY'):
            values[key] = values['RUNS'] / (key + '.json')
        for key, value in values.items():
            stack.enter_context(patch.object(c, key, value))
        yield root


def owner():
    return dict(execution_commit='a' * 40, source_hash='b' * 64,
                execution_attempt='001', study_phase='pilot', job_id='900001', test_evaluation_count=0,
                TEST='NOT_RUN', retry_reason='synthetic infrastructure fixture')


def cpu_proofs(base, good=True):
    for prefix in ('cpu_preflight_', 'no_git_preflight_'):
        create(c.LOGS / (prefix + base['execution_commit'] + '.json'),
               dict(execution_commit=base['execution_commit'], source_hash=base['source_hash'],
                    status='PASS' if good else 'FAIL', cuda_initialized=False))


def saved_gate(base, failed=False):
    value = dict(base, status='RUNNING')
    create(c.GATE, value)
    specs = c.plan()['required_cases']
    registry = Registry(c.GATE, value, specs)
    for i, spec in enumerate(specs):
        def compute(save, spec=spec, i=i):
            # Production callback/Registry/JSON path, synthetic CPU values only.
            save(dict(checks={}, required_keys=[], phase='setup'))
            checks = {key: dict(passed=not (failed and i == 0)) for key in spec['required_keys']}
            row = dict(checks=checks, required_keys=spec['required_keys'])
            save(row)
            return row
        try:
            registry.run(spec, compute)
        except RuntimeError:
            value['status'] = 'FAIL'
            break
    else:
        value['status'] = 'PASS'
    from experiments.mamba3_mimo_time.records import update
    update(c.GATE, value)


def saved_smoke(base, invalid=False):
    rows = []
    for mode in c.MODES:
        rows.append(dict(phase_mode=mode, status='PASS',
                         steps=[dict(step=i, loss=1.0, finite_loss=True,
                                     gradient_norms=dict.fromkeys(c.parameter_keys(mode), 1.0))
                                for i in range(3)],
                         roundtrip_passed=True, roundtrip='weights_only=True'))
    for row in rows:
        row.update(peak_allocated_bytes=1024, peak_reserved_bytes=2048)
        for step in row['steps']:
            step.update(W_before_l2=0. if row['phase_mode'] != 'baseline_dual' else None,
                        W_after_l2=.01, W_update_l2=.01, observer_removed=True, phase_layers=[
                dict(layer=i, correction_shape=[2048,50,32], finite_correction=True,
                     first_padding_neutral=True, native_angle_dtype='torch.float32') for i in range(2)])
    if invalid:
        rows[-1]['steps'][-1]['gradient_norms'].pop('phase_adapter.W')
    create(c.SMOKE, dict(base, status='PASS', batch=2048, history_length=50,
                         kernel_length=56, steps_per_mode=3, rows=rows,
                         targeted_gate_sha256=sha(c.GATE)))


def saved_control(base, historical):
    """Real saved-record validators run against a self-consistent tiny checkpoint."""
    paths = c.paths('baseline_dual')
    paths['checkpoint'].parent.mkdir(parents=True)
    paths['checkpoint'].write_bytes(b'opaque checkpoint bytes; tests never deserialize')
    checkpoint_sha = sha(paths['checkpoint'])
    fixture_reference = copy.deepcopy(historical)
    fixture_reference['checkpoint_sha256'] = checkpoint_sha
    reference_path = c.HERE / 'historical_fixture.json'
    create(reference_path, fixture_reference)
    record = copy.deepcopy(fixture_reference)
    record.update(base, phase_mode='baseline_dual', run_id=paths['run_id'],
                  initial_common_calibrator_hashes=record['initial_calibrator_hashes'],
                  initial_phase_parameters={}, scientific_fit_started=True,
                  checkpoint_path=str(paths['checkpoint']), checkpoint_metadata_path=str(paths['metadata']))
    for key in ('config', 'effective_config'):
        record[key].update(phase_mode='baseline_dual', checkpoint_dir=str(paths['checkpoint'].parent))
    metadata = {key: record[key] for key in
                ('run_id', 'mode', 'phase_mode', 'seed', 'execution_commit',
                 'config_sha256', 'source_hash', 'core_hash', 'checkpoint_sha256')}
    metadata.update(epoch=record['best_epoch'], metrics=record['best_valid_metrics'], phase=None)
    create(paths['metadata'], metadata)
    create(paths['result'], record)
    return reference_path


class ProtocolTests(unittest.TestCase):
    def test_plan_scope_and_initial_budget(self):
        plan = c.plan()
        self.assertEqual(plan['max_scientific_fits'], 15)
        self.assertEqual(plan['max_jobs'], 3)
        self.assertEqual(plan['test_evaluations'], 0)
        self.assertEqual(plan['periods_ms'], [21600000,86400000])
        self.assertEqual(plan['K'], 2)
        self.assertEqual(plan['pilot_seeds'], [2026])
        self.assertEqual(plan['confirmation_seeds'], [2027,2028,2029,2030])
        self.assertEqual(c.COUNTS, dict(baseline_dual=715020,relative_phase=715148,absolute_phase=715148))
        self.assertEqual(len(plan['required_cases']),17)
        self.assertEqual(sum(len(r['required_keys']) for r in plan['required_cases']),532)

    def test_child_environment_preserves_owned_gpu_mask(self):
        parent = {'CUDA_VISIBLE_DEVICES': 'GPU-owned', 'UNCHANGED': 'yes'}
        for stage in ('gate', 'smoke', *c.MODES):
            self.assertEqual(child_environment(stage, parent)['CUDA_VISIBLE_DEVICES'], 'GPU-owned')
        for stage in ('coverage', 'preflight'):
            self.assertEqual(child_environment(stage, parent)['CUDA_VISIBLE_DEVICES'], '')
        self.assertEqual(parent, {'CUDA_VISIBLE_DEVICES': 'GPU-owned', 'UNCHANGED': 'yes'})
        with self.assertRaises(ValueError):
            child_environment('TEST', parent)

    def test_production_child_saves_environment_without_changing_parent(self):
        with scratch(), patch.dict(os.environ, CUDA_VISIBLE_DEVICES='GPU-owned'):
            process = type('Process', (), {'wait': lambda self, timeout: 0})()
            with patch.object(subprocess, 'Popen', return_value=process) as launch:
                pipeline.child('preflight', 'fixture', [], 10**12, c.LOGS / 'child')
            evidence = read(c.LOGS / 'child/child_environment.json')
            self.assertEqual(launch.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'], '')
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], 'GPU-owned')
            self.assertNotEqual(evidence['parent_cuda_visibility'], evidence['child_cuda_visibility'])

    def test_adam_settings_real_json_roundtrip(self):
        import torch
        from experiments.mamba3_three_time.confirmation.state import canonical_optimizer_settings, compare_optimizer_settings
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.Adam([parameter], lr=.001)
        actual = [{key: value for key, value in group.items() if key != 'params'}
                  for group in optimizer.param_groups]
        restored = json.loads(json.dumps(canonical_optimizer_settings(actual), allow_nan=False))
        compare_optimizer_settings(actual, restored)
        changed = copy.deepcopy(restored)
        changed[0]['lr'] = .002
        with self.assertRaises(ValueError):
            compare_optimizer_settings(actual, changed)

    def test_cpu_config_and_reference_never_reopen_cuda(self):
        import torch
        from recbole.config import Config
        from experiments.mamba3_three_time.model import ThreeTimeMamba3Rec
        from experiments.mamba3_absolute_phase.state import effective_check
        self.assertEqual(os.environ.get('CUDA_VISIBLE_DEVICES'), '')
        self.assertFalse(torch.cuda.is_initialized())
        before = os.environ.get('CUDA_VISIBLE_DEVICES')
        with patch.object(torch.cuda, 'init', side_effect=AssertionError('CPU Config initialized CUDA')):
            for mode in c.MODES:
                config = Config(model=ThreeTimeMamba3Rec, config_dict=c.settings(mode, device='cpu'))
                checked = effective_check(config, mode)
                self.assertEqual(checked['status'], 'PASS')
                self.assertEqual(checked['cpu_reference_device'], 'cpu')
                self.assertEqual(str(config['device']), 'cpu')
                self.assertFalse(config['use_gpu'])
                self.assertEqual(config['gpu_id'], '')
        self.assertEqual(os.environ.get('CUDA_VISIBLE_DEVICES'), before)
        self.assertFalse(torch.cuda.is_initialized())

    def test_saved_partial_reason_does_not_collide_with_retry_reason(self):
        with scratch():
            base = owner()
            result = report.write(base, 'fixture failure before GPU')
            self.assertEqual(result['status'], 'INCOMPLETE')
            self.assertEqual(result['retry_reason'], base['retry_reason'])
            self.assertEqual(result['blocking_reason'], 'fixture failure before GPU')
            for mode in c.MODES:
                raw = read(c.paths(mode)['result'])
                self.assertEqual(raw['status'], 'NOT_RUN')
                self.assertEqual(raw['reason'], 'fixture failure before GPU')
                self.assertEqual(raw['retry_reason'], base['retry_reason'])
                self.assertFalse(raw['scientific_fit_started'])
            for contrast in result['contrasts']:
                self.assertEqual(contrast['status'], 'NOT_AVAILABLE')
                self.assertIsNone(contrast['delta'])
            self.assertEqual(read(c.SUMMARY), result)

    def _failure_pipeline(self, failure):
        with scratch(), ExitStack() as stack:
            base = owner()
            cpu_proofs(base, good=failure != 'before_gate')
            stack.enter_context(patch.object(pipeline, 'identity', return_value=base))
            stack.enter_context(patch.object(pipeline, 'inherited', return_value={'status': 'fixture'}))
            stack.enter_context(patch.object(pipeline, 'runtime', return_value={'status': 'fixture'}))
            stack.enter_context(patch.dict(os.environ, CUDA_VISIBLE_DEVICES='GPU-owned', SLURM_JOB_END_TIME='999999999999'))
            # Keep production signal handling from changing this test process.
            stack.enter_context(patch.object(signal, 'signal'))
            stages = []
            def work(stage, *_args):
                stages.append(stage)
                if stage == 'gate':
                    saved_gate(base, failed=failure == 'inside_gate')
                elif stage == 'smoke':
                    saved_smoke(base, invalid=failure == 'smoke')
                else:
                    raise AssertionError('Scientific fit must not be reached')
            stack.enter_context(patch.object(pipeline, 'child', side_effect=work))
            self.assertEqual(pipeline.run(), 1)
            value, summary = read(c.PIPELINE), read(c.SUMMARY)
            expected = [] if failure == 'before_gate' else ['gate'] if failure == 'inside_gate' else ['gate', 'smoke']
            self.assertEqual(stages, expected)
            self.assertEqual([r['stage'] for r in value['stages']], expected)
            self.assertEqual(value['status'], 'FAIL')
            self.assertEqual(summary['status'], 'INCOMPLETE')
            self.assertEqual(value['scientific_fits_started'], 0)
            self.assertEqual(value['scientific_fits_completed'], 0)
            self.assertEqual(value['unknown_scientific_starts'], 0)
            self.assertNotIn('report_traceback', value)
            terminal = read(c.RUNS / 'terminal_metadata.json')
            self.assertEqual(terminal['pipeline_sha256'], sha(c.PIPELINE))
            self.assertEqual(terminal['checkpoints'], {})
            self.assertFalse(terminal['checkpoint_loading'])

    def test_failure_before_gate_preserves_not_run_summary(self):
        self._failure_pipeline('before_gate')

    def test_failure_inside_gate_preserves_not_run_summary(self):
        self._failure_pipeline('inside_gate')

    def test_failure_in_smoke_blocks_every_scientific_fit(self):
        self._failure_pipeline('smoke')

    def test_budget_expires_before_next_fit_retains_completed_control(self):
        historical = read(c.PILOT)
        with scratch(), ExitStack() as stack:
            base, clock = owner(), [1000.0]
            cpu_proofs(base)
            stack.enter_context(patch.object(pipeline, 'identity', return_value=base))
            stack.enter_context(patch.object(pipeline, 'inherited', return_value={'status': 'fixture'}))
            stack.enter_context(patch.object(pipeline, 'runtime', return_value={'status': 'fixture'}))
            stack.enter_context(patch.object(pipeline.time, 'time', side_effect=lambda: clock[0]))
            stack.enter_context(patch.object(signal, 'signal'))
            stack.enter_context(patch.dict(os.environ, SLURM_JOB_END_TIME='999999999999'))
            stages = []
            def work(stage, *_args):
                stages.append(stage)
                if stage == 'gate':
                    saved_gate(base)
                elif stage == 'smoke':
                    saved_smoke(base)
                elif stage == 'baseline_dual':
                    reference = saved_control(base, historical)
                    stack.enter_context(patch.object(c, 'PILOT', reference))
                    clock[0] = float(os.environ['PIPELINE_DEADLINE']) - c.plan()['budget']['min_remaining_to_start_fit_seconds'] + 1
                else:
                    raise AssertionError('No next fit may start after the time budget expires')
            stack.enter_context(patch.object(pipeline, 'child', side_effect=work))
            self.assertEqual(pipeline.run(), 1)
            self.assertEqual(stages, ['gate', 'smoke', 'baseline_dual'])
            value, summary = read(c.PIPELINE), read(c.SUMMARY)
            self.assertEqual(value['status'], 'INCOMPLETE')
            self.assertEqual(value['scientific_fits_started'], 1)
            self.assertEqual(value['scientific_fits_completed'], 1)
            self.assertEqual(summary['rows'][0]['status'], 'PASS')
            self.assertEqual([r['status'] for r in summary['rows'][1:]], ['NOT_RUN', 'NOT_RUN'])
            self.assertNotIn('report_traceback', value)
            self.assertEqual(set(read(c.RUNS / 'terminal_metadata.json')['checkpoints']), {c.paths('baseline_dual')['run_id']})

    def test_startup_identity_failure_is_saved_without_submission(self):
        with scratch(), patch.dict(os.environ, SLURM_JOB_ID='900001'), patch.object(pipeline, 'identity', side_effect=ValueError('owner mismatch')):
            self.assertEqual(pipeline.main(), 1)
            value = read(c.LOGS / 'startup_failure_900001.json')
            self.assertFalse(value['identity_verified'])
            self.assertEqual(value['scientific_fits_started'], 0)
            self.assertEqual(value['test_evaluation_count'], 0)
            self.assertFalse(c.PIPELINE.exists())

    def test_fit_counters_fail_closed_on_unknown_locked_record(self):
        with scratch():
            paths = c.paths('relative_phase')
            create(paths['lock'], {})
            self.assertEqual(pipeline.fit_counters()['unknown_scientific_starts'], 1)
            create(paths['result'], {'scientific_fit_started': True, 'status': 'INCOMPLETE'})
            value = pipeline.fit_counters()
            self.assertEqual(value, dict(scientific_fits_started=1, scientific_fits_completed=0, unknown_scientific_starts=0))

    def test_ownership_binds_identical_valid_coverage_hashes(self):
        expected = dict(execution_commit='a' * 40, source_hash='b' * 64)
        coverage_sha = 'c' * 64
        login = dict(expected, status='PASS', tracked_clean=True, source_blobs_verified=True,
                     published_commit=expected['execution_commit'], coverage_sha256=coverage_sha)
        reservation = dict(expected, status='RESERVED', token='d' * 32, login_sha256='e' * 64,
                           coverage_sha256=coverage_sha, max_scientific_fits=3,
                           jobs_requested=1, tasks=c.tasks(), requested_seconds=c.allocation_seconds())
        provenance.validate_ownership(login, reservation, 'e' * 64, expected, '900001')
        for field in ('login', 'reservation'):
            for changed_hash in ('f' * 64, '', None, 'not-a-sha'):
                changed_login, changed_reservation = copy.deepcopy(login), copy.deepcopy(reservation)
                (changed_login if field == 'login' else changed_reservation)['coverage_sha256'] = changed_hash
                with self.subTest(field=field, coverage_sha256=changed_hash), self.assertRaises(ValueError):
                    provenance.validate_ownership(changed_login, changed_reservation, 'e' * 64, expected, '900001')
        # Equality alone is insufficient: both missing/malformed pointers fail.
        for malformed in ('', None, 'same-but-not-a-sha'):
            with self.subTest(both=malformed), self.assertRaises(ValueError):
                provenance.validate_ownership(dict(login, coverage_sha256=malformed),
                                              dict(reservation, coverage_sha256=malformed),
                                              'e' * 64, expected, '900001')

    def test_identity_rejects_changed_coverage_bytes_after_reservation(self):
        from experiments.mamba3_mimo_time.records import update
        with scratch(), ExitStack() as stack:
            expected = dict(execution_commit='a' * 40, source_hash='b' * 64)
            token, job = 'd' * 32, '900001'
            create(c.COVERAGE, dict(status='PASS', fixture='original coverage bytes'))
            admitted_sha = sha(c.COVERAGE)
            create(c.LOGIN, dict(expected, status='PASS', tracked_clean=True, source_blobs_verified=True,
                                 published_commit=expected['execution_commit'], coverage_sha256=admitted_sha))
            create(c.RESERVATION, dict(expected, status='RESERVED', token=token, login_sha256=sha(c.LOGIN),
                                       coverage_sha256=admitted_sha, max_scientific_fits=3,
                                       jobs_requested=1, tasks=c.tasks(), requested_seconds=c.allocation_seconds()))
            create(c.SUBMISSION, dict(token=token, job_id=job, status='SUBMITTED'))
            stack.enter_context(patch.dict(os.environ, RUN_COMMIT=expected['execution_commit'],
                                            EXPECTED_STUDY_HASH=expected['source_hash'],
                                            RESERVATION_TOKEN=token, SLURM_JOB_ID=job))
            stack.enter_context(patch.object(provenance, 'verify', return_value={'source_hash': expected['source_hash']}))
            stack.enter_context(patch.object(provenance, 'bindings', return_value=expected))
            # Isolate the ownership binding from the expensive coverage-content
            # audit while still hashing the actual immutable proof bytes.
            stack.enter_context(patch.object(provenance, 'coverage_verify', side_effect=lambda commit: sha(c.COVERAGE)))
            self.assertEqual(provenance.identity()['coverage_sha256'], admitted_sha)
            update(c.COVERAGE, dict(status='PASS', fixture='changed but independently valid coverage bytes'))
            self.assertNotEqual(sha(c.COVERAGE), admitted_sha)
            with self.assertRaises(ValueError):
                provenance.identity()

    def test_confirmation_tasks_preserve_seed_and_mode_order(self):
        with patch.object(c, 'STAGE', 'confirmation'), patch.object(c, 'SEEDS', (2027,2028,2029,2030)):
            tasks = c.tasks()
            self.assertEqual([(x['variant'],x['seed']) for x in tasks],
                             [(mode,seed) for seed in (2027,2028,2029,2030) for mode in c.MODES])
            self.assertEqual(len({x['run_id'] for x in tasks}),12)
            self.assertTrue(all(str(x['seed']) in x['run_id'] for x in tasks))
            self.assertEqual(c.allocation_seconds(),28800)
            with patch.object(c, 'EXECUTION_ATTEMPT', '002'):
                self.assertEqual(c.allocation_seconds(),14400)

    def test_real_child_passes_explicit_variant_seed_and_scheduler_mask(self):
        with scratch(), patch.dict(os.environ, CUDA_VISIBLE_DEVICES='GPU-owned'):
            process = type('Process', (), {'wait': lambda self, timeout: 0})()
            args = ['--variant','absolute_phase','--seed','2029']
            with patch.object(subprocess, 'Popen', return_value=process) as launch:
                pipeline.child('absolute_phase','experiments.mamba3_absolute_phase.runner',args,10**12,c.LOGS/'child')
            self.assertEqual(launch.call_args.args[0][-4:],args)
            self.assertEqual(launch.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'],'GPU-owned')
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'],'GPU-owned')

    def test_compute_source_verification_without_git_or_process_launch(self):
        with scratch(), patch.dict(os.environ, PATH=''), patch.object(subprocess,'Popen',side_effect=AssertionError('compute verification attempted an external process')):
            file = c.ROOT/'frozen_fixture.py'
            file.write_text('value = 1\n')
            files = {'frozen_fixture.py':sha(file)}
            manifest = dict(files=files,source_hash=digest(files))
            create(c.MANIFEST,manifest)
            self.assertEqual(provenance.verify(),manifest)
            file.write_text('value = 2\n')
            with self.assertRaisesRegex(ValueError,'Source mismatch'):
                provenance.verify()

    def test_smoke_rejects_missing_or_negative_required_phase_evidence(self):
        with scratch():
            saved_gate(owner())
            saved_smoke(owner())
            good = read(c.SMOKE)
            provenance.validate_smoke(good)
            for key in ('observer_removed','phase_layers'):
                broken = copy.deepcopy(good)
                del broken['rows'][1]['steps'][0][key]
                with self.subTest(missing=key),self.assertRaises((ValueError,KeyError,TypeError)):
                    provenance.validate_smoke(broken)
            for key,value in [('finite_correction',False),('first_padding_neutral',False),
                              ('native_angle_dtype','torch.bfloat16'),('correction_shape',[1,50,32])]:
                broken = copy.deepcopy(good)
                broken['rows'][1]['steps'][0]['phase_layers'][0][key]=value
                with self.subTest(phase_key=key),self.assertRaises((ValueError,KeyError,TypeError)):
                    provenance.validate_smoke(broken)

    def test_budget_expires_before_first_fit_keeps_everything_not_run(self):
        with scratch(), ExitStack() as stack:
            base,clock=owner(),[1000.]
            cpu_proofs(base)
            stack.enter_context(patch.object(pipeline,'identity',return_value=base))
            stack.enter_context(patch.object(pipeline,'inherited',return_value={'status':'fixture'}))
            stack.enter_context(patch.object(pipeline,'runtime',return_value={'status':'fixture'}))
            stack.enter_context(patch.object(pipeline.time,'time',side_effect=lambda:clock[0]))
            stack.enter_context(patch.object(signal,'signal'))
            stack.enter_context(patch.dict(os.environ,SLURM_JOB_END_TIME='999999999999'))
            seen=[]
            def child(stage,*args):
                seen.append(stage)
                if stage=='gate':saved_gate(base)
                elif stage=='smoke':
                    saved_smoke(base)
                    clock[0]=float(os.environ['PIPELINE_DEADLINE'])-c.plan()['budget']['min_remaining_to_start_fit_seconds']+1
                else:raise AssertionError('No scientific child permitted')
            stack.enter_context(patch.object(pipeline,'child',side_effect=child))
            self.assertEqual(pipeline.run(),1)
            self.assertEqual(seen,['gate','smoke'])
            status,summary=read(c.PIPELINE),read(c.SUMMARY)
            self.assertEqual(status['status'],'INCOMPLETE')
            self.assertEqual(summary['scientific_fits_started'],0)
            self.assertTrue(all(r['status']=='NOT_RUN' for r in summary['rows']))
            self.assertIn('Insufficient allocation',summary['blocking_reason'])
            self.assertNotIn('report_traceback',status)

    def test_replay_rejects_scientific_drift_but_distinguishes_serialization(self):
        value = read(c.PILOT)
        value['initial_common_calibrator_hashes'] = value['initial_calibrator_hashes']
        self.assertEqual(report.replay_check(value)['status'], 'PASS')
        changed = copy.deepcopy(value)
        changed['history'][0]['train_loss'] += .01
        with self.assertRaisesRegex(ValueError, 'scientific drift'):
            report.replay_check(changed)
        changed = copy.deepcopy(value)
        changed['checkpoint_sha256'] = 'a'*64
        review = report.replay_check(changed)
        self.assertEqual(review['status'], 'CHECKPOINT_IDENTITY_REVIEW_REQUIRED')
        self.assertTrue(review['scientific_history_exact'])
        self.assertFalse(review['checkpoint_serialization_match'])



if __name__ == '__main__':
    unittest.main()
