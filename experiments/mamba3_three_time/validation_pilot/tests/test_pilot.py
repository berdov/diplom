"""CPU-only policy, ownership, initialization and trainer contract tests."""
import ast
import copy
import inspect
import json
import math
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from recbole.data.interaction import Interaction
from experiments.mamba3_three_time.records_003 import Registry, required_pass
from experiments.mamba3_three_time.evidence import create_record
from experiments.mamba3_context_time.trainer import trainer_class as safe_trainer
from experiments.mamba3_three_time.validation_pilot import config, historical, runner, pipeline, submit, gate
from experiments.mamba3_three_time.validation_pilot.policy import EPS32, from_norms, positional
from experiments.mamba3_three_time.validation_pilot.preflight import initialization
from experiments.mamba3_three_time.validation_pilot.provenance import rng_hash, save_state_dict, sha
from experiments.mamba3_three_time.validation_pilot.diagnostics import Diagnostics
from experiments.mamba3_three_time.validation_pilot.trainer import trainer_class


class PolicyTests(unittest.TestCase):
    def test_zero_signal_not_pass(self):
        r = from_norms(0, 0)
        self.assertEqual(r['status'], 'ZERO_SIGNAL')
        self.assertFalse(r['finite_precision_pass'])
        self.assertFalse(r['informative'])

    def test_future_only_fails(self):
        self.assertEqual(from_norms(0, 1e-30)['rho'], 1)
        self.assertFalse(from_norms(0, 1e-30)['finite_precision_pass'])

    def test_small_nonzero_tail_and_scale_invariance(self):
        rows = [from_norms(2 * s, 1e-8 * s) for s in (1e-25, 1, 16)]
        for row in rows:
            self.assertTrue(row['finite_precision_pass'])
            self.assertAlmostEqual(row['rho'], rows[0]['rho'])
        self.assertFalse(from_norms(1, 1e-5)['finite_precision_pass'])

    def test_frozen_boundary(self):
        self.assertTrue(from_norms(1-EPS32, EPS32)['finite_precision_pass'])
        self.assertFalse(from_norms(1-EPS32, 2*EPS32)['finite_precision_pass'])

    def test_nonfinite_and_negative_rejected(self):
        for p, f in ((math.nan, 0), (1, math.inf), (-1, 0), (1, -1), (1e308, 1e308)):
            self.assertFalse(from_norms(p, f)['finite_precision_pass'])
        r = positional(torch.tensor([[[1.], [math.inf]]]), 1)
        self.assertIsNone(r['future_l2'])
        json.dumps(r, allow_nan=False)

    def test_positional_fp64_no_mutation(self):
        g = torch.tensor([[[1., 2.], [1e-8, -1e-8]]])
        before = g.clone()
        row = positional(g, 1)
        self.assertTrue(torch.equal(g, before))
        self.assertFalse(row['legacy_exact_zero'])
        self.assertTrue(row['finite_precision_pass'])
        self.assertEqual(row['nonzero_count'], 2)
        self.assertEqual(row['norm_dtype'], 'float64')
        self.assertAlmostEqual(positional(g*16, 1)['rho'], row['rho'])
        with self.assertRaises(ValueError): positional(g, 2)


class Contracts(unittest.TestCase):
    def test_new_fixtures_actual_seed_and_no_rng_drift(self):
        before = rng_hash()
        a = gate.histories(50, device='cpu')
        b = gate.histories(50, device='cpu', seed=2026)
        self.assertFalse(torch.equal(a[0], b[0]))
        self.assertTrue(torch.equal(a[0], gate.histories(50, device='cpu')[0]))
        self.assertEqual(before, rng_hash())
        items, lengths, times = gate.histories(17, device='cpu', padded=True, zero_gap=True)
        self.assertEqual(lengths[-1], 10)
        self.assertTrue(bool((items[-1, 10:] == 0).all()))
        self.assertTrue(bool((times == times[:, :1]).all()))

    def test_config_counts_rng_and_independent_calibrators(self):
        row = initialization('cpu')
        self.assertTrue(row['passed'], row)
        self.assertEqual([r['parameters'] for r in row['rows']], [610572, 610638])
        self.assertFalse(torch.cuda.is_initialized())

    def test_nested_failure_blocks_registry_and_persists(self):
        result, snapshots = {}, []
        registry = Registry(result, lambda: snapshots.append(copy.deepcopy(result)), ['a'])
        with self.assertRaises(RuntimeError):
            registry.run('a', lambda: dict(passed=True, hidden=dict(passed=False)), gate=True)
        self.assertEqual(snapshots[-1]['cases'][0]['status'], 'FAIL')
        self.assertFalse(registry.close())
        self.assertFalse(required_pass(dict(status='INCONCLUSIVE')))
        with self.assertRaises(ValueError): Registry({}, lambda: None, ['a', 'a'])

    def test_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'old.json'
            create_record(p, dict(old=True))
            before = sha(p)
            with self.assertRaises(FileExistsError): create_record(p, dict(old=False))
            self.assertEqual(sha(p), before)
            with patch.object(config, 'GATE', p):
                with self.assertRaises(FileExistsError): config.unused()

    def test_historical_raw_scope_immutable(self):
        p = config.PARENT / 'evidence/attempt_004/runs/siso_diagnostics_004.json'
        before = sha(p)
        raw = json.loads(p.read_text())
        old = copy.deepcopy(raw)
        row = historical.audit(raw)
        self.assertEqual(row['cases'], 53)
        self.assertEqual(len(row['residuals']), 30)
        self.assertEqual(raw, old)
        self.assertEqual(sha(p), before)
        self.assertFalse(row['historical_exact_zero_reclassified'])
        raw['cases'][0]['checks']['injected_failure'] = dict(passed=False)
        with self.assertRaises(ValueError): historical.audit(raw)

    def test_historical_missing_case_blocked(self):
        raw = json.loads((config.PARENT / 'evidence/attempt_004/runs/siso_diagnostics_004.json').read_text())
        raw['cases'].pop()
        with self.assertRaises(ValueError): historical.audit(raw)

    def test_required_gate_coverage(self):
        ids = gate.expected_ids()
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len([k for k in ids if k.startswith('prefix_')]), 12)
        self.assertIn('C_leaf_gradient_sum', ids)
        self.assertIn('VJP_L50', ids)
        self.assertEqual(config.plan()['max_scientific_fits'], 2)
        with self.assertRaises(ValueError): config.paths('MIMO')

    def test_diagnostics_heads_and_rng(self):
        before = rng_hash()
        d = Diagnostics('dual')
        scale = torch.ones(2, 5, 2)
        active = torch.ones(2, 5, dtype=torch.bool)
        d.hook(None, (None, active), (scale, scale, scale))
        row = d.result()
        self.assertTrue(row['write_equals_phase'])
        self.assertEqual(row['temporal_heads'], 2)
        self.assertEqual(set(row['scales']), {'decay', 'write', 'phase'})
        self.assertEqual(before, rng_hash())

    def test_validation_only_guard_and_fit_not_overridden(self):
        valid = object()
        class Base:
            def evaluate(self, data, **kwargs): return 'VALID'
            def fit(self, *args): return 'unchanged'
        cls = trainer_class(Base, valid, {}, {})
        instance = cls()
        self.assertIs(cls.fit, Base.fit)
        self.assertEqual(instance.evaluate(valid), 'VALID')
        for kwargs in ({}, dict(load_best_model=True), dict(model_file='checkpoint')):
            with self.assertRaises(ValueError): instance.evaluate(object(), **kwargs)
        with self.assertRaises(ValueError): instance.evaluate(valid, load_best_model=True)

    def test_actual_consumed_first_batch_and_no_peek(self):
        with tempfile.TemporaryDirectory() as directory:
            record = {}
            data = Interaction({'item_id': torch.tensor([3, 4])})
            class Base:
                def _train_epoch(self, loader, epoch, loss, show_progress):
                    return loss(next(iter(loader)))
            cls = safe_trainer(Base, object(), record, {'result': Path(directory) / 'result.json'})
            obj = cls()
            calls = []
            obj.model = types.SimpleNamespace(calculate_loss=lambda x: calls.append(x) or 1.)
            self.assertNotIn('first_train_batch_sha256', record)
            self.assertEqual(obj._train_epoch(iter([data]), 0), 1.)
            self.assertEqual(len(calls), 1)
            self.assertIn('first_train_batch_sha256', record)

    def test_safe_checkpoint_and_last_equal_epoch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            p = dict(checkpoint=root/'state.pth', metadata=root/'meta.json', result=root/'run.json')
            model = torch.nn.Linear(2, 3)
            record = dict(mode='dual', seed=2026, run_id='fixture', execution_commit='fixture',
                          config_sha256='fixture', source_hash='fixture', core_hash='fixture', history=[])
            cls = safe_trainer(object, object(), record, p)
            obj = cls(); obj.model = model; obj.best_valid_score = .1
            for epoch in (2, 3):
                record['history'].append(dict(epoch=epoch, valid_ndcg10=.1, valid_metrics={'ndcg@10': .1}, diagnostics={}))
                obj._save_checkpoint(epoch)
            self.assertEqual(record['best_epoch'], 3)
            restored = torch.load(p['checkpoint'], weights_only=True, map_location='cpu')
            self.assertTrue(all(torch.equal(v, restored[k]) for k, v in model.state_dict().items()))
            self.assertEqual(record['checkpoint_sha256'], sha(p['checkpoint']))

    def test_no_scheduler_in_pipeline_no_test_loader_no_unsafe_load(self):
        literals = [n.value for n in ast.walk(ast.parse(inspect.getsource(pipeline))) if isinstance(n, ast.Constant)]
        self.assertNotIn('sbatch', literals)
        source = inspect.getsource(runner)
        self.assertIn('del reserved', source)
        self.assertEqual(source.count('FullSortEvalDataLoader(config,'), 1)
        self.assertNotIn('next(iter(', source)
        self.assertNotIn('weights_only=False', source)
        self.assertEqual(inspect.getsource(submit).count("['sbatch', '--parsable'"), 1)

    def test_admission_failure_marks_both_not_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stages = []
            def child(module, args, path, deadline):
                stages.append(module.rsplit('.', 1)[-1])
                if stages[-1] == 'gate':
                    raise RuntimeError('injected admission failure')
            def p(mode): return dict(result=root/(mode+'.json'))
            base = dict(job_id='cpu-fixture', TEST='NOT_RUN', test_evaluation_count=0)
            with patch.multiple(pipeline, LOGS=root, LOCK=root/'pipeline.lock', GATE=root/'gate.json',
                                SMOKE=root/'smoke.json', SUMMARY=root/'summary.json'), \
                 patch.object(pipeline, 'identity', return_value=base), \
                 patch.object(pipeline, 'require_reservation'), patch.object(pipeline, 'unused'), \
                 patch.object(pipeline, 'run_child', side_effect=child), patch.object(pipeline, 'paths', side_effect=p), \
                 patch.dict('os.environ'), patch.object(pipeline.signal, 'signal'), \
                 patch.object(pipeline.traceback, 'print_exc'):
                with self.assertRaises(SystemExit) as error: pipeline.main()
                self.assertEqual(error.exception.code, 1)
            self.assertEqual(stages, ['gate', 'report'])
            for mode in config.MODES:
                row = json.loads(p(mode)['result'].read_text())
                self.assertEqual(row['status'], 'NOT_RUN')
                self.assertFalse(row['scientific_fit_started'])
            self.assertEqual(json.loads((root/'pipeline_status.json').read_text())['scientific_fits'], 0)


if __name__ == '__main__':
    unittest.main()
