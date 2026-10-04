"""The production callback → Registry → JSON → acceptance chain."""
import json
import tempfile
import unittest
from pathlib import Path
from experiments.mamba3_mimo_time.records import Registry, accepted_cases, create, read
from experiments.mamba3_absolute_phase.gate import run_required_case


class GateContractTests(unittest.TestCase):
    def invoke(self, keys, function, fails=False):
        specs = [dict(id='fixture', required_keys=keys)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'gate.json'
            value = dict(status='RUNNING')
            create(path, value)
            registry = Registry(path, value, specs)
            if fails:
                with self.assertRaises(RuntimeError):
                    run_required_case(registry, specs[0], function)
            else:
                run_required_case(registry, specs[0], function)
            result = read(path)
            self.assertEqual(result, json.loads(json.dumps(value, allow_nan=False)))
            self.assertEqual(accepted_cases(result['cases'], specs), not fails)
            return result['cases'][0]

    def test_empty_progress_running_then_real_pass(self):
        def check(save):
            save(dict(checks={}, required_keys=[], stage='not a verdict'))
            save(dict(checks={'x': {'passed': True}}, required_keys=['x']))
            return dict(checks={'x': {'passed': True}}, required_keys=['x'])
        result = self.invoke(['x'], check)
        self.assertEqual(result['status'], 'PASS')
        self.assertEqual(result['progress_history'][0]['checks'], {})

    def test_empty_final_fails(self):
        result = self.invoke(['x'], lambda save: dict(checks={}, required_keys=[]), True)
        self.assertEqual(result['missing_keys'], ['x'])

    def test_missing_leaf_fails(self):
        result = self.invoke(['x', 'y'], lambda save: dict(checks={'x': {'passed': True}}, required_keys=['x']), True)
        self.assertEqual(result['missing_keys'], ['y'])

    def test_observed_failure_cannot_disappear_in_callback(self):
        for replacement in ({}, {'x': {'passed': True}}):
            def check(save):
                save(dict(checks={'x': {'passed': False}}, required_keys=['x']))
                save(dict(checks=replacement, required_keys=sorted(replacement)))
                return dict(checks=replacement, required_keys=sorted(replacement))
            row = self.invoke(['x'], check, True)
            self.assertFalse(row['checks']['x']['passed'])
            self.assertIn('lost or changed', row['traceback'])

    def test_observed_failure_cannot_disappear_in_final(self):
        def check(save):
            save(dict(checks={'x': {'passed': False}}, required_keys=['x']))
            return dict(checks={'x': {'passed': True}}, required_keys=['x'])
        row = self.invoke(['x'], check, True)
        self.assertFalse(row['checks']['x']['passed'])

    def test_exception_preserves_evidence_and_traceback(self):
        def check(save):
            save(dict(checks={'x': {'passed': True}}, required_keys=['x'], fixture=[1, 2]))
            raise ValueError('deliberate gate failure')
        row = self.invoke(['x', 'y'], check, True)
        self.assertEqual(row['fixture'], [1, 2])
        self.assertIn('deliberate gate failure', row['traceback'])

    def test_nonfinite_and_nested_negative_fail(self):
        for leaf in ({'passed': True, 'nested': {'passed': False}}, {'passed': False}):
            self.invoke(['x'], lambda save: dict(checks={'x': leaf}, required_keys=['x']), True)


if __name__ == '__main__':
    unittest.main()
