"""CPU mathematical admission and real production hook lifecycle."""
import tempfile
import unittest
from pathlib import Path
import torch
from experiments.mamba3_mimo_time.records import Registry, accepted_cases, create, read
from experiments.mamba3_mimo_time.prefix_checks import check_prefix
from experiments.mamba3_head_timescales.progress import progress_callback
from experiments.mamba3_absolute_phase.gate import run_required_case
from experiments.mamba3_absolute_phase.checks import (
    OPERATION_KEYS, NEGATIVE_KEYS, phase_operations, negative_controls,
    required_cases, informative_history,
)
from experiments.mamba3_time_memory.checks import LeakingFixture


class SyntheticChecksTests(unittest.TestCase):
    def invoke(self, name, keys, function):
        specs = [dict(id=name, required_keys=sorted(keys))]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'gate.json'
            value = dict(status='RUNNING')
            create(path, value)
            run_required_case(Registry(path, value, specs), specs[0], function)
            value = read(path)
            self.assertTrue(accepted_cases(value['cases'], specs))
            return value['cases'][0]

    def test_phase_operation_cpu_through_real_registry(self):
        self.invoke('phase_operation', OPERATION_KEYS,
                    lambda save: phase_operations(save, device='cpu'))

    def test_actual_negative_detectors_cpu_through_registry(self):
        row = self.invoke('negative_controls', NEGATIVE_KEYS,
                          lambda save: negative_controls(save, device='cpu'))
        self.assertFalse(row['checks']['detached_leaking_detected']['positional_residual_detected'])
        self.assertTrue(row['checks']['detached_leaking_detected']['first_progress_empty'])

    def test_prefix_callback_empty_progress_and_no_grad_interventions(self):
        class Causal(LeakingFixture):
            def encode_sequence(self, items, lengths, timestamps, *, oracle=None):
                return self.item_embedding(items)
        net = Causal()
        items = torch.arange(18).reshape(2, 9)+1
        lengths = torch.full((2,), 9, dtype=torch.long)
        timestamps = torch.arange(18, dtype=torch.float64).reshape(2, 9)
        changed_items = items.clone()
        changed_items[0, 4:] += 23
        changed_times = timestamps.clone()
        changed_times[0, 4:] += 1000
        keys = ['residual', 'cross_user_gradient', 'finite_output',
                'prefix_intervention0', 'cross_user_intervention0',
                'prefix_intervention1', 'cross_user_intervention1']
        def check(save):
            latest = {}
            checks = check_prefix(net, (items, lengths, timestamps), 4, 1., None,
                ((changed_items, lengths, timestamps), (items, lengths, changed_times)),
                progress_callback(save, latest))
            self.assertFalse(net.item_embedding._forward_hooks)
            self.assertTrue(latest['phase']['hook_removed'])
            self.assertTrue(all(not s['grad_enabled'] for s in latest['phase']['stages']
                                if s['stage'].startswith('intervention')))
            return dict(checks=checks, required_keys=sorted(checks))
        self.invoke('real_prefix', keys, check)

    def test_hook_finally_after_actual_forward_exception(self):
        class Broken(LeakingFixture):
            def encode_sequence(self, *args, **kwargs):
                super().encode_sequence(*args, **kwargs)
                raise RuntimeError('deliberate exception after hook capture')
        net = Broken()
        items = torch.ones((2, 4), dtype=torch.long)
        data = (items, torch.full((2,), 4, dtype=torch.long), torch.ones((2, 4), dtype=torch.float64))
        latest = {}
        saved = []
        with self.assertRaisesRegex(RuntimeError, 'deliberate exception'):
            check_prefix(net, data, 2, 1., None, (), progress_callback(saved.append, latest))
        self.assertFalse(net.item_embedding._forward_hooks)
        self.assertTrue(latest['phase']['hook_removed'])
        self.assertEqual(latest['phase']['capture_count'], 1)
        self.assertFalse(saved[0]['checks'])

    def test_case_ids_and_leaves_unique(self):
        specs = required_cases(['x', 'y'])
        self.assertEqual(len(specs), 17)
        self.assertEqual(len({s['id'] for s in specs}), len(specs))
        for spec in specs:
            self.assertEqual(spec['required_keys'], sorted(set(spec['required_keys'])))
            self.assertTrue(spec['required_keys'])

    def test_informative_fixture_is_fixed_and_multievent(self):
        items, lengths, timestamps = informative_history(device='cpu')
        self.assertEqual(items.tolist(), [[3, 71, 109, 23, 401, 82, 9, 55]])
        self.assertEqual(lengths.tolist(), [8])
        self.assertEqual(timestamps[0, 0], timestamps[0, 1])
        self.assertGreater(timestamps[0, -1], timestamps[0, 0])


if __name__ == '__main__':
    unittest.main()
