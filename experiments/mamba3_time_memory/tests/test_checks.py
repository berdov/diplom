"""Production progress/Registry persistence, including real prefix callbacks."""
import json
import tempfile
import unittest
from pathlib import Path
import torch
from experiments.mamba3_mimo_time.records import Registry, accepted_cases, create, read
from experiments.mamba3_mimo_time.prefix_checks import check_prefix
from experiments.mamba3_head_timescales.progress import progress_callback
from experiments.mamba3_time_memory.checks import (
    LeakingFixture, NEGATIVE_KEYS, negative_controls, selector_reader, SELECTOR_KEYS,
)


class RegistryTests(unittest.TestCase):
    def invoke(self, specs, function, raises=False):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'case.json'; result = dict(status='RUNNING')
            create(path,result); registry = Registry(path,result,specs)
            if raises:
                with self.assertRaises(RuntimeError): registry.run(specs[0],function)
            else:
                registry.run(specs[0],function)
            saved = read(path)
            self.assertEqual(saved,json.loads(json.dumps(result,allow_nan=False)))
            return saved

    def test_real_production_prefix_progress_and_negative_detectors(self):
        specs = [dict(id='negative_controls',required_keys=sorted(NEGATIVE_KEYS))]
        saved = self.invoke(specs,lambda save: negative_controls(save,device='cpu'))
        self.assertTrue(accepted_cases(saved['cases'],specs))
        for key in ('leaking_detected','detached_leaking_detected'):
            self.assertTrue(saved['cases'][0]['checks'][key]['first_progress_empty'])
        # Detached leakage evades the positional derivative, but both direct
        # interventions still fail. This is why metadata gradients alone fail.
        self.assertFalse(saved['cases'][0]['checks']['detached_leaking_detected']['positional_residual_detected'])

    def test_selector_reader_production_case_cpu_json(self):
        specs = [dict(id='selector_reader',required_keys=sorted(SELECTOR_KEYS))]
        saved = self.invoke(specs,lambda save: selector_reader(save,device='cpu'))
        self.assertTrue(accepted_cases(saved['cases'],specs))

    def test_real_prefix_callback_registry_json_acceptance(self):
        class Causal(LeakingFixture):
            def encode_sequence(self,items,lengths,timestamps,*,oracle=None):
                return self.item_embedding(items)
        net = Causal(); items = torch.arange(18).reshape(2,9) + 1
        lengths = torch.full((2,),9,dtype=torch.long)
        timestamps = torch.arange(18,dtype=torch.float64).reshape(2,9)
        altered = items.clone(); altered[0,4:] += 23
        changed_times = timestamps.clone(); changed_times[0,4:] += 1000
        keys = ['residual','cross_user_gradient','finite_output',
                'prefix_intervention0','cross_user_intervention0',
                'prefix_intervention1','cross_user_intervention1']
        specs = [dict(id='real_prefix',required_keys=sorted(keys))]
        def function(save):
            latest = {}
            checks = check_prefix(net,(items,lengths,timestamps),4,1.,None,
                                  ((altered,lengths,timestamps),(items,lengths,changed_times)),
                                  progress_callback(save,latest))
            return dict(checks=checks,required_keys=sorted(checks),hook_phase=latest['phase'])
        saved = self.invoke(specs,function)
        self.assertTrue(accepted_cases(saved['cases'],specs))
        history = saved['cases'][0]['progress_history']
        self.assertFalse(history[0]['checks'])
        self.assertEqual(history[0]['phase']['stages'][0]['stage'],'gradient_forward_started')
        self.assertTrue(saved['cases'][0]['hook_phase']['hook_removed'])

    def test_empty_final_fails(self):
        specs = [dict(id='empty',required_keys=['x'])]
        def function(save):
            callback = progress_callback(save,{})
            callback({},dict(stage='started'))
            return dict(checks={},required_keys=[])
        saved = self.invoke(specs,function,raises=True)
        self.assertFalse(accepted_cases(saved['cases'],specs))
        self.assertEqual(saved['cases'][0]['missing_keys'],['x'])

    def test_missing_final_leaf_fails(self):
        specs = [dict(id='missing',required_keys=['x','y'])]
        def function(save):
            callback = progress_callback(save,{})
            callback({},dict(stage='started'))
            callback({'x':dict(passed=True)},dict(stage='partial'))
            return dict(checks={'x':dict(passed=True)},required_keys=['x'])
        saved = self.invoke(specs,function,raises=True)
        self.assertEqual(saved['cases'][0]['missing_keys'],['y'])

    def test_failed_leaf_cannot_disappear_or_turn_positive(self):
        specs = [dict(id='sticky',required_keys=['x'])]
        for next_checks in ({},{'x':dict(passed=True)}):
            def function(save):
                callback = progress_callback(save,{})
                callback({'x':dict(passed=False)},dict(stage='measured'))
                callback(next_checks,dict(stage='later'))
                return dict(checks=next_checks,required_keys=sorted(next_checks))
            saved = self.invoke(specs,function,raises=True)
            self.assertFalse(saved['cases'][0]['checks']['x']['passed'])
            self.assertIn('lost or changed',saved['cases'][0]['traceback'])

    def test_real_hook_finally_on_failure(self):
        class Broken(LeakingFixture):
            def encode_sequence(self,*args,**kwargs):
                super().encode_sequence(*args,**kwargs)
                raise RuntimeError('deliberate failure after capture')
        net = Broken(); items = torch.ones((2,4),dtype=torch.long)
        data = (items,torch.full((2,),4,dtype=torch.long),torch.ones((2,4),dtype=torch.float64))
        seen = {}; records = []
        with self.assertRaisesRegex(RuntimeError,'deliberate failure'):
            check_prefix(net,data,2,1.,None,(),progress_callback(records.append,seen))
        self.assertFalse(net.item_embedding._forward_hooks)
        self.assertTrue(seen['phase']['hook_removed'])
        self.assertEqual(seen['phase']['capture_count'],1)
        self.assertFalse(records[0]['checks'])


if __name__ == '__main__': unittest.main()
