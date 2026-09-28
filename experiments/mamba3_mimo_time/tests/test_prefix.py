import copy
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from experiments.mamba3_mimo_time import config as c, records as r

TORCH = importlib.util.find_spec('torch') is not None
if TORCH:
    import torch

    class Tiny(torch.nn.Module):
        def __init__(self, fault=None):
            super().__init__()
            self.item_embedding = torch.nn.Embedding(7112, 3)
            with torch.no_grad():
                self.item_embedding.weight.copy_(torch.arange(7112*3).reshape(7112,3).remainder(23)/20+.1)
            self.other = torch.nn.Parameter(torch.ones(3))
            self.fault, self.calls = fault, []

        def encode_sequence(self, items, lengths, times, oracle=None):
            self.calls.append(dict(grad=torch.is_grad_enabled(), hooks=len(self.item_embedding._forward_hooks)))
            if self.fault == 'forward':
                raise RuntimeError('synthetic forward failure')
            if self.fault == 'frozen_embedding':
                with torch.no_grad():
                    return self.item_embedding(items)
            x = self.item_embedding(items)
            if self.fault == 'duplicate':
                x = x + self.item_embedding(items)
            if self.fault == 'detached':
                return x.detach()
            if self.fault == 'unused':
                return self.other.expand_as(x)
            if self.fault in ('backward', 'nonfinite') and x.requires_grad:
                def fail(g):
                    if self.fault == 'backward':
                        raise RuntimeError('synthetic backward failure')
                    return torch.full_like(g, float('nan'))
                x.register_hook(fail)
            if not torch.is_grad_enabled() and self.fault in ('intervention0', 'intervention1'):
                if sum(not v['grad'] for v in self.calls) == int(self.fault[-1])+1:
                    raise RuntimeError('synthetic intervention failure')
            if self.fault == 'future':
                x = x + x[:, -1:, :]
            if self.fault == 'future_detached':
                x = x + x[:, -1:, :].detach()
            return x


@unittest.skipUnless(TORCH, 'PyTorch required for real CPU hook regression')
class PrefixTests(unittest.TestCase):
    def setUp(self):
        from experiments.mamba3_mimo_time.prefix_checks import check_prefix
        self.check = check_prefix
        self.items = torch.arange(1,35).reshape(2,17)
        self.data = (self.items, torch.tensor([17,17]), torch.arange(34).reshape(2,17).double())
        items = self.items.clone(); items[0,7:] = (items[0,7:]+37)%7111+1
        times = self.data[2].clone(); times[0,7:] += 90000000
        self.interventions = ((items,self.data[1],self.data[2]),(self.items,self.data[1],times))
        self.saved = []

    def run_check(self, net):
        return self.check(net,self.data,7,1,None,self.interventions,
                          lambda checks,phase:self.saved.append(copy.deepcopy((checks,phase))))

    def test_old_lifetime_reproduces_exact_failure(self):
        net=Tiny(); captured=[]
        def old_hook(_m,_a,x):
            x.retain_grad();captured.append(x)
        hook=net.item_embedding.register_forward_hook(old_hook)
        try:
            net.encode_sequence(*self.data).sum().backward()
            self.assertIsNotNone(captured[0].grad)
            with torch.no_grad(),self.assertRaisesRegex(RuntimeError,"can't retain_grad"):
                net.encode_sequence(*self.interventions[0])
        finally:
            hook.remove()
        self.assertFalse(net.item_embedding._forward_hooks)

    def test_production_capture_interventions_and_repeat(self):
        net=Tiny()
        for _ in range(2):
            checks=self.run_check(net)
            self.assertTrue(all(r.leaf_pass(v) for v in checks.values()))
            phase=self.saved[-1][1]
            self.assertEqual(phase['capture_count'],1)
            self.assertTrue(phase['hook_removed'])
            self.assertEqual(phase['snapshots']['positional_gradient']['shape'],[2,17,3])
            events=phase['stages']
            self.assertIn('positional_gradient_received',[e['stage'] for e in events])
            after=[e for e in events if e['stage']=='intervention_completed']
            self.assertEqual(len(after),2)
            self.assertTrue(all(e['capture_count']==1 and not e['grad_enabled'] and not e['output_requires_grad'] for e in after))
            self.assertFalse(net.item_embedding._forward_hooks)
        self.assertEqual([(e['grad'],e['hooks']) for e in net.calls],[(True,1),(False,0),(False,0)]*2)

    def test_cleanup_for_forward_backward_missing_detached_and_nonfinite(self):
        for fault in ('forward','backward','detached','unused','frozen_embedding','duplicate','nonfinite'):
            with self.subTest(fault=fault):
                net=Tiny(fault)
                with self.assertRaises((RuntimeError,ValueError)):
                    self.run_check(net)
                self.assertFalse(net.item_embedding._forward_hooks)
                self.assertTrue(self.saved[-1][1]['hook_removed'])

    def test_grad_disabled_is_failure_and_cleans_hook(self):
        net=Tiny()
        with torch.no_grad(),self.assertRaisesRegex(ValueError,'grad enabled'):
            self.run_check(net)
        self.assertFalse(net.item_embedding._forward_hooks)

    def production_case(self, net, path):
        from experiments.mamba3_mimo_time import admission
        spec=next(x for x in c.plan()['required_cases'] if x['id']=='prefix_base_L17_P7_x1')
        record={'status':'RUNNING'};r.create(path,record)
        registry=r.Registry(path,record,[spec])
        with patch.object(admission,'fresh',return_value=net),patch.object(admission,'histories',return_value=self.data):
            registry.run(spec,lambda save:admission.dispatch(spec,save))
        return r.read(path)['cases'][0]

    def test_complete_production_prefix_causal_fixture(self):
        with tempfile.TemporaryDirectory() as d:
            net=Tiny();row=self.production_case(net,Path(d)/'case.json')
            self.assertEqual(row['status'],'PASS')
            self.assertEqual(len(row['checks']),14)
            self.assertFalse(row['missing_keys'])
            self.assertFalse(net.item_embedding._forward_hooks)
            self.assertEqual(len(net.calls),6)

    def test_suffix_dependency_detected_even_when_detached(self):
        for fault in ('future','future_detached'):
            with self.subTest(fault=fault),tempfile.TemporaryDirectory() as d:
                p=Path(d)/'case.json';net=Tiny(fault)
                with self.assertRaisesRegex(RuntimeError,'Required admission case failed'):
                    self.production_case(net,p)
                row=r.read(p)['cases'][0]
                self.assertEqual(row['status'],'FAIL')
                self.assertFalse(row['checks']['local:prefix_intervention0']['passed'])
                self.assertEqual(row['checks']['local:residual']['passed'],fault=='future_detached')
                self.assertFalse(net.item_embedding._forward_hooks)

    def test_intervention_failure_keeps_partial_evidence_not_pass(self):
        for fault in ('intervention0','intervention1'):
            with self.subTest(fault=fault),tempfile.TemporaryDirectory() as d:
                p=Path(d)/'case.json';net=Tiny(fault)
                with self.assertRaisesRegex(RuntimeError,'Required admission case failed'):
                    self.production_case(net,p)
                row=r.read(p)['cases'][0]
                self.assertEqual(row['status'],'FAIL')
                self.assertIn('synthetic intervention failure',row['traceback'])
                self.assertIn('local:residual',row['checks'])
                self.assertIn('local:cross_user_gradient',row['checks'])
                self.assertEqual(len(row['checks']),3 if fault=='intervention0' else 5)
                self.assertEqual(len(row['missing_keys']),11 if fault=='intervention0' else 9)
                self.assertTrue(row['prefix_phases']['local']['hook_removed'])
                self.assertFalse(net.item_embedding._forward_hooks)
                self.assertFalse(r.accepted_cases([row],c.plan()['required_cases']))
