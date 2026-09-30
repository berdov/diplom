"""Real gate callback, prefix helper, Registry and disk JSON; CPU fixtures only."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
try:
    import torch
except ModuleNotFoundError as exc:
    if exc.name!='torch':raise
    raise unittest.SkipTest('PyTorch unavailable; cluster integration tests mandatory') from exc

from experiments.mamba3_head_timescales import gate,config as c
from experiments.mamba3_head_timescales.progress import progress_callback
from experiments.mamba3_timeaware.time_inputs import history_gaps
from experiments.mamba3_mimo_time.records import Registry,accepted_cases,case,create,read


class Tiny(torch.nn.Module):
    def __init__(self,leak=False,detached=True,fail=None):
        super().__init__()
        self.item_embedding=torch.nn.Embedding(128,3).double()
        self.times=torch.nn.Module();self.times.calibrators=torch.nn.ModuleDict()
        self.leak,self.detached,self.fail=leak,detached,fail
        self.forwards=[]
        with torch.no_grad():self.item_embedding.weight.copy_(torch.arange(384).reshape(128,3)/1000)

    def encode_sequence(self,items,lengths,timestamps,oracle=None):
        self.forwards.append(dict(grad_enabled=torch.is_grad_enabled(),hooks=len(self.item_embedding._forward_hooks)))
        if self.fail=='before':raise RuntimeError('fixture before numerical checks')
        if self.fail=='partial' and not torch.is_grad_enabled():raise RuntimeError('fixture after partial checks')
        e=self.item_embedding(items)
        gaps,_=history_gaps(timestamps,items!=0)
        result=e+.1*torch.log1p(gaps/1000)[...,None]
        if self.leak:
            suffix=.2*e[:,7:].mean(1,keepdim=True)+(.01*timestamps[:,7:].mean(1,keepdim=True)/1000)[...,None]
            result=result+(suffix.detach() if self.detached else suffix)
        return result


def histories(_length):
    items=torch.arange(1,21).reshape(2,10)
    times=torch.tensor([[100,250,500,800,1200,1700,2300,3000,3800,4700],
                        [200,400,700,1100,1600,2200,2900,3700,4600,5600]],dtype=torch.float64)
    return items,torch.tensor([10,10]),times


def old_callback(save,latest):
    def persist(checks,phase):
        latest.update(checks=checks,phase=phase)
        save(case(checks,hook_phase=phase))
    return persist


class ProgressIntegrationTests(unittest.TestCase):
    def execute(self,model,old=False):
        spec=next(x for x in c.plan()['required_cases'] if x['id']=='causality_fixed')
        snapshots=[];payloads=[]
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'gate.json';result={};create(path,result)
            registry=Registry(path,result,[spec])
            def function(save):
                def observed(payload):
                    save(payload)  # Real Registry.save, real atomic JSON write.
                    snapshots.append(read(path))
                    payloads.append((payload,copy.deepcopy(payload)))
                return gate.causal('fixed',observed)
            with patch.object(gate,'fresh',return_value=model),patch.object(gate,'histories',side_effect=histories), \
                 patch.object(gate,'progress_callback',old_callback if old else progress_callback):
                try:registry.run(spec,function)
                except RuntimeError:pass  # Outcome is asserted from the actual saved failure below.
            saved=read(path)
        return saved,[spec],snapshots,payloads

    def test_old_callback_reproduces_exact_failure_before_forward(self):
        net=Tiny();saved,specs,_,_=self.execute(net,old=True);row=saved['cases'][0]
        self.assertEqual(row['status'],'FAIL');self.assertEqual(row['checks'],{})
        self.assertIn('ValueError: Empty required checks',row['traceback'])
        self.assertIn('gradient_forward_started',row['traceback'])
        self.assertEqual(row['missing_keys'],specs[0]['required_keys'])
        self.assertEqual(net.forwards,[]);self.assertFalse(net.item_embedding._forward_hooks)
        self.assertFalse(accepted_cases(saved['cases'],specs))

    def test_first_empty_progress_is_running_on_disk_and_not_accepted(self):
        _,specs,snapshots,_=self.execute(Tiny())
        first=snapshots[0]['cases'][0]
        self.assertEqual(first['status'],'RUNNING');self.assertEqual(first['checks'],{})
        self.assertEqual(first['required_keys'],[])
        self.assertEqual(first['hook_phase']['stages'][0]['stage'],'gradient_forward_started')
        self.assertFalse(accepted_cases(snapshots[0]['cases'],specs))
        self.assertNotIn('passed',first)

    def test_full_production_cpu_chain_passes_all_ten_checks(self):
        net=Tiny();saved,specs,snapshots,_=self.execute(net);row=saved['cases'][0]
        self.assertTrue(accepted_cases(saved['cases'],specs),row)
        self.assertEqual(len(row['checks']),10)
        self.assertTrue(row['checks']['timestamp_residual']['passed'])
        self.assertGreater(row['checks']['timestamp_residual']['prefix_l2'],0)
        self.assertFalse(net.item_embedding._forward_hooks)
        self.assertEqual(net.forwards,[dict(grad_enabled=True,hooks=1),dict(grad_enabled=False,hooks=0),dict(grad_enabled=False,hooks=0)])
        phase=row['hook_phase']
        self.assertTrue(phase['hook_removed']);self.assertEqual(phase['capture_count'],1)
        for name in ('gradient_forward_completed','positional_gradient_received','hook_removed','intervention_started','intervention_completed'):
            self.assertIn(name,[x['stage'] for x in phase['stages']])
        # Only the final Registry decision is PASS, never an intermediate partial.
        self.assertTrue(all(x['cases'][0]['status']=='RUNNING' for x in snapshots))

    def test_exception_before_checks_retains_phase_traceback_and_cleanup(self):
        net=Tiny(fail='before');saved,specs,_,_=self.execute(net);row=saved['cases'][0]
        self.assertEqual(row['status'],'FAIL');self.assertEqual(row['checks'],{})
        self.assertIn('fixture before numerical checks',row['traceback'])
        self.assertTrue(row['hook_phase']['hook_removed'])
        self.assertEqual(row['missing_keys'],specs[0]['required_keys'])
        self.assertFalse(net.item_embedding._forward_hooks)
        self.assertFalse(accepted_cases(saved['cases'],specs))

    def test_exception_after_partial_keeps_measured_values_only(self):
        net=Tiny(fail='partial');saved,specs,snapshots,_=self.execute(net);row=saved['cases'][0]
        self.assertEqual(row['status'],'FAIL')
        self.assertEqual(set(row['checks']),{'residual','cross_user_gradient','finite_output'})
        self.assertEqual(len(row['missing_keys']),7)
        self.assertIn('fixture after partial checks',row['traceback'])
        self.assertTrue(row['hook_phase']['hook_removed']);self.assertFalse(net.item_embedding._forward_hooks)
        for prior in snapshots:
            for key,value in prior['cases'][0]['checks'].items():self.assertEqual(row['checks'][key],value)
        self.assertFalse(accepted_cases(saved['cases'],specs))

    def test_detached_suffix_leak_fails_forward_interventions(self):
        net=Tiny(leak=True);saved,specs,snapshots,_=self.execute(net);row=saved['cases'][0]
        self.assertEqual(row['status'],'FAIL');self.assertEqual(row['missing_keys'],[])
        self.assertTrue(row['checks']['residual']['passed'])
        self.assertTrue(row['checks']['timestamp_residual']['passed'])
        for key in ('prefix_intervention0','prefix_intervention1'):
            self.assertFalse(row['checks'][key]['passed'])
            seen=False
            for s in snapshots:
                check=s['cases'][0]['checks'].get(key)
                if check is not None:
                    seen=True;self.assertFalse(check['passed'])
            self.assertTrue(seen)
        self.assertFalse(accepted_cases(saved['cases'],specs))
        self.assertFalse(net.item_embedding._forward_hooks)

    def test_gradient_visible_suffix_leak_also_fails(self):
        saved,specs,_,_=self.execute(Tiny(leak=True,detached=False))
        row=saved['cases'][0]
        self.assertFalse(row['checks']['residual']['passed'])
        self.assertFalse(row['checks']['timestamp_residual']['passed'])
        self.assertFalse(accepted_cases(saved['cases'],specs))

    def test_final_empty_missing_duplicate_and_unexpected_rejected(self):
        good,specs,_,_=self.execute(Tiny());original=good['cases'][0]
        for damage in ('empty','missing','unexpected','duplicate_key'):
            row=copy.deepcopy(original);spec=copy.deepcopy(specs[0])
            if damage=='empty':row['checks']={};row['required_keys']=[]
            elif damage=='missing':row['checks'].pop('finite_output');row['required_keys']=sorted(row['checks'])
            elif damage=='unexpected':row['checks']['unexpected']={'passed':True};row['required_keys']=sorted(row['checks'])
            else:spec['required_keys'].append(spec['required_keys'][0])
            with self.subTest(damage=damage),tempfile.TemporaryDirectory() as directory:
                p=Path(directory)/'gate.json';r={};create(p,r);registry=Registry(p,r,[spec])
                with self.assertRaises(RuntimeError):registry.run(spec,lambda save:row)
                self.assertEqual(read(p)['cases'][0]['status'],'FAIL')
                self.assertFalse(accepted_cases(read(p)['cases'],[spec]))
        with self.assertRaises(ValueError):case({})
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'gate.json';r={};create(p,r)
            with self.assertRaises(ValueError):Registry(p,r,[specs[0],specs[0]])
        self.assertFalse(accepted_cases(good['cases']+good['cases'],specs))

    def test_repeated_execution_does_not_leak_hooks_or_mutate_snapshots(self):
        net=Tiny();first,specs,snaps,payloads=self.execute(net);frozen=copy.deepcopy(first)
        second,specs2,_,_=self.execute(net)
        self.assertTrue(accepted_cases(first['cases'],specs));self.assertTrue(accepted_cases(second['cases'],specs2))
        self.assertEqual(first,frozen);self.assertFalse(net.item_embedding._forward_hooks)
        for live,old in payloads:self.assertEqual(live,old)
        self.assertEqual(snaps[0]['cases'][0]['checks'],{})
        self.assertEqual(len(snaps[0]['cases'][0]['hook_phase']['stages']),1)

    def test_progress_cannot_erase_negative_or_capture_live_tensor(self):
        # Real Registry save; errors retain the previously observed failure.
        spec=dict(id='negative',required_keys=['a'])
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'gate.json';r={};create(p,r);registry=Registry(p,r,[spec])
            def run(save):
                callback=progress_callback(save,{})
                callback({'a':{'passed':False}},{'stages':[]})
                callback({},{'stages':[]})
            with self.assertRaises(RuntimeError):registry.run(spec,run)
            self.assertFalse(read(p)['cases'][0]['checks']['a']['passed'])
            self.assertFalse(accepted_cases(read(p)['cases'],[spec]))
        with self.assertRaises(TypeError):progress_callback(lambda _:None,{})({},dict(tensor=torch.ones(1,requires_grad=True)))


if __name__=='__main__':unittest.main()
