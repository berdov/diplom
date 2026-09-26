"""Targeted CPU regression; unittest needs no pytest installation."""

import copy
import json
import tempfile
import threading
import unittest
import types
from pathlib import Path
from unittest.mock import patch
import torch
from experiments.mamba3_three_time import drift_capture as capture
from experiments.mamba3_three_time import attempt_004 as attempt
from experiments.mamba3_three_time.records_003 import Registry, required_pass
from experiments.mamba3_three_time.evidence import create_record
from experiments.mamba3_three_time.numerics_004 import loss_report, d_oracles, cotangent
from experiments.mamba3_three_time.fixtures import kernel_inputs
from experiments.mamba3_three_time.reference import recurrence
from experiments.mamba3_three_time.gpu_diagnostics_004 import specs, verdicts
from experiments.mamba3_three_time import kernels


def fake_dqkv(x):
    return x*2,x.clone(),x.clone(),x.clone(),x.clone(),x.sum(),None


def fake_invoke(fn,inputs,**kwargs):
    return fn(**inputs),[dict(config=dict(num_warps=4,num_stages=2,maxnreg=128))]


class Probe(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x):
        ctx.handle=capture.forward_handle()
        return x*2

    @staticmethod
    def backward(ctx,g):
        values=capture.dqkv_call(fake_dqkv,'upstream',invocation=ctx.handle,x=g)
        capture.capture_stages(invocation=ctx.handle,after=values[0])
        return values[0]


class Attempt004CPU(unittest.TestCase):
    def test_real_siso_ctx_uses_explicit_handle(self):
        self.assertIn('forward_handle',kernels.SISO.forward.__code__.co_names)
        self.assertIn('diagnostic_invocation',kernels.SISO.forward.__code__.co_names)
        self.assertIn('diagnostic_invocation',kernels.SISO.backward.__code__.co_names)

    def test_autotuner_recorder_isolates_cache(self):
        class Jit:
            src='frozen'
            cache_key='frozen'
            def __getitem__(self,grid):
                return lambda *args,**kw:types.SimpleNamespace(name='kernel',hash='hash',metadata=None)
        class Tuner:
            fn=Jit()
            configs=[types.SimpleNamespace(**capture.FIXED_LAUNCH,kwargs={})]
            cache={}
            best_config=configs[0]
            def __getitem__(self,grid):
                def launch(*args,**kw):
                    self.cache['new']='local'
                    return self.fn[grid](*args,**kw)
                return launch
        tuner=Tuner()
        fake=types.SimpleNamespace(runtime=types.SimpleNamespace(driver=types.SimpleNamespace(
            active=types.SimpleNamespace(get_current_target=lambda:'mock'))))
        with patch.dict('sys.modules',triton=fake):
            for matched in (False,True):
                recorder=capture.LaunchRecorder(tuner,matched=matched)
                recorder[(1,)](torch.ones(1),CHUNK_SIZE=64)
                self.assertEqual(recorder.records[0]['config']['num_warps'],4)
                self.assertEqual(tuner.cache,{})

    def test_cross_thread_forward_backward(self):
        rows,events,errors=[],[],[]
        x=torch.ones(3,requires_grad=True)
        with patch.object(capture,'invoke',fake_invoke),capture.capturing(rows,events):
            y=Probe.apply(Probe.apply(x))
        def backward():
            try:
                with patch.object(capture,'invoke',fake_invoke):y.sum().backward()
            except BaseException as exc:errors.append(exc)
        thread=threading.Thread(target=backward)
        thread.start();thread.join(timeout=15)
        self.assertFalse(thread.is_alive())
        self.assertFalse(errors)
        self.assertEqual(len(rows),2)
        self.assertEqual(len({r['invocation_id'] for r in rows}),2)
        for r in rows:
            self.assertNotEqual(r['forward_thread'],r['backward_thread'])
            self.assertEqual((r['forward_calls'],r['backward_calls']),(1,1))
            self.assertTrue(r['stages'])
        self.assertTrue(torch.equal(x.grad,torch.full_like(x,4)))

    def test_trace_on_off_parity_and_rng(self):
        def run(enabled):
            torch.manual_seed(2026)
            x=torch.randn(9,requires_grad=True)
            rows=[]
            if enabled:
                with capture.capturing(rows):y=Probe.apply(x)
            else:y=Probe.apply(x)
            y.square().sum().backward()
            return y.detach(),x.grad,torch.get_rng_state()
        with patch.object(capture,'invoke',fake_invoke):
            a,b=run(False),run(True)
        self.assertTrue(all(torch.equal(x,y) for x,y in zip(a,b)))

    def test_invocation_ownership_not_last_record(self):
        rows=[]
        with capture.capturing(rows):a,b=capture.forward_handle(),capture.forward_handle()
        with patch.object(capture,'invoke',fake_invoke):
            capture.dqkv_call(fake_dqkv,'upstream',invocation=a,x=torch.ones(2))
            capture.dqkv_call(fake_dqkv,'upstream',invocation=b,x=torch.ones(2)*2)
        capture.capture_stages(invocation=a,mark=torch.tensor(11))
        capture.capture_stages(invocation=b,mark=torch.tensor(22))
        self.assertEqual([r['stages']['mark'].item() for r in rows],[11,22])

    def test_snapshot_storage_and_layout(self):
        x=torch.arange(12.).reshape(3,4).t()
        copies,meta=capture.snapshot_inputs(dict(x=x))
        before=x.clone()
        x.add_(100)
        self.assertTrue(torch.equal(copies['x'],before))
        self.assertTrue(meta['x']['storage_independent'])
        self.assertTrue(meta['x']['layout_preserved'])
        expanded=torch.ones(1).expand(3)
        _,info=capture.snapshot_inputs(dict(x=expanded))
        self.assertFalse(info['x']['layout_preserved'])

    def test_nested_failure_registry_and_unique_ids(self):
        with self.assertRaises(ValueError):Registry({},lambda:None,['a','a'])
        result={};r=Registry(result,lambda:None,['a'])
        r.start('a');row=r.finish('a',dict(passed=True,nested=dict(passed=False)))
        self.assertFalse(row['passed']);self.assertFalse(r.close())
        self.assertTrue(required_pass(dict(required=False,passed=False)))

    def test_overwrite_protection(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp)
            with patch.multiple(attempt,LOGS=base,RUNS=base,LOCK=base/'lock',PIPELINE=base/'pipe',SUMMARY=base/'summary'):
                attempt.require_unused()
                create_record(attempt.LOCK,dict(owned=True))
                with self.assertRaises(FileExistsError):attempt.require_unused()
                with self.assertRaises(FileExistsError):create_record(attempt.LOCK,{})
                self.assertEqual(json.loads(attempt.LOCK.read_text()),dict(owned=True))

    def test_near_zero_loss_bound_does_not_reclassify(self):
        profile=dict(atol=.003,rtol=.08,relative_norm_limit=.08)
        y=torch.zeros(1,dtype=torch.float32)
        a=torch.tensor([1e-5])
        row=loss_report(a,y,profile)
        self.assertTrue(row['mixed_only']['passed'])
        self.assertFalse(row['legacy_with_norm_cap']['passed'])
        self.assertIsNone(row['pure_relative_error_fp64'])
        self.assertTrue(row['bound_explains_fp64_delta'])
        self.assertFalse(row['acceptance_changed'])
        y=torch.tensor([-1.,.2,1.,-.5])
        row=loss_report(y+torch.tensor([.001,-.001,.002,.001]),y,profile)
        self.assertTrue(row['bound_explains_fp64_delta'])

    def test_cotangents_fixed_and_quantized(self):
        for kind in ('signed','nonnegative'):
            a,m=cotangent((1,7,2,4),kind,'cpu')
            b,_=cotangent((1,7,2,4),kind,'cpu')
            self.assertTrue(torch.equal(a,b))
            self.assertAlmostEqual(m['pre_cast_l2'],1.)
            self.assertEqual(a.dtype,torch.bfloat16)
            self.assertTrue(bool((a>=0).all()) if kind=='nonnegative' else bool((a<0).any()))

    def test_D_oracle_independent_reference_and_zero_ssm(self):
        values=kernel_inputs('MIMO',length=3,device='cpu',dtype=torch.float64,tiny=True,tied=False)
        with torch.no_grad():
            for k in ('q','k','qb','kb'):values[k].zero_()
        out=recurrence(**values)
        go,_=cotangent(out.shape,'signed','cpu',dtype=torch.float64)
        expected=torch.autograd.grad(out,values['d'],go)[0]
        oracle=d_oracles(values,go)['ideal_fp64']
        self.assertTrue(torch.allclose(expected,oracle,atol=1e-12,rtol=1e-12))
        vp=values['v'].unsqueeze(-2)*values['mv']
        direct=(vp*values['d'][None,None,:,None,None]*torch.nn.functional.silu(
            values['z'].unsqueeze(-2)*values['mz'])*values['mo']).sum(-2)
        self.assertTrue(torch.allclose(out,direct,atol=1e-12,rtol=1e-12))

    def test_independent_slots_gradcheck(self):
        values=kernel_inputs('SISO',length=3,device='cpu',dtype=torch.float64,tiny=True)
        keys=('adt','dw','dp')
        def fun(*args):return recurrence(**{**values,**dict(zip(keys,args))})
        self.assertTrue(torch.autograd.gradcheck(fun,tuple(values[k] for k in keys),eps=1e-6,atol=1e-5,rtol=.001))

    def test_plan_registry_and_non_authorizing_summary(self):
        plan=json.loads(attempt.PLAN.read_text())
        old=json.loads((attempt.HERE/'test_plan_003.json').read_text())
        self.assertEqual(plan['structural'],old['structural'])
        self.assertEqual(plan['reference'],old['reference'])
        self.assertFalse(plan['training_authorized'])
        for arch in ('SISO','MIMO'):
            ids=[k for k,_ in specs(arch,plan)]
            self.assertEqual(len(ids),len(set(ids)))
        summary=verdicts(dict(cases=[]))
        self.assertFalse(summary['training_authorized'])
        self.assertEqual(summary['implementation_parity'],'UNKNOWN')

    def test_launcher_single_submit_and_no_scientific_calls(self):
        import ast
        source=(attempt.HERE/'submit_004.py').read_text()
        self.assertEqual(source.count('["sbatch",'),1)
        self.assertFalse(any(s in source for s in ('squeue','sacct','sleep','watch')))
        launcher=(attempt.ROOT/'slurm/mamba3_three_time_diagnostics_004.sh').read_text()
        for s in ('--time=01:30:00','--mem=0','--no-requeue','--cpus-per-task=4',
                  '--gres=gpu:a100:1','--partition=rocky','--account=proj_1833','--constraint=type_e'):
            self.assertIn(s,launcher)
        for name in ('diagnostics_004.py','mimo_diagnostics_004.py','gpu_diagnostics_004.py'):
            for node in ast.walk(ast.parse((attempt.HERE/name).read_text())):
                if isinstance(node,ast.Call):
                    function=node.func.attr if isinstance(node.func,ast.Attribute) else getattr(node.func,'id','')
                    self.assertNotIn(function,('fit','evaluate','create_dataset','data_preparation','step'))


if __name__=='__main__':unittest.main()
