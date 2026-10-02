import copy
import inspect
import types
import unittest
from unittest.mock import patch
import torch
from experiments.mamba3_gap_trap.centered import config as c
from experiments.mamba3_gap_trap.centered.modulation import GapTrap,centered_q
from experiments.mamba3_gap_trap.centered.checks import modulation_checks,CHECK_KEYS
from experiments.mamba3_gap_trap.centered import pipeline,report,runner,provenance
from experiments.mamba3_gap_trap.centered.reuse import bind
from experiments.mamba3_gap_trap.centered.state import initial,paired,transfer_common,effective_check
from experiments.mamba3_gap_trap.centered.process_env import child_environment


class CenteredTests(unittest.TestCase):
    def test_all_centered_leaves(self):
        rows={};modulation_checks('cpu',lambda k,v:rows.update({k:v}))
        self.assertEqual(set(rows),set(CHECK_KEYS))
        for key,row in rows.items():
            with self.subTest(key=key):self.assertTrue(row['passed'],row)

    def test_analytic_gradient(self):
        g=torch.tensor([[0.,.1*838393,838393.,4*838393]],dtype=torch.float64,requires_grad=True)
        model=GapTrap().double()
        with torch.no_grad():model.alpha.fill_(.4)
        value=model(g,torch.ones_like(g,dtype=torch.bool))
        dg,da=torch.autograd.grad(value.sum(),(g,model.alpha))
        self.assertTrue(torch.allclose(dg,.4*2*838393/(g+838393)**2,atol=1e-18,rtol=1e-10))
        self.assertAlmostEqual(float(da),float(((g-838393)/(g+838393)).sum()),places=12)

    def test_gradcheck(self):
        g=torch.tensor([[1.,838393.,4e6]],dtype=torch.float64,requires_grad=True)
        a=torch.tensor(.4,dtype=torch.float64,requires_grad=True)
        mask=torch.ones_like(g,dtype=torch.bool)
        self.assertTrue(torch.autograd.gradcheck(lambda x,y:y.clamp(0,1)*centered_q(x,mask,torch.float64),(g,a)))

    def test_boundary_derivative(self):
        m=GapTrap().double();g=torch.tensor([[4*838393.]],dtype=torch.float64);a=torch.ones_like(g,dtype=torch.bool)
        for value,direction in [(0.,1.),(1.,-1.)]:
            with torch.no_grad():m.alpha.fill_(value)
            y=m(g,a);derivative=torch.autograd.grad(y.sum(),m.alpha)[0]
            with torch.no_grad():m.alpha.add_(direction*1e-6)
            self.assertAlmostEqual(float(derivative),float((m(g,a)-y.detach())/(direction*1e-6)),places=8)

    def test_invalid(self):
        for value in (-1.,float('nan'),float('inf')):
            g=torch.tensor([[value]])
            with self.assertRaises(ValueError):GapTrap()(g,torch.ones_like(g,dtype=torch.bool))
        with self.assertRaises(ValueError):GapTrap()(torch.zeros(1,1),torch.zeros(1,1))

    def test_tanh_equivalence(self):
        ratios=torch.tensor([[.01,.1,1.,10.,100.]],dtype=torch.float64)
        self.assertTrue(torch.allclose(centered_q(ratios*838393,torch.ones_like(ratios,dtype=torch.bool),torch.float64),torch.tanh(.5*ratios.log()),atol=1e-15,rtol=1e-15))

    def test_cpu_common_gradient_identity(self):
        weights=torch.randn(3,4,requires_grad=True)
        twin=weights.detach().clone().requires_grad_()
        gaps=torch.tensor([[0.,838393.,4e6,1.]],dtype=torch.float64);active=torch.ones_like(gaps,dtype=torch.bool)
        m=GapTrap()
        old=weights.sigmoid();new=(twin+m(gaps,active)).sigmoid()
        self.assertTrue(torch.equal(old,new))
        old.square().sum().backward();new.square().sum().backward()
        self.assertTrue(torch.equal(weights.grad,twin.grad))
        self.assertTrue(torch.isfinite(m.alpha.grad))

    def test_counts_state_rng_config_first_batch(self):
        from recbole.config import Config
        from recbole.utils import init_seed
        from experiments.mamba3_three_time.config import SyntheticCatalog
        from experiments.mamba3_three_time.model import ThreeTimeMamba3Rec
        from experiments.mamba3_gap_trap.centered.model import GapTrapMamba3Rec
        records=[];models=[];batches=[]
        for mode in c.MODES:
            cfg=Config(model=ThreeTimeMamba3Rec,config_dict=c.settings(mode,'cpu'));effective_check(cfg,mode)
            init_seed(2026,True)
            generator=torch.Generator().manual_seed(2026)
            loader=torch.utils.data.DataLoader(torch.arange(100),batch_size=16,shuffle=True,generator=generator)
            net=GapTrapMamba3Rec(cfg,SyntheticCatalog());models.append(net)
            self.assertEqual(sum(p.numel() for p in net.parameters()),c.COUNTS[mode])
            self.assertEqual(set(dict(net.named_parameters())),set(c.plan()['common_parameter_keys'])|({'gap_trap.alpha'} if mode=='centered_gap_trap' else set()))
            records.append(initial(net,loader));batches.append(next(iter(loader)))
        paired(*records);transfer_common(*models)
        self.assertTrue(torch.equal(*batches));self.assertFalse(torch.cuda.is_initialized())
        self.assertEqual(models[1].gap_trap.alpha.dtype,torch.float32)
        self.assertEqual(float(models[1].gap_trap.alpha),0.)
        self.assertEqual(len([k for k in models[1].state_dict() if k.startswith('gap_trap.')]),1)

    def test_alpha_does_not_consume_rng(self):
        before=torch.get_rng_state().clone();GapTrap()
        self.assertTrue(torch.equal(before,torch.get_rng_state()))

    def test_no_target_timestamp(self):
        from experiments.mamba3_timeaware.model import TimeAwareMamba3Rec
        class StrictHistory(dict):
            def __getitem__(self,key):
                if key not in ('items','lengths','history_times'):raise AssertionError('Target access')
                return super().__getitem__(key)
        args=[]
        dummy=types.SimpleNamespace(ITEM_SEQ='items',ITEM_SEQ_LEN='lengths',time_sequence_field='history_times',forward=lambda *values:args.extend(values))
        TimeAwareMamba3Rec._encode(dummy,StrictHistory(items='i',lengths='l',history_times='t',timestamp='forbidden'))
        self.assertEqual(args,['i','l','t'])
        from experiments.mamba3_gap_trap.centered.model import GapTrapMamba3Rec
        self.assertIs(GapTrapMamba3Rec._encode,TimeAwareMamba3Rec._encode)
        self.assertNotIn('target',inspect.signature(GapTrapMamba3Rec.encode_sequence).parameters)

    def test_private_bindings_and_parent_immutable(self):
        from experiments.mamba3_gap_trap import runner as old
        before=dict(vars(old))
        private=bind(old,{'c':c},'experiments.mamba3_gap_trap.centered')
        self.assertIs(private['main'].__code__,old.main.__code__)
        self.assertIs(private['main'].__globals__['train'],private['train'])
        self.assertEqual(private['main'].__globals__['__package__'],'experiments.mamba3_gap_trap.centered')
        self.assertTrue(all(vars(old)[k] is v for k,v in before.items()))
        self.assertIs(runner.train.__globals__['c'],c)
        self.assertIs(runner.train.__globals__['initial'],initial)
        self.assertNotEqual(old.c.HERE,c.HERE)

    def test_paths_plan_and_environment(self):
        from experiments.mamba3_gap_trap import config as old
        self.assertEqual(c.MODES,('fixed_replay','centered_gap_trap'))
        for mode in c.MODES:
            self.assertTrue(c.paths(mode)['result'].is_relative_to(c.HERE))
            self.assertTrue(c.paths(mode)['checkpoint'].is_relative_to(c.HERE))
        self.assertFalse(c.HERE==old.HERE)
        p=c.plan();self.assertEqual((p['max_jobs'],p['max_scientific_fits'],p['test_evaluations'],p['automatic_retries']),(1,2,0,0))
        for spec in p['required_cases']:self.assertEqual(spec['required_keys'],sorted(spec['required_keys']))
        env={'CUDA_VISIBLE_DEVICES':'fixture'}
        self.assertEqual(child_environment('centered_gap_trap',env)['CUDA_VISIBLE_DEVICES'],'fixture')
        self.assertEqual(child_environment('preflight',env)['CUDA_VISIBLE_DEVICES'],'')
        self.assertEqual(env,{'CUDA_VISIBLE_DEVICES':'fixture'})
        with self.assertRaises(ValueError):child_environment('gap_trap',env)

    def test_child_routes_only_centered(self):
        with patch.object(pipeline,'_child') as run:
            for stage in ('gate','smoke',*c.MODES):
                fit=stage in c.MODES
                pipeline.child(stage,'experiments.mamba3_gap_trap.'+('runner' if fit else stage),['--variant',stage] if fit else [],1.,c.LOGS/stage)
                self.assertTrue(run.call_args.args[1].startswith('experiments.mamba3_gap_trap.centered.'))
            with self.assertRaises(ValueError):pipeline.child('gap_trap','experiments.mamba3_gap_trap.runner',[],1.,c.LOGS)
            with self.assertRaises(ValueError):pipeline.child('gate','other.gate',[],1.,c.LOGS)

    def test_replay_contract_and_negative_controls(self):
        from experiments.mamba3_gap_trap import config as old
        from experiments.mamba3_mimo_time.records import read
        reference=read(old.paths('fixed_replay')['result'])
        self.assertEqual(report.replay_check(reference)['status'],'PASS')
        for key in ('checkpoint_sha256','best_epoch','first_train_batch_sha256','rng_components'):
            altered=copy.deepcopy(reference);altered[key]='wrong'
            with self.assertRaises(ValueError):report.replay_check(altered)
        altered=copy.deepcopy(reference);altered['history'][3]['valid_metrics']['ndcg@10']+=.0001
        with self.assertRaises(ValueError):report.replay_check(altered)
        altered=copy.deepcopy(reference);altered['history'][0]['train_seconds']+=123
        self.assertEqual(report.replay_check(altered)['status'],'PASS')

    def test_replay_failure_blocks_second_fit(self):
        from experiments.mamba3_gap_trap import config as old
        from experiments.mamba3_mimo_time.records import read
        r=read(old.paths('fixed_replay')['result']);r.update(run_id=c.paths('fixed_replay')['run_id'])
        report.validate_record(r,'fixed_replay')
        r['checkpoint_sha256']='incorrect'
        with self.assertRaises(ValueError):report.validate_record(r,'fixed_replay')
        self.assertIs(report.write.__globals__['validate_record'],report.validate_record)

    def test_diagnostics_grid_bf16(self):
        from experiments.mamba3_gap_trap.centered.trainer import diagnostics,trainer_class
        model=types.SimpleNamespace(gap_trap_mode='centered_gap_trap',gap_trap=GapTrap())
        with torch.no_grad():model.gap_trap.alpha.fill_(.0005)
        d=diagnostics(model)
        self.assertEqual(d['q_centered'][0],-1.)
        self.assertEqual(d['q_centered'][5],0.)
        self.assertGreater(d['bf16']['rounded_to_zero_fraction'],0.)
        self.assertEqual(len(d['current_fraction']),3)
        self.assertIs(trainer_class.__globals__['diagnostics'],diagnostics)


if __name__=='__main__':unittest.main()
