import copy
import json
import unittest
try:
    import torch
except ModuleNotFoundError as exc:
    if exc.name != 'torch':raise
    raise unittest.SkipTest('PyTorch unavailable; cluster CPU tests remain mandatory') from exc
from experiments.mamba3_head_timescales import config as c
from experiments.mamba3_head_timescales.calibrator import LearnedReference
from experiments.mamba3_head_timescales.checks import fresh_cal,nonzero,calibrator_checks,tied_checks
from experiments.mamba3_timeaware.time_inputs import TimeCalibrator,history_gaps


class CalibratorTests(unittest.TestCase):
    def test_reference_and_masks_both_variants(self):
        for variant in ('shared_tau','head_tau'):
            results={};calibrator_checks(variant,'cpu',lambda k,v:results.update({k:v}))
            expected=next(s['required_keys'] for s in c.plan()['required_cases'] if s['id']=='calibrator_'+variant)
            self.assertEqual(sorted(results),expected)
            self.assertTrue(all(r['passed'] for r in results.values()),results)

    def test_alpha0_with_nonzero_and_zero_last(self):
        gaps=torch.linspace(0,9,15,dtype=torch.float64).reshape(3,5)*838393
        active=torch.ones_like(gaps,dtype=torch.bool);active[1,3:]=False
        for nonzero_last in (False,True):
            old=TimeCalibrator(2,838393).double()
            if nonzero_last:nonzero(old)
            for variant in ('shared_tau','head_tau'):
                new=LearnedReference(copy.deepcopy(old),variant)
                torch.testing.assert_close(new(gaps,active),old(gaps,active),atol=1e-6,rtol=1e-5)
                if nonzero_last:
                    names=['first.weight','first.bias','last.weight','last.bias']
                    ga=torch.autograd.grad(new(gaps,active).sum(),[dict(new.named_parameters())[n] for n in names])
                    gb=torch.autograd.grad(old(gaps,active).sum(),[dict(old.named_parameters())[n] for n in names])
                    for a,b in zip(ga,gb):torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5)

    def test_tied_nonzero_derivatives(self):
        results={};tied_checks('cpu',lambda k,v:results.update({k:v}))
        self.assertTrue(all(r['passed'] for r in results.values()),results)

    def test_gradcheck_all_leaves(self):
        g=torch.tensor([[.03,.4,1.7],[3.,7.,19.]],dtype=torch.float64)*838393
        active=torch.ones_like(g,dtype=torch.bool)
        for variant in ('shared_tau','head_tau'):
            cal=fresh_cal(variant)
            names=['alpha','first.weight','first.bias','last.weight','last.bias']
            params=dict(cal.named_parameters())
            leaves=tuple(params[n].detach().clone().contiguous().requires_grad_() for n in names)
            def function(*args):return torch.func.functional_call(cal,dict(zip(names,args)),(g,active))
            self.assertTrue(torch.autograd.gradcheck(function,leaves,eps=1e-6,atol=1e-6,rtol=1e-5))

    def test_invalid_inputs(self):
        cal=fresh_cal('head_tau')
        for value in (-1.,float('nan'),float('inf')):
            with self.assertRaises(ValueError):cal(torch.tensor([[value]]),torch.ones((1,1),dtype=torch.bool))
        with self.assertRaises(ValueError):cal(torch.ones(2,3),torch.ones(2,3))
        with self.assertRaises(ValueError):cal(torch.ones(2,3,1),torch.ones(2,3,1,dtype=torch.bool))

    def test_history_zero_first_padding(self):
        valid=torch.tensor([[True]*5,[True,True,True,False,False],[True,False,False,False,False]])
        ts=torch.tensor([[100,100,200,400,800],[100,200,200,0,0],[100,0,0,0,0]],dtype=torch.float64)
        gaps,active=history_gaps(ts,valid)
        self.assertTrue(active[0,1] and active[1,2])
        self.assertFalse(active[:,0].any())
        self.assertFalse(active[2].any())
        cal=fresh_cal('head_tau');out=cal(gaps,active)
        self.assertTrue(torch.equal(out[~active],torch.ones_like(out[~active])))
        self.assertTrue((out[0,1]!=1).any())

    def test_rng_wrapper_and_bound_saturation(self):
        old=TimeCalibrator(2,838393)
        rng=torch.get_rng_state().clone()
        for mode in ('shared_tau','head_tau'):
            cal=LearnedReference(copy.deepcopy(old),mode)
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            for sign in (-1,1):
                with torch.no_grad():cal.alpha.fill_(sign*1e6)
                ratio=cal.log_reference_ratio().exp()
                self.assertTrue(((ratio>=.25)&(ratio<=4)).all())

    def test_alpha_learns_after_zero_last(self):
        g=torch.tensor([[1.,.1,4.,10.]],dtype=torch.float64)*838393
        active=torch.ones_like(g,dtype=torch.bool)
        for variant in ('shared_tau','head_tau'):
            cal=LearnedReference(TimeCalibrator(2,838393).double(),variant)
            opt=torch.optim.Adam(cal.parameters(),lr=.001)
            gradients=[]
            for _ in range(4):
                opt.zero_grad();loss=(cal(g,active)-1.3).square().mean();loss.backward()
                gradients.append(cal.alpha.grad.detach().clone());opt.step()
            self.assertTrue(torch.equal(gradients[0],torch.zeros_like(gradients[0])))
            self.assertTrue(any(bool(torch.count_nonzero(g)) for g in gradients[1:]))
            self.assertTrue(bool(torch.count_nonzero(cal.alpha)))

    def test_cpu_model_counts_common_state_and_config(self):
        from recbole.config import Config
        from recbole.utils import init_seed
        from experiments.mamba3_three_time.model import ThreeTimeMamba3Rec
        from experiments.mamba3_three_time.config import SyntheticCatalog
        from experiments.mamba3_head_timescales.model import HeadTimescaleMamba3Rec
        from experiments.mamba3_head_timescales.state import initial,paired,effective_check,transfer_common
        from experiments.mamba3_three_time.confirmation.state import canonical_optimizer_settings,compare_optimizer_settings
        import tempfile
        rows=[]
        with tempfile.TemporaryDirectory() as directory:
            for variant in c.MODES:
                cfg=Config(model=ThreeTimeMamba3Rec,config_dict=c.settings(variant,'cpu',directory))
                effective_check(cfg,variant)
                init_seed(2026,True)
                net=HeadTimescaleMamba3Rec(cfg,SyntheticCatalog()).cpu()
                self.assertEqual(sum(p.numel() for p in net.parameters()),c.COUNTS[variant])
                self.assertTrue(all(v.device.type=='cpu' for v in list(net.parameters())+list(net.buffers())))
                rows.append(initial(net))
                if variant=='fixed':
                    self.assertIs(type(net.times.calibrators['decay']),TimeCalibrator)
                    fixed=net
                else:transfer_common(fixed,net)
                expected=set(c.plan()['common_parameter_keys'])| (set(c.plan()['alpha_keys']) if variant!='fixed' else set())
                self.assertEqual(set(dict(net.named_parameters())),expected)
                optimizer=torch.optim.Adam(net.parameters(),lr=.001)
                settings=canonical_optimizer_settings([{k:v for k,v in group.items() if k!='params'} for group in optimizer.param_groups])
                compare_optimizer_settings(settings,json.loads(json.dumps(settings)))
                bad=json.loads(json.dumps(settings));bad[0]['lr']=.002
                with self.assertRaises(ValueError):compare_optimizer_settings(settings,bad)
                del optimizer
            paired(rows[0],rows[1]);paired(rows[0],rows[2])
            pilot=json.loads(c.PILOT.read_text())
            self.assertEqual(rows[0]['initial_backbone_sha256'],pilot['initial_backbone_sha256'])
            self.assertEqual(rows[0]['initial_common_calibrator_hashes'],pilot['initial_calibrator_hashes'])
        self.assertFalse(torch.cuda.is_initialized())

    def test_hook_exception_cleanup(self):
        from experiments.mamba3_mimo_time.prefix_checks import check_prefix
        class Broken(torch.nn.Module):
            def __init__(self):
                super().__init__();self.item_embedding=torch.nn.Embedding(8,3)
            def encode_sequence(self,*args,**kwargs):raise RuntimeError('fixture failure')
        net=Broken();data=(torch.ones(2,5,dtype=torch.long),torch.tensor([5,5]),torch.zeros(2,5))
        phases=[]
        with self.assertRaises(RuntimeError):check_prefix(net,data,2,1,None,[],lambda c,p:phases.append(copy.deepcopy(p)))
        self.assertFalse(net.item_embedding._forward_hooks)
        self.assertTrue(phases[-1]['hook_removed'])


if __name__=='__main__':unittest.main()
